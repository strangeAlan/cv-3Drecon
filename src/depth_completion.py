"""Completion of the fused MVS cloud with the feed-forward reconstruction (hybrid module 3).

PatchMatch + stereo fusion is accurate but leaves holes on weak texture; the feed-forward
reconstruction is dense and self-consistent but less accurate. The hybrid output is the fused
MVS cloud plus the feed-forward points that fall into its holes:
  1. per view, every pixel of the feed-forward depth (resampled to the common 1600 px grid,
     distorted image, predicted pinhole intrinsics) is linked to the undistorted PatchMatch
     pixel seeing the same ray (refined distorted camera);
  2. one robust Sim(3) maps the feed-forward reconstruction (its own depth and poses, kept
     together because they are mutually consistent) onto the MVS frame, fitted on pixels where
     both have depth, then refined by similarity ICP onto the fused MVS cloud;
  3. optional guided correction (cfg["guided"]): the remaining depth residual along each refined
     camera ray, measured where PatchMatch has depth, is propagated with a normalized guided
     filter (He et al., grayscale image as guide) and applied to fill points near MVS support;
  4. fill points = transformed (and corrected) feed-forward points whose pixel is not covered by
     the projected fused MVS points.
"""
import cv2
import numpy as np
import open3d as o3d
import pycolmap
from scipy.spatial import cKDTree

from colmap_common import cam_from_world, read_colmap_array
from common import pc_grid, resample_depth, umeyama


def box(x, r):
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), normalize=True, borderType=cv2.BORDER_REFLECT)


def guided_filter(guide, src, r, eps):
    """He et al. 2010, gray guide."""
    mean_i, mean_p = box(guide, r), box(src, r)
    cov_ip = box(guide * src, r) - mean_i * mean_p
    var_i = box(guide * guide, r) - mean_i * mean_i
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return box(a, r) * guide + box(b, r)


def coverage(points, K, E, hw, radius):
    """Pixels hit by the projection of points (in front of the camera), dilated by radius."""
    h, w = hw
    cam = points @ E[:3, :3].T + E[:3, 3]
    cam = cam[cam[:, 2] > 0]
    u = np.floor(cam[:, 0] / cam[:, 2] * K[0, 0] + K[0, 2]).astype(int)
    v = np.floor(cam[:, 1] / cam[:, 2] * K[1, 1] + K[1, 2]).astype(int)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    m = np.zeros(hw, np.uint8)
    m[v[ok], u[ok]] = 1
    return cv2.dilate(m, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)) > 0


def _link_view(ff_depth, ff_K, ff_ext, cam_dist, K_und, E_und, d_mvs, covered):
    """Feed-forward points of one view (FF frame) on the common grid, with the linked undistorted
    PatchMatch pixel: MVS point (MVS frame) where PatchMatch has depth, and hole flags."""
    H, W = cam_dist.height, cam_dist.width
    d, K = resample_depth(ff_depth, ff_K, pc_grid((H, W)))
    h, w = d.shape
    v, u = np.nonzero(d > 0)
    z = d[v, u].astype(np.float64)
    ff_cam = np.stack([(u + 0.5 - K[0, 2]) * z / K[0, 0], (v + 0.5 - K[1, 2]) * z / K[1, 1], z], 1)
    ff_pts = (ff_cam - ff_ext[:3, 3]) @ ff_ext[:3, :3]
    # same pixel on the distorted original image -> ray of the refined camera -> undistorted pixel
    img = np.stack([(u + 0.5) * W / w, (v + 0.5) * H / h], 1)
    xy = np.asarray(cam_dist.cam_from_img(img))[:, :2]
    qu = np.floor(xy[:, 0] * K_und[0, 0] + K_und[0, 2]).astype(int)
    qv = np.floor(xy[:, 1] * K_und[1, 1] + K_und[1, 2]).astype(int)
    hd, wd = d_mvs.shape
    inside = (qu >= 0) & (qu < wd) & (qv >= 0) & (qv < hd)
    qu, qv = np.where(inside, qu, 0), np.where(inside, qv, 0)
    zm = np.where(inside, d_mvs[qv, qu], 0).astype(np.float64)
    mvs_pts = (np.concatenate([xy * zm[:, None], zm[:, None]], 1) - E_und[:3, 3]) @ E_und[:3, :3]
    hole = inside & ~covered[qv, qu]
    return {"grid_hw": (h, w), "v": v, "u": u, "ff_pts": ff_pts, "xy": xy, "zm": zm,
            "mvs_pts": mvs_pts, "has_mvs": zm > 0, "hole": hole}


def completion_points(dense_dir, image_dir, rec_sparse, ff_by_name, fused_xyz, cfg, variants=("fill", "gf")):
    """Fill points for the holes of the fused MVS cloud, in the MVS (COLMAP) world frame, for each
    variant ("fill": aligned feed-forward points, "gf": plus guided depth correction).
    Returns {variant: (points, stats)}.

    ff_by_name: image name -> (depth, intrinsics at the depth grid, world-to-camera 4x4)."""
    rec_und = pycolmap.Reconstruction(dense_dir / "sparse")
    sparse_by_name = {im.name: im for im in rec_sparse.images.values()}
    views = {}
    for im in rec_und.images.values():
        dpath = dense_dir / "stereo" / "depth_maps" / f"{im.name}.geometric.bin"
        if im.name not in ff_by_name or not dpath.exists():
            continue
        d_mvs = read_colmap_array(dpath)
        K_und = rec_und.cameras[im.camera_id].calibration_matrix()
        E_und = cam_from_world(im)
        covered = coverage(fused_xyz, K_und, E_und, d_mvs.shape, cfg["coverage_radius"])
        cam_dist = rec_sparse.cameras[sparse_by_name[im.name].camera_id]
        ff_depth, ff_K, ff_ext = ff_by_name[im.name]
        views[im.name] = _link_view(ff_depth, ff_K, ff_ext, cam_dist, K_und, E_und, d_mvs, covered)
        views[im.name]["E_und"] = E_und

    # global Sim(3): feed-forward frame -> MVS frame, on pixels with both depths
    rng = np.random.default_rng(0)
    P, Q = [], []
    for vw in views.values():
        idx = np.flatnonzero(vw["has_mvs"])
        if len(idx) > cfg["align_samples_per_view"]:
            idx = rng.choice(idx, cfg["align_samples_per_view"], replace=False)
        P.append(vw["ff_pts"][idx])
        Q.append(vw["mvs_pts"][idx])
    P, Q = np.concatenate(P), np.concatenate(Q)
    if len(P) < cfg["min_support_pixels"]:
        empty = (np.empty((0, 3), np.float32), {"skipped": f"only {len(P)} FF/MVS correspondences"})
        return {v: empty for v in variants}
    inl = np.ones(len(P), bool)
    for _ in range(5):
        R, t, s = umeyama(P[inl], Q[inl])
        r = np.linalg.norm(s * P @ R.T + t - Q, axis=1)
        inl = r < 3 * np.median(r)
    align = {"align_n": int(len(P)), "align_inlier_frac": float(inl.mean()),
             "align_resid_median": float(np.median(r[inl])), "align_scale": float(s)}
    # similarity ICP refinement onto the fused MVS cloud, using feed-forward points in covered pixels
    if len(fused_xyz) >= cfg["min_support_pixels"]:
        cov = np.concatenate([vw["ff_pts"][~vw["hole"]] for vw in views.values()])
        if len(cov) > cfg["icp_samples"]:
            cov = cov[rng.choice(len(cov), cfg["icp_samples"], replace=False)]
        ref = fused_xyz.astype(np.float64)
        tree = cKDTree(ref)
        for _ in range(cfg["icp_iters"]):
            X = s * cov @ R.T + t
            d, idx = tree.query(X, k=1, workers=8)
            m = d < 3 * np.median(d)
            if m.sum() < cfg["min_support_pixels"]:
                break
            R2, t2, s2 = umeyama(X[m], ref[idx[m]])
            R, t, s = R2 @ R, s2 * R2 @ t + t2, s2 * s
        align["icp_resid_median"] = float(np.median(d[m]))
    return {v: _fill_points(views, R, t, s, image_dir, cfg, v == "gf", align) for v in variants}


def _fill_points(views, R, t, s, image_dir, cfg, guided, align):
    stats = dict(align, views={})
    pts = []
    for name, vw in views.items():
        X = s * vw["ff_pts"] @ R.T + t
        n_corr = 0
        if guided:
            E = vw["E_und"]
            z_ff = (X @ E[:3, :3].T + E[:3, 3])[:, 2]  # depth along the refined camera
            h, w = vw["grid_hw"]
            gray = cv2.imread(str(image_dir / name), cv2.IMREAD_GRAYSCALE)  # distorted image = FF grid
            gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            resid = np.zeros((h, w), np.float32)
            support = np.zeros((h, w), np.float32)
            m = vw["has_mvs"]
            resid[vw["v"][m], vw["u"][m]] = (vw["zm"][m] - z_ff[m])
            support[vw["v"][m], vw["u"][m]] = 1.0
            r_, eps = cfg["gf_radius"], cfg["gf_eps"]
            m_s = guided_filter(gray, support, r_, eps)
            corr = guided_filter(gray, resid, r_, eps) / np.maximum(m_s, 1e-6)
            corr[m_s < cfg["min_support"]] = 0.0
            c = corr[vw["v"], vw["u"]].astype(np.float64)
            ok = z_ff > 0
            C = -E[:3, :3].T @ E[:3, 3]
            X[ok] = C + (X[ok] - C) * ((z_ff[ok] + c[ok]) / z_ff[ok])[:, None]
            n_corr = int((vw["hole"] & (c != 0)).sum())
        fill = X[vw["hole"]]
        if len(fill):
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(fill)).voxel_down_sample(cfg["voxel"])
            pts.append(np.asarray(pcd.points, np.float32))
        stats["views"][name] = {"hole_frac": float(vw["hole"].mean()), "mvs_frac": float(vw["has_mvs"].mean()),
                                "corrected_frac": n_corr / max(int(vw["hole"].sum()), 1)}
    xyz = np.concatenate(pts) if pts else np.empty((0, 3), np.float32)
    return xyz, stats
