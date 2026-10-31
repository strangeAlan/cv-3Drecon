"""Evaluate every finished case of a run, cache per-case metrics, then write
results/<run_name>/metrics.csv.

Metrics per case:
- pose (DA3 compute_pose, registered views only): auc03/05/15/30
- pose with failures: same AUCs over all view pairs of the manifest, pairs that
  touch an unregistered view count as 180 deg error (only differs for COLMAP)
- recon (DA3 recon_unposed: RANSAC-Umeyama Sim(3) on cameras, TSDF fusion,
  vs. the whole-scene GT mesh): acc, comp, precision, recall, fscore @ 0.25 m.
  VGGT/DA3 go through ETH3D.fuse3d unchanged (distorted images, pinhole
  intrinsics, as in the DA3 benchmark); COLMAP uses fuse_colmap (same steps,
  undistorted depth grid)
- recall_vis / fscore_vis: recall against GT points visible in the case's own views
  (back-projected GT depth), so recall is not dominated by scene coverage; combined
  with the whole-mesh precision
- point-cloud geometry without TSDF: aligned depth maps are back-projected and
  voxelized at 5 mm; accuracy is exact point-to-GT-mesh distance and completeness
  is visible-GT-to-prediction nearest-neighbor distance

Usage: python src/evaluate.py configs/1027-prelim.yaml [--methods colmap vggt da3]
       [--workers 16] [--force]
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path

# A case is the unit of parallelism.  Keep native numeric libraries from creating
# another full thread pool in every spawned worker.
for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

from common import (ETH3D_ROOT, ROOT, case_dir, eth3d_dataset, load_config, pc_grid,  # first: preloads libusb
                    resample_depth, umeyama)

import cv2
import numpy as np
import open3d as o3d
import pycolmap
import torch
from scipy.spatial import cKDTree

import depth_anything_3.bench.datasets.eth3d as da3_eth3d
from depth_anything_3.bench.utils import (align_to_first_camera, calculate_auc_np, compute_pose,
                                          create_tsdf_volume, evaluate_3d_reconstruction, fuse_depth_to_tsdf,
                                          sample_points_from_mesh, se3_to_relative_pose_error)
from depth_anything_3.utils.geometry import as_homogeneous
from depth_anything_3.utils.pose_align import align_poses_umeyama


PC_VOXEL = 0.005
PC_THRESHOLDS = (0.02, 0.05, 0.10, 0.25)
ICP_ITERS = 10
ICP_MAX_DIST = 0.25
ICP_WORKERS = 4
DISTANCE_CHUNK = 1_000_000
_WORKER_DS = None


class EmptyMesh(Exception):
    pass


def strict_sample_points_from_mesh(mesh, num_points):
    """DA3's sampler silently returns uniform random points in [-1, 1]^3 when the mesh
    is empty (this is how open3d 0.20 TSDF failures went unnoticed); raise instead."""
    if len(mesh.triangles) == 0:
        raise EmptyMesh
    return sample_points_from_mesh(mesh, num_points)


da3_eth3d.sample_points_from_mesh = strict_sample_points_from_mesh


def pose_metrics(res, gt_meta, registered):
    pred = torch.from_numpy(as_homogeneous(res["extrinsics"])).double()
    gt = torch.from_numpy(as_homogeneous(gt_meta["extrinsics"])).double()
    out = {f"pose_{k}": float(v) for k, v in compute_pose(pred, gt).items()}
    # failures counted: errors of registered pairs + 180 deg for every other pair
    n_all = len(registered)
    n_pairs_all = n_all * (n_all - 1) // 2
    r, t = se3_to_relative_pose_error(align_to_first_camera(pred), align_to_first_camera(gt), len(pred))
    r, t = r.cpu().numpy(), t.cpu().numpy()
    pad = np.full(n_pairs_all - len(r), 180.0)
    r, t = np.concatenate([r, pad]), np.concatenate([t, pad])
    for th in (3, 5, 15, 30):
        out[f"posefail_auc{th:02d}"] = float(calculate_auc_np(r, t, max_threshold=th)[0])
    return out


def gt_cameras(scene):
    """Full GT camera models (incl. THIN_PRISM_FISHEYE distortion) keyed by image basename."""
    calib = ETH3D_ROOT / scene / "dslr_calibration_jpg"
    cams = {}
    for line in open(calib / "cameras.txt"):
        if line.strip() and not line.startswith("#"):
            p = line.split()
            cams[p[0]] = pycolmap.Camera(model=p[1], width=int(p[2]), height=int(p[3]),
                                         params=[float(x) for x in p[4:]])
    lines = [l for l in open(calib / "images.txt").read().splitlines()[4:]]
    out = {}
    for i, line in enumerate(lines):
        p = line.split()
        if i % 2 == 0 and len(p) >= 10:
            out[os.path.basename(p[9])] = cams[p[8]]
    return out


def pixel_rays(cam, u, v):
    """Normalized camera coordinates (x, y) of pixel indices (u, v) for a COLMAP camera."""
    img = np.stack([u + 0.5, v + 0.5], 1).astype(np.float64)  # COLMAP: pixel centers at +0.5
    return np.asarray(cam.cam_from_img(img))[:, :2]


def _voxelize(points, voxel):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)))
    return pcd.voxel_down_sample(voxel)


def visible_gt_points(ds, scene, image_files, voxel, stride=4):
    """Back-project GT depth of the given views into world points (GT poses, GT distortion)."""
    data = ds.get_data(scene)
    by_name = {os.path.basename(p): i for i, p in enumerate(data.image_files)}
    cams = gt_cameras(scene)
    pts = []
    for p in image_files:
        name = os.path.basename(p)
        i = by_name[name]
        h, w = int(data.aux.heights[i]), int(data.aux.widths[i])
        dpath = ETH3D_ROOT / scene / "ground_truth_depth" / "dslr_images" / name
        if not dpath.exists():
            return None
        depth = np.fromfile(dpath, dtype=np.float32).reshape(h, w)
        valid = ds._load_gt_mask(scene, name, (h, w)) & np.isfinite(depth) & (depth > 0)
        v, u = np.nonzero(valid[::stride, ::stride])
        v, u = v * stride, u * stride
        z = depth[v, u]
        xy = pixel_rays(cams[name], u, v)
        cam = np.concatenate([xy * z[:, None], z[:, None]], 1)
        E = data.extrinsics[i]
        pts.append(np.asarray(_voxelize((cam - E[:3, 3]) @ E[:3, :3], voxel).points))  # R^T (x - t)
    return _voxelize(np.concatenate(pts), voxel)


def fuse_colmap(ds, scene, res, gt_meta, info, fuse_path):
    """DA3 recon_unposed fusion for COLMAP depth on the undistorted grid.

    Same RANSAC-Umeyama alignment, GT-mask rule, TSDF settings and mesh sampling as
    ETH3D.fuse3d; the only difference is that the GT mask (distorted grid) is
    looked up through COLMAP's own estimated distorted camera.
    """
    _, _, scale, ext = align_poses_umeyama(gt_meta["extrinsics"].copy(), as_homogeneous(res["extrinsics"]).copy(),
                                           return_aligned=True, ransac=True, random_state=42)
    depths = []
    for i, path in enumerate(gt_meta["image_files"]):
        name = os.path.basename(str(path))
        d = res["depth"][i].copy()
        hd, wd = d.shape
        K = res["intrinsics"][i]
        c = info["colmap_cameras"][i]
        cam = pycolmap.Camera(model=c["model"], width=c["width"], height=c["height"], params=c["params"])
        mask = ds._load_gt_mask(scene, name, (c["height"], c["width"]))
        v, u = np.mgrid[0:hd, 0:wd]
        x = ((u.ravel() + 0.5) - K[0, 2]) / K[0, 0]
        y = ((v.ravel() + 0.5) - K[1, 2]) / K[1, 1]
        uv = np.asarray(cam.img_from_cam(np.stack([x, y, np.ones_like(x)], 1)))
        uu, vv = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
        inside = (uu >= 0) & (uu < c["width"]) & (vv >= 0) & (vv < c["height"])
        keep = np.zeros(hd * wd, bool)
        keep[inside] = mask[vv[inside], uu[inside]]
        d = d * keep.reshape(hd, wd)
        d[~np.isfinite(d)] = 0
        depths.append(d * scale)
    depths = np.stack(depths)
    gray = np.full((*depths.shape, 3), 128, np.uint8)  # color does not enter the metrics
    volume = create_tsdf_volume(voxel_length=ds.voxel_length, sdf_trunc=ds.sdf_trunc)
    mesh = fuse_depth_to_tsdf(volume, depths, gray, res["intrinsics"], ext, max_depth=ds.max_depth)
    pcd = strict_sample_points_from_mesh(mesh, ds.sampling_number)
    os.makedirs(os.path.dirname(fuse_path), exist_ok=True)
    o3d.io.write_point_cloud(str(fuse_path), pcd)


def _views(ds, scene, res, gt_meta, info):
    """Per registered view, on the common grid: predicted depth and intrinsics, the method-frame
    world-to-camera pose, the distorted original pixel every grid pixel maps to, and whether that
    pixel has GT (DA3 GT-mask rule).

    VGGT/DA3 depth lives on the distorted image (pinhole intrinsics, as in the DA3 benchmark);
    COLMAP/hybrid depth lives on the undistorted grid and is mapped through its own estimated
    distorted camera."""
    data = ds.get_data(scene)
    by_name = {os.path.basename(p): i for i, p in enumerate(data.image_files)}
    colmap_like = info["method"].startswith(("colmap", "hybrid"))
    views = []
    for i, path in enumerate(gt_meta["image_files"]):
        name = os.path.basename(str(path))
        j = by_name[name]
        H, W = int(data.aux.heights[j]), int(data.aux.widths[j])
        if colmap_like:
            c = info["colmap_cameras"][i]
            depth, K = resample_depth(res["depth"][i], res["intrinsics"][i], pc_grid((c["height"], c["width"])))
            cam = pycolmap.Camera(model=c["model"], width=c["width"], height=c["height"], params=c["params"])
            h, w = depth.shape
            v, u = np.mgrid[0:h, 0:w]
            x = ((u.ravel() + 0.5) - K[0, 2]) / K[0, 0]
            y = ((v.ravel() + 0.5) - K[1, 2]) / K[1, 1]
            uv = np.asarray(cam.img_from_cam(np.stack([x, y, np.ones_like(x)], 1)))
            src_u = np.floor(uv[:, 0]).reshape(h, w)
            src_v = np.floor(uv[:, 1]).reshape(h, w)
        else:
            depth, K = resample_depth(res["depth"][i], res["intrinsics"][i], pc_grid((H, W)))
            h, w = depth.shape
            src_v, src_u = np.meshgrid(np.floor((np.arange(h) + 0.5) * H / h),
                                       np.floor((np.arange(w) + 0.5) * W / w), indexing="ij")
        inside = (src_u >= 0) & (src_u < W) & (src_v >= 0) & (src_v < H)
        src_u = np.where(inside, src_u, 0).astype(np.int32)
        src_v = np.where(inside, src_v, 0).astype(np.int32)
        gt_ok = inside & ds._load_gt_mask(scene, name, (H, W))[src_v, src_u]
        views.append({"name": name, "j": j, "hw": (H, W), "depth": depth, "K": K,
                      "ext": as_homogeneous(res["extrinsics"][i]).astype(np.float64),
                      "src_u": src_u, "src_v": src_v, "gt_ok": gt_ok})
    return views


def backproject(depth, K, ext, v=None, u=None):
    """Pinhole z-depth (pixel centers at +0.5, COLMAP convention) -> world points."""
    if v is None:
        v, u = np.nonzero(depth > 0)
    z = depth[v, u].astype(np.float64)
    cam = np.stack([(u + 0.5 - K[0, 2]) * z / K[0, 0], (v + 0.5 - K[1, 2]) * z / K[1, 1], z], 1)
    return (cam - ext[:3, 3]) @ ext[:3, :3]


def align_sim3_points(ds, scene, views, samples_per_view=20000, iters=5):
    """Robust Sim(3) from per-pixel correspondences: predicted 3D point vs. the GT-depth 3D point
    of the same (distorted) pixel, iteratively trimmed at 3x the median residual.

    Camera-center alignment (DA3 benchmark) is ill-conditioned when the input cameras are close
    together relative to the scene depth (neighbor subsets), so geometry is aligned on geometry."""
    data = ds.get_data(scene)
    cams = gt_cameras(scene)
    rng = np.random.default_rng(0)
    P, Q = [], []
    for vw in views:
        H, W = vw["hw"]
        gt_depth = np.fromfile(ETH3D_ROOT / scene / "ground_truth_depth" / "dslr_images" / vw["name"],
                               dtype=np.float32).reshape(H, W)
        v, u = np.nonzero((vw["depth"] > 0) & vw["gt_ok"])
        su, sv = vw["src_u"][v, u], vw["src_v"][v, u]
        zg = gt_depth[sv, su]
        good = np.isfinite(zg) & (zg > 0)
        v, u, su, sv, zg = v[good], u[good], su[good], sv[good], zg[good].astype(np.float64)
        if len(v) > samples_per_view:
            idx = np.sort(rng.choice(len(v), samples_per_view, replace=False))
            v, u, su, sv, zg = v[idx], u[idx], su[idx], sv[idx], zg[idx]
        if len(v) == 0:
            continue
        P.append(backproject(vw["depth"], vw["K"], vw["ext"], v, u))
        cam = np.concatenate([pixel_rays(cams[vw["name"]], su, sv) * zg[:, None], zg[:, None]], 1)
        E = data.extrinsics[vw["j"]]
        Q.append((cam - E[:3, 3]) @ E[:3, :3])
    if not P or sum(len(p) for p in P) < 100:
        return None
    P, Q = np.concatenate(P), np.concatenate(Q)
    inl = np.ones(len(P), bool)
    for _ in range(iters):
        R, t, s = umeyama(P[inl], Q[inl])
        r = np.linalg.norm(s * P @ R.T + t - Q, axis=1)
        inl = r < 3 * np.median(r)
    return R, t, s, {"align_n": int(len(P)), "align_inlier_frac": float(inl.mean()),
                     "align_resid_median": float(np.median(r[inl]))}


def fused_points_in_gt_region(ds, scene, gt_meta, pts):
    """Keep (already aligned) fused points that project (GT pose, GT distorted camera) onto a
    GT-valid pixel of at least one of the case's views."""
    data = ds.get_data(scene)
    by_name = {os.path.basename(p): i for i, p in enumerate(data.image_files)}
    cams = gt_cameras(scene)
    keep = np.zeros(len(pts), bool)
    for path in gt_meta["image_files"]:
        name = os.path.basename(str(path))
        E = data.extrinsics[by_name[name]]
        cam_pts = pts @ E[:3, :3].T + E[:3, 3]
        front = cam_pts[:, 2] > 0
        if not front.any():
            continue
        c = cams[name]
        uv = np.asarray(c.img_from_cam(cam_pts[front]))
        uu, vv = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
        inside = (uu >= 0) & (uu < c.width) & (vv >= 0) & (vv < c.height)
        mask = ds._load_gt_mask(scene, name, (c.height, c.width))
        hit = np.zeros(front.sum(), bool)
        hit[inside] = mask[vv[inside], uu[inside]]
        keep[np.flatnonzero(front)[hit]] = True
    return pts[keep]


def prediction_points(ds, scene, res, gt_meta, info, cdir=None, voxel=PC_VOXEL):
    """Each method's standard dense output, Sim(3)-aligned on per-pixel GT correspondences,
    restricted to the GT region and voxelized. Returns (points, alignment diagnostics).

    VGGT/DA3: per-view depth maps on the common grid, back-projected.
    COLMAP/hybrid: the stereo-fusion point cloud when present (runs from 10/27 on), else
    the per-view geometric depth maps (which carry outliers that fusion would remove)."""
    views = _views(ds, scene, res, gt_meta, info)
    aligned = align_sim3_points(ds, scene, views)
    if aligned is None:
        return np.empty((0, 3)), {"align_failed": True}
    R, t, s, diag = aligned
    fused_path = None if cdir is None else Path(cdir) / "exports" / "fused_points.npy"
    if info["method"].startswith(("colmap", "hybrid")) and fused_path is not None and fused_path.exists():
        raw = np.load(fused_path)
        fill_path = fused_path.with_name("fill_points.npy")
        if fill_path.exists():  # hybrid with depth completion: MVS cloud + feed-forward hole fills
            raw = np.concatenate([raw, np.load(fill_path)])
        pts = s * raw.astype(np.float64) @ R.T + t
        pts = fused_points_in_gt_region(ds, scene, gt_meta, pts)
        return (np.asarray(_voxelize(pts, voxel).points) if len(pts) else np.empty((0, 3))), diag
    pts = []
    for vw in views:
        p = backproject(vw["depth"] * vw["gt_ok"], vw["K"], vw["ext"])
        if len(p):
            pts.append(np.asarray(_voxelize(s * p @ R.T + t, voxel).points))
    return (np.asarray(_voxelize(np.concatenate(pts), voxel).points) if pts else np.empty((0, 3))), diag


GT_CLOUD_DIR = ETH3D_ROOT.resolve().parent / "eth3d_gtcloud"


def gt_reference_cloud(ds, scene, stride=2):
    """Whole-scene GT surface: GT depth of every view back-projected (stride 2, 5 mm voxels),
    cached on disk. Used instead of combined_mesh.ply, which misses much of the scanned
    surface (only 6% of courtyard/DSC_0307 GT-depth rays hit it)."""
    path = GT_CLOUD_DIR / f"{scene}_s{stride}_v{int(PC_VOXEL * 1000)}mm.npy"
    if path.exists():
        return np.load(path)
    pts = np.asarray(visible_gt_points(ds, scene, ds.get_data(scene).image_files, PC_VOXEL, stride=stride).points)
    GT_CLOUD_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.npy")
    np.save(tmp, pts.astype(np.float32))
    os.replace(tmp, path)
    return pts


def _nearest_distances(reference, query):
    if len(reference) == 0 or len(query) == 0:
        return np.empty(0, dtype=np.float64)
    tree = cKDTree(reference, balanced_tree=False, compact_nodes=False)
    distances = []
    for start in range(0, len(query), DISTANCE_CHUNK):
        distance, _ = tree.query(query[start:start + DISTANCE_CHUNK], k=1, workers=1)
        distances.append(distance)
    return np.concatenate(distances)


def icp_sim3(points, tree, reference, iters=ICP_ITERS, max_dist=ICP_MAX_DIST):
    """Similarity ICP of the final point cloud onto the GT cloud (refinement step of the Tanks and
    Temples protocol): pairs closer than min(3x median distance, max_dist), Umeyama, repeat.
    Every method gets the same refinement on its own final output."""
    if len(points) < 100:
        return points, {}
    R, s = np.eye(3), 1.0
    for _ in range(iters):
        d, idx = tree.query(points, k=1, workers=ICP_WORKERS)
        m = d < min(3 * np.median(d), max_dist)
        if m.sum() < 100:
            break
        R, t, s = umeyama(points[m], reference[idx[m]])
        points = s * points @ R.T + t
    return points, {"icp_last_rot_deg": float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))),
                    "icp_last_scale": float(s)}


def point_cloud_metrics_from_points(reference_points, pred_points, visible_points):
    """Centimeter-scale metrics: accuracy = pred -> whole-scene GT cloud, completeness =
    GT visible in the case's views -> pred. The prediction is first refined onto the GT cloud
    with similarity ICP."""
    reference_points = np.asarray(reference_points, dtype=np.float64)
    pred_points = np.asarray(pred_points, dtype=np.float64)
    visible_points = np.asarray(visible_points, dtype=np.float64)
    icp = {}
    if len(pred_points):
        tree = cKDTree(reference_points, balanced_tree=False, compact_nodes=False)
        pred_points, icp = icp_sim3(pred_points, tree, reference_points)
    accuracy = _nearest_distances(reference_points, pred_points)
    completeness = _nearest_distances(pred_points, visible_points)
    out = {
        **icp,
        "acc_pc_median": float(np.median(accuracy)) if len(accuracy) else float("inf"),
        "comp_pc_median": float(np.median(completeness)) if len(completeness) else float("inf"),
    }
    for threshold in PC_THRESHOLDS:
        suffix = f"{threshold:.2f}"
        precision = float(np.mean(accuracy < threshold)) if len(accuracy) else 0.0
        recall = float(np.mean(completeness < threshold)) if len(completeness) else 0.0
        fscore = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        out[f"precision_pc@{suffix}"] = precision
        out[f"recall_pc@{suffix}"] = recall
        out[f"fscore_pc@{suffix}"] = fscore
    return out


def empty_point_cloud_metrics():
    out = {"acc_pc_median": float("inf"), "comp_pc_median": float("inf")}
    for threshold in PC_THRESHOLDS:
        suffix = f"{threshold:.2f}"
        out.update({f"precision_pc@{suffix}": 0.0, f"recall_pc@{suffix}": 0.0,
                    f"fscore_pc@{suffix}": 0.0})
    return out


def point_cloud_metrics(ds, scene, res, gt_meta, info, cdir=None):
    pred_points, diag = prediction_points(ds, scene, res, gt_meta, info, cdir)
    visible = visible_gt_points(ds, scene, list(gt_meta["image_files"]), PC_VOXEL)
    visible_points = np.asarray(visible.points) if visible is not None else np.empty((0, 3))
    out = point_cloud_metrics_from_points(gt_reference_cloud(ds, scene), pred_points, visible_points)
    out.update(diag)
    return out


def eval_case(ds, cdir, scene, da3_protocol=True):
    info = json.load(open(cdir / "info.json"))
    row = {"method": info["method"], "scene": scene, "n_views": info["case"]["n_views"],
           "sampling": info["case"].get("sampling", "random"), "seed": info["case"]["seed"], "n_input": len(info["registered"]),
           "n_registered": int(sum(info["registered"])),
           "time_s": round(sum(info["time_s"].values()), 2)}
    if info.get("failed"):
        row.update({"posefail_auc03": 0.0, "posefail_auc05": 0.0, "posefail_auc15": 0.0, "posefail_auc30": 0.0,
                    "fscore": 0.0, "fscore_vis": 0.0})
        row.update(empty_point_cloud_metrics())
        return row
    res_path = cdir / "exports" / "mini_npz" / "results.npz"
    res = dict(np.load(res_path))
    gt_meta = dict(np.load(cdir / "exports" / "gt_meta.npz", allow_pickle=True))
    row.update(pose_metrics(res, gt_meta, info["registered"]))
    if len(res["extrinsics"]) < 3:
        # Sim(3) alignment to GT needs >= 3 camera centers; reconstruction counts as failed
        row.update({"too_few_views": True, "fscore": 0.0, "fscore_vis": 0.0})
        row.update(empty_point_cloud_metrics())
        return row

    fuse_path = cdir / "exports" / "fuse" / "pcd.ply"
    fused = da3_protocol  # DA3-benchmark columns (TSDF fusion at full resolution, slow): repro check only
    if fused and not fuse_path.exists():
        try:
            if info["method"].startswith(("colmap", "hybrid")):  # undistorted-grid depth
                fuse_colmap(ds, scene, res, gt_meta, info, fuse_path)
            else:
                ds.fuse3d(scene, str(res_path), str(fuse_path), "recon_unposed")
        except EmptyMesh:
            row.update({"empty_mesh": True, "fscore": 0.0, "fscore_vis": 0.0})
            fused = False
    if fused:
        row.update(ds.eval3d(scene, str(fuse_path)))

        # recall restricted to GT surface visible in this case's views; precision stays the
        # whole-mesh one (GT depth covers only scanned pixels, so it cannot judge precision)
        vis = visible_gt_points(ds, scene, list(gt_meta["image_files"]), ds.down_sample)
        if vis is not None:
            m = evaluate_3d_reconstruction(o3d.io.read_point_cloud(str(fuse_path)), vis,
                                           threshold=ds.eval_threshold, down_sample=ds.down_sample)
            p, r = row["precision"], m["recall"]
            row.update({"recall_vis": r, "fscore_vis": 2 * p * r / (p + r) if p + r > 0 else 0.0})
    row.update(point_cloud_metrics(ds, scene, res, gt_meta, info, cdir))
    return row


def _init_worker():
    global _WORKER_DS
    _WORKER_DS = eth3d_dataset()


def _eval_case_task(task):
    index, cdir_string, scene, force, da3_protocol = task
    cdir = Path(cdir_string)
    metrics_path = cdir / "metrics.json"
    if metrics_path.exists() and not force:
        with open(metrics_path) as f:
            return index, json.load(f), True

    row = eval_case(_WORKER_DS, cdir, scene, da3_protocol)
    temporary = metrics_path.with_name(f".{metrics_path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "w") as f:
            json.dump(row, f, indent=2)
        os.replace(temporary, metrics_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return index, row, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--methods", nargs="*", default=["colmap", "vggt", "da3"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-da3-protocol", action="store_true",
                    help="skip the DA3-benchmark columns (TSDF fusion); they are only needed for the repro check")
    args = ap.parse_args()
    if args.workers < 1:
        ap.error("--workers must be at least 1")
    cfg = load_config(args.config)
    run_dir = ROOT / "runs" / cfg["run_name"]
    cases = json.load(open(ROOT / "manifests" / f"{cfg['run_name']}.json"))["cases"]

    tasks = []
    for method in args.methods:
        for case in cases:
            cdir = case_dir(run_dir, method, case)
            if not (cdir / "info.json").exists():
                continue
            tasks.append((len(tasks), str(cdir), case["scene"], args.force, not args.no_da3_protocol))

    by_index = {}
    if tasks:
        context = mp.get_context("spawn")
        processes = min(args.workers, len(tasks))
        with context.Pool(processes=processes, initializer=_init_worker) as pool:
            for index, row, cached in pool.imap_unordered(_eval_case_task, tasks):
                by_index[index] = row
                shown = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}
                print(f"{'cached' if cached else 'computed'} {shown}", flush=True)
    rows = [by_index[i] for i in range(len(tasks))]

    out_dir = ROOT / "results" / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {out_dir / 'metrics.csv'}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
