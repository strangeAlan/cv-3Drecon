"""DA3-initialised bundle adjustment (DA3-BA): DA3 cameras and depth refined with SIFT tracks.

The 10/27 hybrid triangulated SIFT matches with the DA3 cameras and ran BA on the result. DA3's
pose errors of a few degrees are hundreds of pixels on 6048 px images, so COLMAP's triangulation
rejected almost every match (~300 points per case) and BA made the cameras worse than DA3's.
Here no triangulation is needed to start: DA3 predicts a depth for every pixel, so every SIFT
keypoint is lifted to 3D directly.

Per case (same images as all other methods):
  1. SIFT + exhaustive matching with geometric verification (COLMAP baseline settings)
  2. tracks = connected components of the verified inlier matches (tracks that see an image twice
     are dropped); every observation is lifted to 3D with the DA3 depth, camera and pose, the
     track point is the median, observations farther than `lift_tol` x depth from it are dropped
  3. sparse model with the DA3 cameras (shared focal length per COLMAP camera, distortion 0). Camera
     model RADIAL (k1, k2): COLMAP's default SIMPLE_RADIAL cannot represent the ETH3D lenses (fit to the
     GT model: 11 px RMS, 61 px at the corners, focal 2% short; RADIAL ~1-1.6 px RMS), and BA then
     trades the model error for biased focal lengths and poses
  4. coarse-to-fine BA: rounds of robust (Cauchy) BA over poses, focal length, distortion and points,
     each followed by removal of observations with a reprojection error above the round's threshold
  5. COLMAP triangulation with the refined cameras (complete / merge tracks) and a final BA round
  6. views with fewer than `min_obs` observations keep their DA3 pose, mapped into the BA frame by
     the similarity between the DA3 and refined camera centres of the other views; with fewer than
     3 well-constrained views the whole case falls back to DA3
  7. dense output: DA3 depth maps with the refined cameras, either rescaled by one global factor
     (da3ba) or per view to the depth of the view's SfM points (da3ba_ds)

  8. --dense: COLMAP's dense stage (PatchMatch + fusion) on the refined cameras, holes of the fused cloud
     filled with the da3ba_ds depth

Outputs: runs/<run>/da3ba and runs/<run>/da3ba_ds in the common case format (feed-forward style:
depth on the DA3 grid, pinhole intrinsics), evaluated by evaluate.py like VGGT/DA3; with --dense
also hybrid_da3ba (fused MVS cloud) and hybrid_da3ba_fill (MVS + DA3-BA fill), COLMAP style.

Ablation: da3ba.init: triangulate starts from COLMAP's triangulation with the DA3 cameras instead of lifting.
With da3ba.ff_method: vggt the same refinement runs on VGGT's predictions (outputs vggtba, vggtba_ds).

Usage: python src/run_da3ba.py configs/<run>.yaml [--shard i/n] [--dense]
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pycolmap

from colmap_common import dense, distorted_cameras, export_case, extract_and_match, shard_filter
from depth_completion import completion_points
from common import ROOT, GpuMemSampler, case_dir, case_gt, eth3d_dataset, is_done, load_config, save_case, umeyama


def read_db(db_path):
    db = pycolmap.Database.open(db_path)
    images = {im.name: im for im in db.read_all_images()}
    cameras = {c.camera_id: c for c in db.read_all_cameras()}
    keypoints = {im.image_id: db.read_keypoints(im.image_id)[:, :2].astype(np.float64) for im in images.values()}
    pair_ids, geoms = db.read_two_view_geometries()
    matches = []
    for pid, g in zip(pair_ids, geoms):
        if len(g.inlier_matches):
            a, b = pycolmap.pair_id_to_image_pair(pid)
            matches.append((a, b, np.asarray(g.inlier_matches)))
    db.close()
    return images, cameras, keypoints, matches


def build_tracks(matches):
    """Union-find over (image_id, keypoint index); drop tracks with two keypoints in one image."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, m in matches:
        for i, j in m:
            ra, rb = find((a, int(i))), find((b, int(j)))
            if ra != rb:
                parent[ra] = rb
    groups = {}
    for node in parent:
        groups.setdefault(find(node), []).append(node)
    tracks = []
    for g in groups.values():
        imgs = [n[0] for n in g]
        if len(g) >= 2 and len(set(imgs)) == len(imgs):
            tracks.append(sorted(g))
    return tracks


def lift(tracks, keypoints, ff, view_of, cameras_wh, lift_tol):
    """3D point per track from the DA3 depth of its observations (median), inconsistent observations
    removed. view_of: image_id -> index into the DA3 arrays; cameras_wh: image_id -> (W, H)."""
    depth, K, E = ff["depth"], ff["intrinsics"].astype(np.float64), ff["extrinsics"].astype(np.float64)
    h, w = depth.shape[1:]
    # lift every keypoint once
    world = {}
    for img_id, kp in keypoints.items():
        v = view_of[img_id]
        W, H = cameras_wh[img_id]
        u, vv = kp[:, 0] * w / W, kp[:, 1] * h / H  # continuous, corner origin
        iu, iv = np.clip(u.astype(int), 0, w - 1), np.clip(vv.astype(int), 0, h - 1)
        z = depth[v][iv, iu].astype(np.float64)
        cam = np.stack([(u - K[v, 0, 2]) / K[v, 0, 0] * z, (vv - K[v, 1, 2]) / K[v, 1, 1] * z, z], 1)
        p = (cam - E[v, :3, 3]) @ E[v, :3, :3]
        p[~(np.isfinite(z) & (z > 0))] = np.nan
        world[img_id] = (p, z)
    out = []
    for t in tracks:
        P = np.array([world[i][0][k] for i, k in t])
        Z = np.array([world[i][1][k] for i, k in t])
        ok = np.isfinite(P).all(1)
        if ok.sum() < 2:
            continue
        med = np.median(P[ok], 0)
        keep = ok & (np.linalg.norm(P - med, axis=1) < lift_tol * Z)
        if keep.sum() >= 2:
            out.append((np.median(P[keep], 0), [t[k] for k in np.flatnonzero(keep)]))
    return out


def init_reconstruction(images, cameras, keypoints, ff, names):
    """DA3 cameras (intrinsics scaled to the original images, shared per COLMAP camera) and poses."""
    h, w = ff["depth"].shape[1:]
    K = ff["intrinsics"].astype(np.float64).copy()
    for i, n in enumerate(names):
        cam = cameras[images[n].camera_id]
        K[i, 0] *= cam.width / w
        K[i, 1] *= cam.height / h
    rec = pycolmap.Reconstruction()
    idx_by_cam = {}
    for i, n in enumerate(names):
        idx_by_cam.setdefault(images[n].camera_id, []).append(i)
    for cam_id, idx in idx_by_cam.items():
        cam = cameras[cam_id]
        f = np.median((K[idx, 0, 0] + K[idx, 1, 1]) / 2)
        cx, cy = np.median(K[idx, 0, 2]), np.median(K[idx, 1, 2])
        cam.params = {"SIMPLE_RADIAL": [f, cx, cy, 0.0], "RADIAL": [f, cx, cy, 0.0, 0.0],
                      "OPENCV": [f, f, cx, cy, 0.0, 0.0, 0.0, 0.0]}[cam.model.name]  # distortion 0 = DA3 pinhole
        rec.add_camera_with_trivial_rig(cam)
    for i, n in enumerate(names):
        dbi = images[n]
        pts = pycolmap.Point2DList([pycolmap.Point2D(xy) for xy in keypoints[dbi.image_id]])
        image = pycolmap.Image(name=n, points2D=pts, camera_id=dbi.camera_id, image_id=dbi.image_id)
        rec.add_image_with_trivial_frame(image, pycolmap.Rigid3d(ff["extrinsics"][i, :3, :4].astype(np.float64)))
    return rec


def observations(rec):
    """(image_id, point2D_idx, point3D_id, reprojection error in px) of every observation."""
    rows = []
    for img_id, im in rec.images.items():
        cam = rec.cameras[im.camera_id]
        T = im.cam_from_world()
        idx = [k for k, p in enumerate(im.points2D) if p.has_point3D()]
        if not idx:
            continue
        pids = np.array([im.points2D[k].point3D_id for k in idx])
        xyz = np.array([rec.points3D[p].xyz for p in pids])
        xy = np.array([im.points2D[k].xy for k in idx])
        c = xyz @ T.rotation.matrix().T + T.translation
        err = np.full(len(idx), np.inf)
        front = c[:, 2] > 1e-6
        if front.any():
            err[front] = np.linalg.norm(np.asarray(cam.img_from_cam(c[front])) - xy[front], axis=1)
        rows += list(zip([img_id] * len(idx), idx, pids, err))
    return rows


def ba_round(rec, loss_scale, max_err, refine_focal=True, refine_distortion=True):
    """One robust BA round, then removal of observations above max_err."""
    opts = pycolmap.BundleAdjustmentOptions()
    opts.refine_focal_length = refine_focal
    opts.refine_extra_params = refine_distortion
    opts.refine_principal_point = False
    opts.print_summary = False
    opts.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY
    opts.ceres.loss_function_scale = loss_scale
    pycolmap.bundle_adjustment(rec, opts)
    removed = 0
    for img_id, k, pid, err in observations(rec):
        if err > max_err and rec.exists_point3D(pid):
            rec.delete_observation(img_id, k)
            removed += 1
    return removed


def sim3_pose(E, R, t, s):
    """World-to-camera pose E (DA3 frame) in the frame x' = s R x + t."""
    out = np.eye(4)
    out[:3, :3] = E[:3, :3] @ R.T
    out[:3, 3] = s * E[:3, 3] - out[:3, :3] @ t
    return out


def centers(E):
    return -np.einsum("nji,nj->ni", E[:, :3, :3], E[:, :3, 3])


def run_case(work, case, gt, ff, ccfg, bcfg):
    image_dir = Path(gt["image_files"][0]).parent
    names = [Path(p).name for p in gt["image_files"]]
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    db = work / "database.db"
    times, log = {}, {}
    t0 = time.time()
    extract_and_match(db, image_dir, names, ccfg)
    times["features_matching"] = time.time() - t0

    t0 = time.time()
    images, cameras, keypoints, matches = read_db(db)
    view_of = {images[n].image_id: i for i, n in enumerate(names)}
    wh = {images[n].image_id: (cameras[images[n].camera_id].width, cameras[images[n].camera_id].height) for n in names}
    tracks = build_tracks(matches)
    rec = init_reconstruction(images, cameras, keypoints, ff, names)
    if bcfg.get("init", "lift") == "lift":
        lifted = lift(tracks, keypoints, ff, view_of, wh, bcfg["lift_tol"])
        for xyz, track in lifted:
            rec.add_point3D(xyz, pycolmap.Track([pycolmap.TrackElement(i, k) for i, k in track]))
    else:  # ablation "triangulate": COLMAP triangulation with the DA3 cameras (the 10/27 hybrid's start)
        rec = pycolmap.triangulate_points(rec, db, image_dir, work / "tri0", clear_points=True)
    n_init = rec.num_points3D()
    log.update(n_tracks=len(tracks), n_points_init=n_init,
               init_err_median=float(np.median([o[3] for o in observations(rec)])) if n_init else None)
    ok = n_init >= bcfg["min_points"]
    intr = (bcfg.get("refine_focal", True), bcfg.get("refine_distortion", True))
    if ok:
        rounds = []
        for loss_scale, max_err in bcfg["rounds"]:
            removed = ba_round(rec, loss_scale, max_err, *intr)
            rounds.append({"loss_scale": loss_scale, "max_err": max_err, "removed": removed,
                           "points": rec.num_points3D()})
        rec = pycolmap.triangulate_points(rec, db, image_dir, work / "tri", clear_points=False)
        for loss_scale, max_err in bcfg["final_rounds"]:
            removed = ba_round(rec, loss_scale, max_err, *intr)
            rounds.append({"loss_scale": loss_scale, "max_err": max_err, "removed": removed,
                           "points": rec.num_points3D(), "after_triangulation": True})
        log["rounds"] = rounds
        log["final_err_median"] = float(np.median([o[3] for o in observations(rec)])) if rec.num_points3D() else None
    times["ba"] = time.time() - t0

    # refined poses; weak views keep their DA3 pose, mapped into the BA frame
    E_ff = np.tile(np.eye(4), (len(names), 1, 1))
    E_ff[:, :3] = ff["extrinsics"][:, :3]
    n_obs = np.array([rec.images[images[n].image_id].num_points3D if ok else 0 for n in names])
    strong = n_obs >= bcfg["min_obs"]
    log["n_obs_per_view"] = n_obs.tolist()
    if strong.sum() < 3:
        log["fallback"] = "all views (fewer than 3 well-constrained views)"
        return E_ff, ff["intrinsics"].astype(np.float64), np.ones(len(names)), np.zeros(len(names), bool), times, log, None
    E_ba = np.tile(np.eye(4), (len(names), 1, 1))
    for i, n in enumerate(names):
        T = rec.images[images[n].image_id].cam_from_world()
        E_ba[i, :3, :3], E_ba[i, :3, 3] = T.rotation.matrix(), T.translation
    R, t, s = umeyama(centers(E_ff[strong]), centers(E_ba[strong]))
    for i in np.flatnonzero(~strong):
        E_ba[i] = sim3_pose(E_ff[i], R, t, s)
        im = rec.images[images[names[i]].image_id]  # trivial rig: camera pose = frame pose
        rec.frames[im.frame_id].rig_from_world = pycolmap.Rigid3d(E_ba[i, :3, :4])

    # intrinsics back on the DA3 depth grid; per-view depth scale from the view's SfM points
    h, w = ff["depth"].shape[1:]
    K = ff["intrinsics"].astype(np.float64).copy()
    ratios = np.full(len(names), np.nan)
    for i, n in enumerate(names):
        im = rec.images[images[n].image_id]
        cam = rec.cameras[im.camera_id]
        f, cx, cy = cam.params[:3]
        K[i] = [[f * w / cam.width, 0, cx * w / cam.width], [0, f * h / cam.height, cy * h / cam.height], [0, 0, 1]]
        if not strong[i]:
            continue
        T = im.cam_from_world()
        idx = [k for k, p in enumerate(im.points2D) if p.has_point3D()]
        xyz = np.array([rec.points3D[im.points2D[k].point3D_id].xyz for k in idx])
        xy = np.array([im.points2D[k].xy for k in idx])
        z_sfm = (xyz @ T.rotation.matrix().T + T.translation)[:, 2]
        iu = np.clip((xy[:, 0] * w / cam.width).astype(int), 0, w - 1)
        iv = np.clip((xy[:, 1] * h / cam.height).astype(int), 0, h - 1)
        z_ff = ff["depth"][i][iv, iu]
        good = (z_ff > 0) & (z_sfm > 0)
        if good.sum() >= bcfg["min_obs"]:
            ratios[i] = np.median(z_sfm[good] / z_ff[good])
    log["depth_scale_per_view"] = [None if np.isnan(r) else float(r) for r in ratios]
    return E_ba, K, ratios, strong, times, log, rec


def dense_case(run_dir, case, gt, ccfg, comp_cfg):
    """Classical dense stage on the DA3-BA cameras: COLMAP undistortion + PatchMatch + stereo fusion
    (exactly the baseline's dense stage), then the holes of the fused cloud filled with the DA3-BA
    depth (da3ba_ds output, same frame; depth_completion.py without guided correction).
    Writes hybrid_da3ba (fused MVS cloud) and hybrid_da3ba_fill (plus fill points)."""
    src = case_dir(run_dir, "da3ba", case)
    work = src / "colmap_work"
    out_mvs, out_fill = case_dir(run_dir, "hybrid_da3ba", case), case_dir(run_dir, "hybrid_da3ba_fill", case)
    if not (work / "sparse").exists():
        return "no sparse model (DA3 fallback), skipped"
    image_dir = Path(gt["image_files"][0]).parent
    names = [Path(p).name for p in gt["image_files"]]
    rec = pycolmap.Reconstruction(work / "sparse")
    ba = dict(np.load(case_dir(run_dir, "da3ba_ds", case) / "exports" / "mini_npz" / "results.npz"))
    E = np.tile(np.eye(4), (len(names), 1, 1))
    E[:, :3] = ba["extrinsics"][:, :3]
    ff_by_name = {n: (ba["depth"][i], ba["intrinsics"][i], E[i]) for i, n in enumerate(names)}
    after = lambda dense_dir, fused: completion_points(dense_dir, image_dir, rec, ff_by_name, fused, comp_cfg,
                                                       variants=["fill"])
    rec_dense, xyz, times, min_pix, completion = dense(work, work / "sparse", image_dir, ccfg, after)
    sparse_info = json.load(open(src / "info.json"))
    times = {**sparse_info["time_s"], **times}
    for k, (o, name) in enumerate([(out_mvs, "hybrid_da3ba"), (out_fill, "hybrid_da3ba_fill")]):
        fill_xyz, info = None, {"fusion_min_num_pixels": min_pix, "config": ccfg}
        if name.endswith("fill"):
            fill_xyz, info["completion"] = completion["fill"]
        export_case(o, name, case, gt, rec_dense, distorted_cameras(rec), xyz, times, info, fill_xyz, work=work,
                    cleanup=False)
    shutil.rmtree(work / "dense", ignore_errors=True)
    return f"{len(xyz) / 1e6:.2f}M fused, +{len(completion['fill'][0]) / 1e6:.2f}M fill, {sum(times.values()):.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--shard", help="i/n: run every n-th case starting at i")
    ap.add_argument("--dense", action="store_true", help="dense stage on finished DA3-BA cases (hybrid_da3ba*)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ccfg, bcfg = cfg["colmap"], cfg["da3ba"]
    ff_method = bcfg.get("ff_method", "da3")  # vggt: the same refinement on VGGT's cameras and depth
    run_dir = ROOT / "runs" / cfg["run_name"]
    src_dir = ROOT / "runs" / cfg["source_run"]
    cases = shard_filter(json.load(open(ROOT / "manifests" / f"{cfg['run_name']}.json"))["cases"], args.shard)
    ds = eth3d_dataset()
    if args.dense:
        for case in cases:
            if is_done(case_dir(run_dir, "hybrid_da3ba_fill", case)):
                continue
            msg = dense_case(run_dir, case, case_gt(ds, case), ccfg, cfg["completion"])
            print(f"hybrid_da3ba {case['scene']} {case['sampling']} {case['n_views']} views s{case['seed']}: {msg}",
                  flush=True)
        return
    for case in cases:
        out, out_ds = case_dir(run_dir, f"{ff_method}ba", case), case_dir(run_dir, f"{ff_method}ba_ds", case)
        if is_done(out) and is_done(out_ds):
            continue
        ff_dir = case_dir(src_dir, ff_method, case)
        ff = dict(np.load(ff_dir / "exports" / "mini_npz" / "results.npz"))
        ff_info = json.load(open(ff_dir / "info.json"))
        gt = case_gt(ds, case)
        t0 = time.time()
        with GpuMemSampler() as mem:
            E, K, ratios, strong, times, log, rec = run_case(out / "colmap_work", case, gt, ff, ccfg, bcfg)
        global_scale = float(np.nanmedian(ratios)) if np.isfinite(ratios).any() else 1.0
        per_view = np.where(np.isfinite(ratios), ratios, global_scale)
        ff_time = sum(ff_info["time_s"].values())
        for o, name, scale in [(out, f"{ff_method}ba", np.full(len(E), global_scale)),
                               (out_ds, f"{ff_method}ba_ds", per_view)]:
            info = {"method": name,
                    "case": {k: case.get(k, "random") for k in ("scene", "n_views", "sampling", "seed", "images")},
                    "registered": [True] * len(E),
                    "time_s": {ff_method: ff_time, **times},
                    "refined_view": strong.tolist(), "depth_scale": scale.tolist(),
                    "peak_gpu_gb": mem.peak_gb, "config": bcfg, **log}
            save_case(o, ff["depth"] * scale[:, None, None], E[:, :3, :4], K, gt, info)
        for d in ("tri", "tri0"):
            shutil.rmtree(out / "colmap_work" / d, ignore_errors=True)
        (out / "colmap_work" / "database.db").unlink(missing_ok=True)
        if rec is not None:  # final sparse model, input of the dense stage
            (out / "colmap_work" / "sparse").mkdir(exist_ok=True)
            rec.write(out / "colmap_work" / "sparse")
        print(f"{ff_method}ba {case['scene']} {case['sampling']} {case['n_views']} views s{case['seed']}: "
              f"{log.get('n_points_init')} lifted pts, err {log.get('init_err_median') or float('nan'):.1f} -> "
              f"{log.get('final_err_median') or float('nan'):.2f}px, refined {int(strong.sum())}/{len(strong)}, "
              f"{log.get('fallback', '')} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
