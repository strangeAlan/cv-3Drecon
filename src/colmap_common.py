"""Shared COLMAP stages for the classical baseline (run_colmap.py) and the hybrid method
(run_hybrid.py): feature extraction/matching, and the dense stage (undistortion ->
PatchMatch stereo -> stereo fusion) plus export to the common case format.

Both methods go through exactly the same dense stage; they differ only in where the
sparse model (camera poses and intrinsics) comes from.

Exported per case, besides the common results.npz / gt_meta.npz / info.json:
    exports/fused_points.npy   COLMAP stereo-fusion point cloud (float32, COLMAP world frame),
                               the standard COLMAP dense output used by the point-cloud metrics
Depth maps live on COLMAP's undistorted (pinhole) grid, while ETH3D images and GT masks
are distorted; info.json["colmap_cameras"] keeps each view's estimated distorted camera so
evaluate.py can map the GT mask onto the undistorted grid.
"""
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from common import save_case


def read_colmap_array(path):
    """COLMAP dense .bin format: ASCII header 'w&h&c&' then float32, column-major."""
    with open(path, "rb") as f:
        header = b""
        while header.count(b"&") < 3:
            header += f.read(1)
        w, h, c = map(int, header.decode().split("&")[:3])
        data = np.fromfile(f, np.float32)
    return data.reshape((w, h, c), order="F").transpose(1, 0, 2).squeeze()


def cam_from_world(image):
    pose = image.cam_from_world
    pose = pose() if callable(pose) else pose
    m = np.eye(4)
    m[:3] = pose.matrix()
    return m


def device(ccfg):
    return pycolmap.Device.cuda if ccfg["gpu"] else pycolmap.Device.cpu


def extract_and_match(db, image_dir, names, ccfg):
    extract_opts = pycolmap.FeatureExtractionOptions()
    extract_opts.max_image_size = ccfg["sift_max_image_size"]
    reader_opts = pycolmap.ImageReaderOptions()
    reader_opts.camera_model = ccfg["camera_model"]
    pycolmap.extract_features(db, image_dir, image_names=names,
                              camera_mode=getattr(pycolmap.CameraMode, ccfg["camera_mode"]),
                              reader_options=reader_opts, extraction_options=extract_opts,
                              device=device(ccfg))
    pycolmap.match_exhaustive(db, device=device(ccfg))


def distorted_cameras(rec):
    """Estimated (distorted) camera of every registered image, keyed by image name."""
    cams = {}
    for im in rec.images.values():
        if im.has_pose:
            c = rec.cameras[im.camera_id]
            cams[im.name] = {"model": c.model.name, "width": c.width, "height": c.height,
                             "params": [float(x) for x in c.params]}
    return cams


def fusion_min_num_pixels(n_registered):
    """COLMAP's default (5 consistent pixels) assumes dense captures; with 5 views it would
    demand that every view sees the point. Relax it for sparse inputs."""
    if n_registered <= 5:
        return 2
    if n_registered <= 10:
        return 3
    return 5


def dense(work, sparse_dir, image_dir, ccfg, after_fusion=None):
    """Undistort + PatchMatch (geometric consistency) + stereo fusion on a sparse model.
    after_fusion(dense_dir, fused_xyz) runs while the dense workspace still exists (hybrid depth
    completion); its return value is passed through."""
    times = {}
    out = work / "dense"
    rec = pycolmap.Reconstruction(sparse_dir)
    t0 = time.time()
    undist_opts = pycolmap.UndistortCameraOptions()
    undist_opts.max_image_size = ccfg["dense_max_image_size"]
    pycolmap.undistort_images(out, sparse_dir, image_dir, undistort_options=undist_opts,
                              num_patch_match_src_images=ccfg.get("patch_match_src_images", 20))
    times["undistort"] = time.time() - t0
    t0 = time.time()
    pm_opts = pycolmap.PatchMatchOptions()
    pm_opts.geom_consistency = True  # COLMAP CLI default; writes *.geometric.bin
    pycolmap.patch_match_stereo(out, options=pm_opts)
    times["patch_match"] = time.time() - t0
    t0 = time.time()
    fusion_opts = pycolmap.StereoFusionOptions()
    fusion_opts.min_num_pixels = fusion_min_num_pixels(rec.num_reg_images())
    fused = pycolmap.stereo_fusion(out / "fused.ply", out, options=fusion_opts, output_type="PLY")
    times["fusion"] = time.time() - t0
    xyz = np.array([p.xyz for p in fused.points3D.values()], np.float32).reshape(-1, 3)
    extra = None
    if after_fusion is not None:
        t0 = time.time()
        extra = after_fusion(out, xyz)
        times["completion"] = time.time() - t0
    return pycolmap.Reconstruction(out / "sparse"), xyz, times, fusion_opts.min_num_pixels, extra


def export_case(out, method, case, gt, rec_dense, distorted, xyz, times, extra_info, fill_xyz=None,
                work=None, cleanup=True):
    """Write the common case format for a dense COLMAP-style reconstruction (dense workspace in
    work, default out/colmap_work; cleanup removes it afterwards)."""
    work = out / "colmap_work" if work is None else work
    names = [Path(p).name for p in gt["image_files"]]
    by_name = {im.name: im for im in rec_dense.images.values() if im.has_pose}
    reg = [n in by_name for n in names]
    info = {"method": method,
            "case": {k: case.get(k, "random") for k in ("scene", "n_views", "sampling", "seed", "images")},
            "registered": reg, "time_s": times,
            "colmap_cameras": [distorted[n] for n in names if n in by_name], **extra_info}
    keep = [i for i, r in enumerate(reg) if r]
    depths, exts, ixts = [], [], []
    hw = None
    for i in keep:
        im = by_name[names[i]]
        cam = rec_dense.cameras[im.camera_id]
        dpath = work / "dense" / "stereo" / "depth_maps" / f"{im.name}.geometric.bin"
        d = read_colmap_array(dpath) if dpath.exists() else np.zeros((cam.height, cam.width), np.float32)
        K = cam.calibration_matrix().copy()
        if hw is None:
            hw = d.shape
        if d.shape != hw:  # keep one shape for stacking; scale K accordingly
            K[0] *= hw[1] / d.shape[1]
            K[1] *= hw[0] / d.shape[0]
            d = cv2.resize(d, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
        depths.append(d)
        exts.append(cam_from_world(im))
        ixts.append(K)
    sub_gt = {k: (v[keep] if isinstance(v, np.ndarray) else [v[i] for i in keep]) for k, v in gt.items()}
    info["depth_hw"] = list(hw)
    info["n_fused_points"] = int(len(xyz))
    save_case(out, np.stack(depths), np.stack(exts), np.stack(ixts), sub_gt, info)
    np.save(out / "exports" / "fused_points.npy", xyz)
    if fill_xyz is not None:
        np.save(out / "exports" / "fill_points.npy", fill_xyz)
        info["n_fill_points"] = int(len(fill_xyz))
        json.dump(info, open(out / "info.json", "w"), indent=2)
    if cleanup:  # keep the sparse models for inspection, drop the bulky dense workspace
        shutil.rmtree(work / "dense", ignore_errors=True)
        (work / "database.db").unlink(missing_ok=True)
    return reg


def write_failure(out, method, case, gt, times, extra_info, n_registered):
    out.mkdir(parents=True, exist_ok=True)
    names = [Path(p).name for p in gt["image_files"]]
    info = {"method": method,
            "case": {k: case.get(k, "random") for k in ("scene", "n_views", "sampling", "seed", "images")},
            "registered": [False] * len(names), "n_registered_sparse": n_registered,
            "time_s": times, "failed": True, **extra_info}
    json.dump(info, open(out / "info.json", "w"), indent=2)
    shutil.rmtree(out / "colmap_work" / "dense", ignore_errors=True)
    (out / "colmap_work" / "database.db").unlink(missing_ok=True)


def shard_filter(cases, shard):
    """'i/n' -> every n-th case starting at i (for running several workers in parallel)."""
    if not shard:
        return cases
    i, n = map(int, shard.split("/"))
    return [c for k, c in enumerate(cases) if k % n == i]


def match_stats(db_path, min_pair_inliers=15):
    """Feature/matching statistics from the COLMAP database, for the failure analysis:
    keypoints per image, verified pairs and their inliers, and how many images are connected to
    the largest component of the verified-pair graph (pairs with >= min_pair_inliers inliers,
    COLMAP's default minimum for two-view geometry)."""
    db = pycolmap.Database.open(db_path)
    images = db.read_all_images()
    kps = [db.num_keypoints_for_image(im.image_id) for im in images]
    pair_ids, geoms = db.read_two_view_geometries()
    edges = []
    inliers = []
    for pid, g in zip(pair_ids, geoms):
        n = len(g.inlier_matches)
        if n == 0:
            continue
        inliers.append(n)
        if n >= min_pair_inliers:
            edges.append(pycolmap.pair_id_to_image_pair(pid))
    db.close()
    # largest connected component of the verified-pair graph
    parent = {im.image_id: im.image_id for im in images}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        parent[find(a)] = find(b)
    comp = {}
    for i in parent:
        comp[find(i)] = comp.get(find(i), 0) + 1
    n_pairs = len(images) * (len(images) - 1) // 2
    return {"n_images": len(images), "keypoints_mean": float(np.mean(kps)) if kps else 0.0,
            "keypoints_min": int(min(kps)) if kps else 0, "n_pairs": n_pairs,
            "n_verified_pairs": len(inliers), "n_pairs_min_inliers": len(edges),
            "pair_inliers_median": float(np.median(inliers)) if inliers else 0.0,
            "largest_component": max(comp.values()) if comp else 0}
