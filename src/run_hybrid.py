"""Hybrid reconstruction: feed-forward cameras + classical refinement and dense matching.

Per case:
  1. SIFT + exhaustive matching (same settings as the COLMAP baseline)
  2. sparse model initialised from a feed-forward model (VGGT or DA3): every view gets
     the predicted pose and intrinsics (scaled to the original image size, radial
     distortion 0), so no view can fail to register
  3. triangulation of the SIFT matches with these cameras
  4. refine=ba: global bundle adjustment of poses, focal length, radial distortion and
     points (robust loss), re-triangulation, second BA; refine=none: keep the
     feed-forward cameras
  5. the COLMAP baseline's dense stage (undistortion, PatchMatch, stereo fusion)
  6. --complete: the holes of the fused MVS cloud are filled with the aligned feed-forward
     reconstruction, without (_fill) and with (_gf) guided depth correction (depth_completion.py),
     saved as fill_points.npy; all variants share one dense reconstruction

Output method names: hybrid_<init>_<refine>[_fill|_gf], same files as the COLMAP baseline.

Usage: python src/run_hybrid.py configs/<run>.yaml --init vggt|da3 --refine none|ba [--complete] [--shard i/n]
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pycolmap

from colmap_common import dense, distorted_cameras, export_case, extract_and_match, shard_filter, write_failure
from common import ROOT, case_dir, case_gt, eth3d_dataset, is_done, load_config
from depth_completion import completion_points


def init_reconstruction(db_path, ff, names):
    """Sparse model with feed-forward cameras, ids taken from the COLMAP database."""
    db = pycolmap.Database.open(db_path)
    db_images = {im.name: im for im in db.read_all_images()}
    db_cameras = {c.camera_id: c for c in db.read_all_cameras()}
    db.close()

    h, w = ff["depth"].shape[1:]
    K = ff["intrinsics"].astype(np.float64).copy()
    for i, n in enumerate(names):  # predicted intrinsics refer to the depth grid
        cam = db_cameras[db_images[n].camera_id]
        K[i, 0] *= cam.width / w
        K[i, 1] *= cam.height / h
    ext = ff["extrinsics"].astype(np.float64)

    rec = pycolmap.Reconstruction()
    idx_by_cam = {}
    for i, n in enumerate(names):
        idx_by_cam.setdefault(db_images[n].camera_id, []).append(i)
    for cam_id, idx in idx_by_cam.items():
        cam = db_cameras[cam_id]
        f = np.median((K[idx, 0, 0] + K[idx, 1, 1]) / 2)
        cx, cy = np.median(K[idx, 0, 2]), np.median(K[idx, 1, 2])
        params = {"SIMPLE_RADIAL": [f, cx, cy, 0.0], "RADIAL": [f, cx, cy, 0.0, 0.0],
                  "OPENCV": [f, f, cx, cy, 0.0, 0.0, 0.0, 0.0], "PINHOLE": [f, f, cx, cy]}[cam.model.name]
        cam.params = params
        rec.add_camera_with_trivial_rig(cam)
    for i, n in enumerate(names):
        dbi = db_images[n]
        image = pycolmap.Image(name=n, camera_id=dbi.camera_id, image_id=dbi.image_id)
        rec.add_image_with_trivial_frame(image, pycolmap.Rigid3d(ext[i, :3, :4]))
    return rec


def bundle_adjust(rec):
    opts = pycolmap.BundleAdjustmentOptions()
    opts.refine_focal_length = True
    opts.refine_extra_params = True
    opts.refine_principal_point = False
    opts.print_summary = False
    opts.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY
    pycolmap.bundle_adjustment(rec, opts)


def run_case(outs, case, gt, ff, ccfg, refine, complete_cfg=None):
    """outs: {variant: case dir}; variant "base" always, "fill" and "gf" with completion. All variants
    share one sparse + dense reconstruction (in the base dir's colmap_work)."""
    work = outs["base"] / "colmap_work"
    image_dir = Path(gt["image_files"][0]).parent
    names = [Path(p).name for p in gt["image_files"]]
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    db = work / "database.db"
    times = {}

    t0 = time.time()
    extract_and_match(db, image_dir, names, ccfg)
    times["features_matching"] = time.time() - t0

    t0 = time.time()
    rec = init_reconstruction(db, ff, names)
    rec = pycolmap.triangulate_points(rec, db, image_dir, work / "tri0")
    n_tri0 = len(rec.points3D)
    if refine == "ba":
        bundle_adjust(rec)
        rec = pycolmap.triangulate_points(rec, db, image_dir, work / "tri1")
        bundle_adjust(rec)
    times["sparse"] = time.time() - t0
    (work / "sparse_best").mkdir()
    rec.write(work / "sparse_best")
    extra = {"config": ccfg, "refine": refine, "n_points3D_init": n_tri0, "n_points3D": len(rec.points3D),
             "mean_reproj_error": float(rec.compute_mean_reprojection_error())}
    if len(rec.points3D) == 0:
        for out in outs.values():
            write_failure(out, out.parent.parent.name, case, gt, times, extra, 0)
        return "FAILED (no triangulated points)"

    after_fusion = None
    if complete_cfg is not None:
        ff_ext = np.tile(np.eye(4), (len(names), 1, 1))
        ff_ext[:, :3] = ff["extrinsics"][:, :3]
        ff_by_name = {n: (ff["depth"][i], ff["intrinsics"][i], ff_ext[i]) for i, n in enumerate(names)}
        after_fusion = lambda dense_dir, fused: completion_points(
            dense_dir, image_dir, rec, ff_by_name, fused, complete_cfg, variants=[v for v in outs if v != "base"])
    rec_dense, xyz, dtimes, min_pix, completion = dense(work, work / "sparse_best", image_dir, ccfg, after_fusion)
    times.update(dtimes)
    extra["fusion_min_num_pixels"] = min_pix
    msg = []
    variants = list(outs)
    for k, variant in enumerate(variants):
        out = outs[variant]
        fill_xyz, info = None, dict(extra)
        if variant != "base":
            fill_xyz, info["completion"] = completion[variant]
        reg = export_case(out, out.parent.parent.name, case, gt, rec_dense, distorted_cameras(rec), xyz, times, info,
                          fill_xyz, work=work, cleanup=k == len(variants) - 1)
        if fill_xyz is not None:
            msg.append(f"{variant} +{len(fill_xyz) / 1e6:.2f}M")
    return (f"{sum(reg)}/{len(reg)} views, {extra['n_points3D']} sparse pts, "
            f"reproj {extra['mean_reproj_error']:.2f}px, {len(xyz) / 1e6:.2f}M fused points " + ", ".join(msg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--init", choices=["vggt", "da3"], required=True)
    ap.add_argument("--refine", choices=["none", "ba"], required=True)
    ap.add_argument("--complete", action="store_true",
                    help="also write the _fill and _gf variants (holes of the fused MVS cloud filled with the "
                         "aligned feed-forward reconstruction, without / with guided depth correction)")
    ap.add_argument("--shard", help="i/n: run every n-th case starting at i")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ccfg = cfg["colmap"]
    base = f"hybrid_{args.init}_{args.refine}"
    variants = {"base": base, **({"fill": base + "_fill", "gf": base + "_gf"} if args.complete else {})}
    run_dir = ROOT / "runs" / cfg["run_name"]
    cases = shard_filter(json.load(open(ROOT / "manifests" / f"{cfg['run_name']}.json"))["cases"], args.shard)
    ds = eth3d_dataset()

    for case in cases:
        outs = {v: case_dir(run_dir, m, case) for v, m in variants.items()}
        if all(is_done(o) for o in outs.values()):
            continue
        ff_dir = case_dir(run_dir, args.init, case)
        if not (ff_dir / "info.json").exists():
            print(f"{base} {case['scene']}: missing {args.init} output, skipped", flush=True)
            continue
        ff = dict(np.load(ff_dir / "exports" / "mini_npz" / "results.npz"))
        t0 = time.time()
        msg = run_case(outs, case, case_gt(ds, case), ff, ccfg, args.refine,
                       cfg["completion"] if args.complete else None)
        print(f"{base} {case['scene']} {case.get('sampling', 'random')} {case['n_views']} views s{case['seed']}: "
              f"{msg}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
