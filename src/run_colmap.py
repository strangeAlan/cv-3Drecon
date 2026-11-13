"""Classical baseline: COLMAP (SIFT -> exhaustive matching -> incremental SfM ->
undistortion -> PatchMatch stereo -> stereo fusion) on every case of a manifest.

Views that COLMAP fails to register are dropped from results.npz/gt_meta.npz and
recorded in info.json["registered"] for the pose metric with failures counted.
See colmap_common.py for the dense stage and the output format.

Usage: python src/run_colmap.py configs/<run>.yaml [--shard i/n]
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import pycolmap

from colmap_common import (dense, distorted_cameras, export_case, extract_and_match, match_stats, shard_filter,
                           write_failure)
from common import ROOT, GpuMemSampler, case_dir, case_gt, eth3d_dataset, is_done, load_config


def run_case(out, case, gt, ccfg):
    work = out / "colmap_work"
    image_dir = Path(gt["image_files"][0]).parent
    names = [Path(p).name for p in gt["image_files"]]
    if work.exists():
        shutil.rmtree(work)
    (work / "sparse").mkdir(parents=True)
    db = work / "database.db"
    times = {}

    t0 = time.time()
    extract_and_match(db, image_dir, names, ccfg)
    times["features_matching"] = time.time() - t0
    stats = match_stats(db)

    t0 = time.time()
    recs = pycolmap.incremental_mapping(db, image_dir, work / "sparse")
    times["sfm"] = time.time() - t0
    rec = max(recs.values(), key=lambda r: r.num_reg_images()) if recs else None
    n_reg = rec.num_reg_images() if rec else 0
    extra = {"config": ccfg, "match_stats": stats}
    if n_reg < 2:
        write_failure(out, ccfg.get("method_name", "colmap"), case, gt, times, extra, n_reg)
        return f"FAILED ({n_reg} registered)"
    (work / "sparse_best").mkdir()
    rec.write(work / "sparse_best")

    rec_dense, xyz, dtimes, min_pix, _ = dense(work, work / "sparse_best", image_dir, ccfg)
    times.update(dtimes)
    extra["fusion_min_num_pixels"] = min_pix
    reg = export_case(out, ccfg.get("method_name", "colmap"), case, gt, rec_dense, distorted_cameras(rec), xyz, times, extra)
    return f"{sum(reg)}/{len(reg)} registered, {len(xyz) / 1e6:.2f}M fused points"


def _add_info(out, **kv):
    info_path = out / "info.json"
    info = json.load(open(info_path))
    info.update(kv)
    json.dump(info, open(info_path, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--shard", help="i/n: run every n-th case starting at i")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ccfg = cfg["colmap"]
    run_dir = ROOT / "runs" / cfg["run_name"]
    cases = shard_filter(json.load(open(ROOT / "manifests" / f"{cfg['run_name']}.json"))["cases"], args.shard)
    ds = eth3d_dataset()

    for case in cases:
        method = ccfg.get("method_name", "colmap")  # e.g. colmap_radial: same pipeline, other camera model
        out = case_dir(run_dir, method, case)
        if is_done(out):
            continue
        t0 = time.time()
        with GpuMemSampler() as mem:
            msg = run_case(out, case, case_gt(ds, case), ccfg)
        _add_info(out, peak_gpu_gb=mem.peak_gb, wall_s=time.time() - t0)
        print(f"{method} {case['scene']} {case.get('sampling', 'random')} {case['n_views']} views s{case['seed']}: "
              f"{msg}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
