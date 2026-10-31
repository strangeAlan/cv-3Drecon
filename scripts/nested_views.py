"""Same views, more context: do the methods get worse on a fixed set of views when more views are added?

Neighbor subsets are nested (same seed -> same anchor, the 5-view set is a prefix of the 10- and
20-view sets; checked on the manifest). For every neighbor 10/20-view case we cut the predictions of the
5 shared views out of the larger reconstruction and evaluate them exactly like a 5-view case
(pose AUC over the 5-view pairs; geometry aligned and scored on those 5 views only). Comparing with
the 5-view run separates "the model degrades with more input views" from "larger subsets are harder".
COLMAP: pose only (its geometry is one fused cloud that cannot be split by view).

Usage: python scripts/nested_views.py 1027-main [--workers 24]
       -> results/<run>/nested_views.csv and a printed summary
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import evaluate as ev  # noqa: E402  (preloads libusb through common)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import ROOT, case_name, eth3d_dataset  # noqa: E402

_DS = None


def _init():
    global _DS
    _DS = eth3d_dataset()


def load_case(cdir):
    info = json.load(open(cdir / "info.json"))
    if info.get("failed"):
        return info, None, None
    res = dict(np.load(cdir / "exports" / "mini_npz" / "results.npz"))
    gt = dict(np.load(cdir / "exports" / "gt_meta.npz", allow_pickle=True))
    return info, res, gt


def subset_metrics(task):
    run, method, scene, seed, n_big, names = task
    run_dir = ROOT / "runs" / run
    cdir = run_dir / method / scene / case_name(n_big, seed, "neighbor")
    info, res, gt = load_case(cdir)
    row = {"method": method, "scene": scene, "seed": seed, "n_context": n_big}
    if res is None:
        return row
    reg_names = [os.path.basename(str(p)) for p in gt["image_files"]]
    idx = [reg_names.index(n) for n in names if n in reg_names]
    registered = [n in reg_names for n in names]
    row["n_registered"] = sum(registered)
    if len(idx) >= 2:
        sub = {k: v[idx] for k, v in res.items()}
        sub_gt = {k: v[idx] for k, v in gt.items()}
        row.update({k: v for k, v in ev.pose_metrics(sub, sub_gt, registered).items() if k.startswith("posefail")})
        if not method.startswith("colmap") and len(idx) >= 3:
            sub_info = dict(info, registered=registered)
            row.update(ev.point_cloud_metrics(_DS, scene, sub, sub_gt, sub_info))
    return row


def main():
    global RUN
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    RUN = args.run
    manifest = json.load(open(ROOT / "manifests" / f"{RUN}.json"))["cases"]
    cases = {(c["scene"], c["sampling"], c["n_views"], c["seed"]): c["images"] for c in manifest}
    tasks = []
    for (scene, sampling, n, seed), images in cases.items():
        if sampling != "neighbor" or n not in (10, 20) or (scene, sampling, 5, seed) not in cases:
            continue
        small = cases[(scene, sampling, 5, seed)]
        assert set(small) <= set(images), (scene, n, seed)
        for method in ("colmap", "vggt", "da3"):
            tasks.append((RUN, method, scene, seed, n, small))
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=()) as pool:
        rows = pool.map(subset_metrics, tasks, chunksize=1)

    d = pd.DataFrame(rows)
    base = pd.read_csv(ROOT / "results" / RUN / "metrics.csv")
    base = base[(base.sampling == "neighbor") & (base.n_views.astype(str) == "5")].copy()
    base["n_context"] = 5
    keep = ["method", "scene", "seed", "n_context", "posefail_auc05", "posefail_auc30", "fscore_pc@0.05",
            "precision_pc@0.05", "recall_pc@0.05", "acc_pc_median"]
    out = pd.concat([base[[c for c in keep if c in base]], d[[c for c in keep if c in d]]], ignore_index=True)
    out.to_csv(ROOT / "results" / RUN / "nested_views.csv", index=False)

    # same (scene, seed) set for every context size: cases that have a 20-view run
    has20 = set(map(tuple, d[d.n_context == 20][["scene", "seed"]].drop_duplicates().values))
    out = out[[(s, sd) in has20 for s, sd in zip(out.scene, out.seed)]]
    for c in [c for c in keep[4:] if c.startswith(("fscore", "precision", "recall"))]:
        out[c] = out[c].fillna(0.0)
    pd.set_option("display.width", 200)
    print(f"5 fixed views evaluated inside 5/10/20-view reconstructions, {len(has20)} (scene, seed) pairs")
    print(out.groupby(["method", "n_context"])[keep[4:]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
