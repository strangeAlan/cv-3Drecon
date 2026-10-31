"""Feature/matching statistics for every case of a run (failure analysis): re-runs only COLMAP's
feature extraction and exhaustive matching with the run's settings and writes
results/<run>/match_stats.csv, joined with COLMAP's registration result of that case.

Usage: python scripts/match_stats.py configs/<run>.yaml [--shard i/n]
"""
import argparse
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from colmap_common import extract_and_match, match_stats, shard_filter  # noqa: E402
from common import ROOT, case_dir, case_gt, eth3d_dataset, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--shard")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_dir = ROOT / "runs" / cfg["run_name"]
    cases = shard_filter(json.load(open(ROOT / "manifests" / f"{cfg['run_name']}.json"))["cases"], args.shard)
    out = ROOT / "results" / cfg["run_name"] / f"match_stats{'_' + args.shard.replace('/', 'of') if args.shard else ''}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    ds = eth3d_dataset()
    rows = []
    for case in cases:
        gt = case_gt(ds, case)
        names = [Path(p).name for p in gt["image_files"]]
        tmp = Path(tempfile.mkdtemp(dir=run_dir))
        try:
            extract_and_match(tmp / "database.db", Path(gt["image_files"][0]).parent, names, cfg["colmap"])
            st = match_stats(tmp / "database.db")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        info = json.load(open(case_dir(run_dir, "colmap", case) / "info.json"))
        row = {"scene": case["scene"], "sampling": case.get("sampling", "random"), "n_views": case["n_views"],
               "seed": case["seed"], "n_registered": sum(info["registered"]), "failed": bool(info.get("failed")), **st}
        rows.append(row)
        print(row, flush=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
