"""Aggregate results/<run>/metrics.csv into the report tables.

Writes results/<run>/summary.csv: per method x sampling x n_views, mean and std over all
(scene, seed) cases, plus case counts. Failed reconstructions enter with F-score 0 and pose
AUC computed with failures counted, so averages are over the same cases for every method.

Usage: python scripts/summarize.py <run_name>
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["colmap", "vggt", "da3"]
METRICS = ["reg_rate", "posefail_auc05", "posefail_auc30", "fscore_pc@0.02", "fscore_pc@0.05", "fscore_pc@0.10",
           "precision_pc@0.05", "recall_pc@0.05", "acc_pc_median", "comp_pc_median", "time_s"]


def load(run):
    d = pd.read_csv(ROOT / "results" / run / "metrics.csv")
    d = d[d.method.isin(METHODS)].copy()
    d["n_views"] = d.n_views.astype(str)
    d["reg_rate"] = d.n_registered / d.n_input
    for c in METRICS:
        if c not in d:
            d[c] = float("nan")
    for c in [c for c in METRICS if c.startswith(("fscore", "precision", "recall"))]:
        d[c] = d[c].fillna(0.0)  # failed reconstructions
    for c in ["acc_pc_median", "comp_pc_median"]:
        d[c] = d[c].replace(float("inf"), float("nan"))
    return d


def main():
    run = sys.argv[1]
    d = load(run)
    # only cases evaluated for every method, so all methods are averaged over the same set
    key = ["scene", "sampling", "n_views", "seed"]
    complete = d.groupby(key).method.nunique()
    complete = complete[complete == d.method.nunique()].index
    d = d.set_index(key).loc[complete].reset_index()
    g = d.groupby(["method", "sampling", "n_views"])[METRICS]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std")).join(g.size().rename("n_cases"))
    out = out.reset_index()
    out.to_csv(ROOT / "results" / run / "summary.csv", index=False)

    view_order = {"5": 0, "10": 1, "20": 2, "all": 3}
    out["_v"] = out.n_views.map(view_order)
    out["_m"] = out.method.map({m: i for i, m in enumerate(METHODS)})
    show = out.sort_values(["sampling", "_v", "_m"])
    cols = ["sampling", "n_views", "method", "n_cases", "reg_rate_mean", "posefail_auc05_mean", "posefail_auc30_mean",
            "fscore_pc@0.02_mean", "fscore_pc@0.05_mean", "fscore_pc@0.10_mean", "precision_pc@0.05_mean",
            "recall_pc@0.05_mean", "acc_pc_median_mean", "comp_pc_median_mean", "time_s_mean"]
    pd.set_option("display.width", 250)
    t = show[cols].copy()
    for c in cols[4:]:
        t[c] = t[c].round(3 if not c.startswith(("acc", "comp", "time")) else 1 if c.startswith("time") else 4)
    t.columns = [c.replace("_mean", "").replace("posefail_", "") for c in t.columns]
    print(t.to_string(index=False))
    print(f"\n{len(complete)} cases with all of {', '.join(sorted(d.method.unique()))}")


if __name__ == "__main__":
    main()
