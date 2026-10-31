"""Compact view of results/<run>/metrics.csv: one row per case and method."""
import sys

import pandas as pd

pd.set_option("display.width", 250)
d = pd.read_csv(f"results/{sys.argv[1]}/metrics.csv")
order = ["colmap", "vggt", "da3", "hybrid_vggt_none", "hybrid_vggt_ba", "hybrid_vggt_ba_fill", "hybrid_vggt_ba_gf",
         "hybrid_da3_none", "hybrid_da3_ba", "hybrid_da3_ba_fill", "hybrid_da3_ba_gf"]
d["method"] = pd.Categorical(d.method, [m for m in order if m in set(d.method)])
d["reg"] = d.n_registered.astype(str) + "/" + d.n_input.astype(str)
cols = ["reg", "posefail_auc05", "posefail_auc30", "fscore_pc@0.02", "fscore_pc@0.05", "fscore_pc@0.10",
        "precision_pc@0.05", "recall_pc@0.05", "acc_pc_median", "comp_pc_median", "time_s"]
cols = [c for c in cols if c in d]
t = d.sort_values(["scene", "sampling", "n_views", "seed", "method"]).set_index(
    ["scene", "sampling", "n_views", "seed", "method"])[cols]
for c in t.columns:
    if c.endswith("median"):
        t[c] = (t[c] * 1000).round(1)
    elif c != "reg":
        t[c] = t[c].round(3)
print(t.to_string())
