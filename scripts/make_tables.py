"""LaTeX tables for the report, straight from the result files (no hand-copied numbers).

tab_repro.tex       DA3-protocol reproduction (1027-repro) vs. published numbers
tab_main.tex        cameras + geometry per sampling x view count (1027-main) and all views (1028-allviews)
tab_scene.tex       per scene: type, SIFT keypoints, COLMAP registration, F@5cm (neighbor subsets, all views)
tab_efficiency.tex  end-to-end runtime and peak GPU memory (1028-efficiency, 1028-allviews)
tab_nested.tex      same 5 views inside 5/10/20-view reconstructions (results/1027-main/nested_views.csv)

Usage: python scripts/make_tables.py [--out report/tables]
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["colmap", "vggt", "da3"]
LABEL = {"colmap": "COLMAP", "vggt": "VGGT", "da3": "DA3"}
INDOOR = {"kicker", "office", "pipes", "relief", "relief_2", "terrains"}  # ETH3D scene categories
PAPER = {"vggt": (26.3, 80.8, 57.2), "da3": (48.4, 91.2, 79.0)}  # DA3 paper, ETH3D, DA3-Giant


def load(run):
    d = pd.read_csv(ROOT / "results" / run / "metrics.csv")
    d = d[d.method.isin(METHODS)].copy()
    d["n_views"] = d.n_views.astype(str)
    d["reg_rate"] = d.n_registered / d.n_input
    for c in [c for c in d if c.startswith(("fscore_pc", "precision_pc", "recall_pc"))]:
        d[c] = d[c].fillna(0.0)
    key = ["scene", "sampling", "n_views", "seed"]
    return d[d.groupby(key).method.transform("nunique") == len(METHODS)]


def bold_best(vals, higher=True):
    best = max(vals) if higher else min(vals)
    return [f"\\textbf{{{v:.1f}}}" if np.isclose(v, best) else f"{v:.1f}" for v in vals]


def tab_repro(out):
    d = pd.read_csv(ROOT / "results" / "1027-repro" / "metrics.csv")
    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             r"Method & Source & AUC@3 & AUC@30 & F1@25\,cm \\", r"\midrule"]
    for m in ["vggt", "da3"]:
        g = d[d.method == m]
        ours = [100 * g.pose_auc03.mean(), 100 * g.pose_auc30.mean(), 100 * g.fscore.mean()]
        lines.append(f"\\multirow{{2}}{{*}}{{{LABEL[m]}}} & paper & " + " & ".join(f"{v:.1f}" for v in PAPER[m]) + r" \\")
        lines.append(" & ours & " + " & ".join(f"{v:.1f}" for v in ours) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_repro.tex").write_text("\n".join(lines) + "\n")


def tab_main(out):
    main, allv = load("1027-main"), load("1028-allviews")
    main = main[main.sampling != "all"]
    cols = [("reg_rate", "Reg."), ("posefail_auc05", "AUC@5"), ("posefail_auc30", "AUC@30"),
            ("fscore_pc@0.02", "F@2"), ("fscore_pc@0.05", "F@5"), ("fscore_pc@0.10", "F@10"),
            ("precision_pc@0.05", "P@5"), ("recall_pc@0.05", "R@5")]
    lines = [r"\begin{tabular}{llr l" + "c" * len(cols) + "}", r"\toprule",
             r"Views & Sampling & $N$ & Method & " + " & ".join(c[1] for c in cols) + r" \\", r"\midrule"]
    blocks = [(n, s, main[(main.n_views == n) & (main.sampling == s)]) for n in ["5", "10", "20"]
              for s in ["neighbor", "random"]] + [("all", "--", allv)]
    for bi, (n, s, g) in enumerate(blocks):
        means = {m: [100 * g[g.method == m][c].mean() for c, _ in cols] for m in METHODS}
        best = [bold_best([means[m][k] for m in METHODS]) for k in range(len(cols))]
        n_cases = len(g[g.method == "colmap"])
        for i, m in enumerate(METHODS):
            head = (f"\\multirow{{3}}{{*}}{{{n}}} & \\multirow{{3}}{{*}}{{{s}}} & \\multirow{{3}}{{*}}{{{n_cases}}}"
                    if i == 0 else " & & ")
            lines.append(f"{head} & {LABEL[m]} & " + " & ".join(best[k][i] for k in range(len(cols))) + r" \\")
        if bi < len(blocks) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_main.tex").write_text("\n".join(lines) + "\n")


def tab_scene(out):
    main, allv = load("1027-main"), load("1028-allviews")
    stats = pd.concat([pd.read_csv(f) for f in glob.glob(str(ROOT / "results" / "1027-main" / "match_stats_*of2.csv"))])
    kp = stats.groupby("scene").keypoints_mean.mean()
    nb = main[main.sampling == "neighbor"]
    rnd = main[(main.sampling == "random") & (main.method == "colmap")]
    lines = [r"\begin{tabular}{lcrcc ccc ccc}", r"\toprule",
             r" & & & \multicolumn{2}{c}{COLMAP reg.} & \multicolumn{3}{c}{F@5, neighbor 5--20} & \multicolumn{3}{c}{F@5, all views} \\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
             r"Scene & Type & SIFT kp. & nb. & rand. & COLMAP & VGGT & DA3 & COLMAP & VGGT & DA3 \\", r"\midrule"]
    scenes = sorted(set(main.scene), key=lambda s: (s in INDOOR, s))
    for s in scenes:
        reg_nb = nb[(nb.scene == s) & (nb.method == "colmap")].reg_rate.mean()
        reg_r = rnd[rnd.scene == s].reg_rate.mean()
        f_nb = bold_best([100 * nb[(nb.scene == s) & (nb.method == m)]["fscore_pc@0.05"].mean() for m in METHODS])
        f_all = bold_best([100 * allv[(allv.scene == s) & (allv.method == m)]["fscore_pc@0.05"].mean() for m in METHODS])
        name = s.replace("_", r"\_")
        lines.append(f"{name} & {'in' if s in INDOOR else 'out'} & {kp[s] / 1000:.1f}k & {100 * reg_nb:.0f} & "
                     f"{100 * reg_r:.0f} & " + " & ".join(f_nb) + " & " + " & ".join(f_all) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_scene.tex").write_text("\n".join(lines) + "\n")


def tab_efficiency(out):
    rows = []
    for run in ["1028-efficiency", "1028-allviews"]:
        for f in glob.glob(str(ROOT / "runs" / run / "*" / "*" / "*" / "info.json")):
            j = json.load(open(f))
            t = j.get("wall_s") or j["time_s"].get("end_to_end")
            rows.append({"method": j["method"], "n_views": str(j["case"]["n_views"]), "time": t,
                         "mem": j.get("peak_gpu_gb"), "n_img": len(j["case"]["images"])})
    d = pd.DataFrame(rows)
    order = ["5", "10", "20", "all"]
    g = d.groupby(["method", "n_views"]).agg(time=("time", "mean"), mem=("mem", "max"), n_img=("n_img", "mean"))
    lines = [r"\begin{tabular}{l" + "c" * len(order) + "c" * len(order) + "}", r"\toprule",
             r" & \multicolumn{4}{c}{Runtime (s)} & \multicolumn{4}{c}{Peak memory (GB)} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             r"Method & " + " & ".join(order) + " & " + " & ".join(order) + r" \\", r"\midrule"]
    for m in METHODS:
        t = [g.loc[(m, n), "time"] for n in order]
        mem = [g.loc[(m, n), "mem"] for n in order]
        fmt = lambda v: f"{v:.0f}" if v >= 10 else f"{v:.1f}"  # noqa: E731
        lines.append(f"{LABEL[m]} & " + " & ".join(fmt(v) for v in t) + " & " + " & ".join(f"{v:.1f}" for v in mem) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_efficiency.tex").write_text("\n".join(lines) + "\n")
    print("mean images in 'all':", g.xs("all", level="n_views").n_img.iloc[0])


def tab_nested(out):
    """Same 5 views evaluated inside 5-, 10- and 20-view reconstructions (scripts/nested_views.py)."""
    d = pd.read_csv(ROOT / "results" / "1027-main" / "nested_views.csv")
    has20 = set(map(tuple, d[d.n_context == 20][["scene", "seed"]].drop_duplicates().values))
    d = d[[(s, sd) in has20 for s, sd in zip(d.scene, d.seed)]]
    cols = [("posefail_auc05", "AUC@5"), ("fscore_pc@0.05", "F@5")]
    ctx = [5, 10, 20]
    lines = [r"\begin{tabular}{l" + "ccc" * len(cols) + "}", r"\toprule",
             " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{c[1]}}}" for c in cols) + r" \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             "Input views & " + " & ".join(" & ".join(str(c) for c in ctx) for _ in cols) + r" \\", r"\midrule"]
    for m in METHODS:
        vals = []
        for c, _ in cols:
            for n in ctx:
                g = d[(d.method == m) & (d.n_context == n)]
                vals.append("--" if c.startswith("fscore") and m == "colmap" else f"{100 * g[c].fillna(0).mean():.1f}")
        lines.append(f"{LABEL[m]} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_nested.tex").write_text("\n".join(lines) + "\n")
    print("nested (scene, seed) pairs:", len(has20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "report" / "tables"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tab_repro(out)
    tab_main(out)
    tab_scene(out)
    tab_efficiency(out)
    tab_nested(out)
    print("tables ->", out)


if __name__ == "__main__":
    main()
