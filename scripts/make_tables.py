"""LaTeX tables for the report, straight from the result files (no hand-copied numbers).

Runs: 1027-main / 1028-allviews (COLMAP, VGGT, DA3), 1113-main / 1113-allviews (COLMAP-R = COLMAP with the RADIAL
camera model, VGGT-BA, DA3-BA, dense stage), 1113-da3ba-explore + 1115-abl-* (ablations), 1028-efficiency +
1115-efficiency / 1115-allviews-time (runtime).

tab_repro.tex       DA3-protocol reproduction (1027-repro) vs. published numbers
tab_pose.tex        pose AUC@5 / AUC@30 per setting, all methods
tab_geometry.tex    F-score at 2 / 5 cm per setting
tab_main.tex        all metrics per setting (appendix)
tab_scene.tex       per scene: SIFT keypoints, COLMAP registration, F@5 (neighbor subsets, all views)
tab_efficiency.tex  end-to-end runtime and peak GPU memory
tab_nested.tex      same 5 views inside 5/10/20-view reconstructions (results/1027-main/nested_views.csv)
tab_ablation.tex    DA3-BA design choices on the 13 exploration cases
tab_dense.tex       dense matching on the DA3-BA cameras

Usage: python scripts/make_tables.py [--out report/tables]
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABEL = {"colmap": "COLMAP", "colmap_radial": "COLMAP-R", "vggt": "VGGT", "vggtba": "VGGT-BA", "da3": "DA3",
         "da3ba": "DA3-BA", "hybrid_da3ba": "DA3-BA + MVS", "hybrid_da3ba_fill": "DA3-BA + MVS + fill"}
INDOOR = {"kicker", "office", "pipes", "relief", "relief_2", "terrains"}  # ETH3D scene categories
PAPER = {"vggt": (26.3, 80.8, 57.2), "da3": (48.4, 91.2, 79.0)}  # DA3 paper, ETH3D, DA3-Giant
KEY = ["scene", "sampling", "n_views", "seed"]
SETTINGS = [("neighbor", "5"), ("neighbor", "10"), ("neighbor", "20"), ("random", "5"), ("random", "10"),
            ("random", "20"), ("all", "all")]
SET_HEAD = (r" & \multicolumn{3}{c}{Neighbor views} & \multicolumn{3}{c}{Random views} & All \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-8}",
            r"Method & 5 & 10 & 20 & 5 & 10 & 20 & views \\")


def load(runs, methods):
    """Metrics of the given runs, restricted to cases that every method in `methods` has."""
    d = pd.concat([pd.read_csv(ROOT / "results" / r / "metrics.csv") for r in runs], ignore_index=True)
    d = d[d.method.isin(methods)].copy()
    d["n_views"] = d.n_views.astype(str)
    d["reg_rate"] = d.n_registered / d.n_input
    for c in [c for c in d if c.startswith(("fscore_pc", "precision_pc", "recall_pc"))]:
        d[c] = d[c].fillna(0.0)  # failed reconstructions count as 0
    d = d.drop_duplicates(KEY + ["method"], keep="last")
    return d[d.groupby(KEY).method.transform("nunique") == len(methods)]


def load_all(methods):
    return pd.concat([load(["1027-main", "1113-main"], methods).query("sampling != 'all'"),
                      load(["1028-allviews", "1113-allviews"], methods)], ignore_index=True)


def bold_best(vals, higher=True, fmt="{:.1f}"):
    finite = [v for v in vals if np.isfinite(v)]
    best = (max(finite) if higher else min(finite)) if finite else np.nan
    return ["--" if not np.isfinite(v) else (f"\\textbf{{{fmt.format(v)}}}" if np.isclose(v, best) else fmt.format(v))
            for v in vals]


def setting_table(d, methods, blocks, name):
    """Rows = methods, columns = the 7 settings; one row block per (metric, title)."""
    lines = [r"\begin{tabular}{l" + "c" * len(SETTINGS) + "}", r"\toprule", *SET_HEAD]
    for col, title in blocks:
        lines += [r"\midrule", f"\\multicolumn{{8}}{{l}}{{\\emph{{{title}}}}} \\\\"]
        cells = {m: [] for m in methods}
        for s, n in SETTINGS:
            g = d[(d.sampling == s) & (d.n_views == n)]
            for m, v in zip(methods, bold_best([100 * g[g.method == m][col].mean() for m in methods])):
                cells[m].append(v)
        lines += [f"{LABEL[m]} & " + " & ".join(cells[m]) + r" \\" for m in methods]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / name).write_text("\n".join(lines) + "\n")


def tab_repro():
    d = pd.read_csv(ROOT / "results" / "1027-repro" / "metrics.csv")
    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             r"Method & Source & AUC@3 & AUC@30 & F1@25\,cm \\", r"\midrule"]
    for m in ["vggt", "da3"]:
        g = d[d.method == m]
        ours = [100 * g.pose_auc03.mean(), 100 * g.pose_auc30.mean(), 100 * g.fscore.mean()]
        lines.append(f"\\multirow{{2}}{{*}}{{{LABEL[m]}}} & paper & " + " & ".join(f"{v:.1f}" for v in PAPER[m]) + r" \\")
        lines.append(" & ours & " + " & ".join(f"{v:.1f}" for v in ours) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_repro.tex").write_text("\n".join(lines) + "\n")


def tab_pose():
    methods = ["colmap", "colmap_radial", "vggt", "vggtba", "da3", "da3ba"]
    d = load_all(methods)
    setting_table(d, methods, [("posefail_auc05", "AUC@5"), ("posefail_auc30", "AUC@30")], "tab_pose.tex")
    print("pose cases:", len(d) // len(methods))


def tab_geometry():
    methods = ["colmap", "colmap_radial", "vggt", "da3", "da3ba"]
    d = load_all(methods)
    setting_table(d, methods, [("fscore_pc@0.02", "F-score at 2\\,cm"), ("fscore_pc@0.05", "F-score at 5\\,cm")],
                  "tab_geometry.tex")


def tab_main():
    methods = ["colmap", "colmap_radial", "vggt", "da3", "da3ba"]
    d = load_all(methods)
    cols = [("reg_rate", "Reg."), ("posefail_auc05", "AUC@5"), ("posefail_auc30", "AUC@30"),
            ("fscore_pc@0.02", "F@2"), ("fscore_pc@0.05", "F@5"), ("fscore_pc@0.10", "F@10"),
            ("precision_pc@0.05", "P@5"), ("recall_pc@0.05", "R@5")]
    k = len(methods)
    lines = [r"\begin{tabular}{llr l" + "c" * len(cols) + "}", r"\toprule",
             r"Views & Sampling & $N$ & Method & " + " & ".join(c[1] for c in cols) + r" \\", r"\midrule"]
    for bi, (s, n) in enumerate(SETTINGS):
        g = d[(d.sampling == s) & (d.n_views == n)]
        best = [bold_best([100 * g[g.method == m][c].mean() for m in methods]) for c, _ in cols]
        for i, m in enumerate(methods):
            head = (f"\\multirow{{{k}}}{{*}}{{{n}}} & \\multirow{{{k}}}{{*}}{{{'--' if s == 'all' else s}}} & "
                    f"\\multirow{{{k}}}{{*}}{{{len(g[g.method == m])}}}" if i == 0 else " & & ")
            lines.append(f"{head} & {LABEL[m]} & " + " & ".join(b[i] for b in best) + r" \\")
        if bi < len(SETTINGS) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_main.tex").write_text("\n".join(lines) + "\n")


def tab_scene():
    methods = ["colmap", "colmap_radial", "vggt", "da3"]
    main = load(["1027-main", "1113-main"], methods).query("sampling != 'all'")
    allv = load(["1028-allviews", "1113-allviews"], methods)
    stats = pd.concat([pd.read_csv(f) for f in glob.glob(str(ROOT / "results" / "1027-main" / "match_stats_*of2.csv"))])
    kp = stats.groupby("scene").keypoints_mean.mean()
    nb = main[main.sampling == "neighbor"]
    rnd = main[main.sampling == "random"]
    lines = [r"\begin{tabular}{lr cc cccc cccc}", r"\toprule",
             r" & & \multicolumn{2}{c}{COLMAP reg.} & \multicolumn{4}{c}{F@5, neighbor 5--20 views}"
             r" & \multicolumn{4}{c}{F@5, all views} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-8}\cmidrule(lr){9-12}",
             r"Scene & SIFT kp. & nb. & rand. & " + " & ".join(LABEL[m] for m in methods) * 1 + " & "
             + " & ".join(LABEL[m] for m in methods) + r" \\"]
    group = None
    for s in sorted(set(main.scene), key=lambda s: (s in INDOOR, s)):
        if (s in INDOOR) != group:
            group = s in INDOOR
            lines += [r"\midrule", f"\\multicolumn{{12}}{{l}}{{\\emph{{{'Indoor' if group else 'Outdoor'}}}}} \\\\"]
        reg_nb = nb[(nb.scene == s) & (nb.method == "colmap")].reg_rate.mean()
        reg_r = rnd[(rnd.scene == s) & (rnd.method == "colmap")].reg_rate.mean()
        f_nb = bold_best([100 * nb[(nb.scene == s) & (nb.method == m)]["fscore_pc@0.05"].mean() for m in methods])
        f_all = bold_best([100 * allv[(allv.scene == s) & (allv.method == m)]["fscore_pc@0.05"].mean() for m in methods])
        lines.append(f"{s.replace('_', chr(92) + '_')} & {kp[s] / 1000:.1f}k & "
                     f"{100 * reg_nb:.0f} & {100 * reg_r:.0f} & " + " & ".join(f_nb) + " & " + " & ".join(f_all) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_scene.tex").write_text("\n".join(lines) + "\n")


def _ba_rows(runs):
    """Runtime / peak memory of DA3-BA and VGGT-BA: feed-forward model end to end + SIFT matching + BA; memory is
    the larger of the feed-forward model's and the SIFT process's peak."""
    rows = []
    for run in runs:
        for f in glob.glob(str(ROOT / "runs" / run / "*ba" / "*" / "*" / "info.json")):
            j = json.load(open(f))
            ff = j["method"][:-2]
            ff_info = json.load(open(ROOT / "runs" / json.load(open(ROOT / "manifests" / f"{run}.json")).get(
                "source", run) / ff / j["case"]["scene"] / Path(f).parent.name / "info.json"))
            rows.append({"method": j["method"], "n_views": str(j["case"]["n_views"]), "time": sum(j["time_s"].values()),
                         "mem": max(j.get("peak_gpu_gb") or 0, ff_info.get("peak_gpu_gb") or 0)})
    return rows


def tab_efficiency():
    rows = []
    for run in ["1028-efficiency", "1028-allviews"]:
        for f in glob.glob(str(ROOT / "runs" / run / "*" / "*" / "*" / "info.json")):
            j = json.load(open(f))
            rows.append({"method": j["method"], "n_views": str(j["case"]["n_views"]),
                         "time": j.get("wall_s") or j["time_s"].get("end_to_end"), "mem": j.get("peak_gpu_gb")})
    rows += _ba_rows([r for r in ["1115-efficiency", "1115-allviews-time"] if (ROOT / "runs" / r).exists()])
    d = pd.DataFrame(rows)
    order = ["5", "10", "20", "all"]
    g = d.groupby(["method", "n_views"]).agg(time=("time", "mean"), mem=("mem", "max"))
    lines = [r"\begin{tabular}{l" + "c" * 2 * len(order) + "}", r"\toprule",
             r" & \multicolumn{4}{c}{Runtime (s)} & \multicolumn{4}{c}{Peak memory (GB)} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             r"Method & " + " & ".join(order) + " & " + " & ".join(order) + r" \\", r"\midrule"]
    fmt = lambda v: "--" if not np.isfinite(v) else (f"{v:.0f}" if v >= 10 else f"{v:.1f}")  # noqa: E731
    for m in ["colmap", "vggt", "da3", "vggtba", "da3ba"]:
        t = [g.time.get((m, n), np.nan) for n in order]
        mem = [g.mem.get((m, n), np.nan) for n in order]
        lines.append(f"{LABEL[m]} & " + " & ".join(fmt(v) for v in t) + " & "
                     + " & ".join("--" if not np.isfinite(v) else f"{v:.1f}" for v in mem) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_efficiency.tex").write_text("\n".join(lines) + "\n")


def tab_nested():
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
    for m in ["colmap", "vggt", "da3"]:
        vals = []
        for c, _ in cols:
            for n in ctx:
                g = d[(d.method == m) & (d.n_context == n)]
                vals.append("--" if c.startswith("fscore") and m == "colmap" else f"{100 * g[c].fillna(0).mean():.1f}")
        lines.append(f"{LABEL[m]} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_nested.tex").write_text("\n".join(lines) + "\n")


ABLATIONS = [("1113-da3ba-explore", "DA3-BA"), ("1115-abl-triangulate", "start from triangulation"),
             ("1115-abl-simpleradial", "1-parameter radial camera"), ("1115-abl-fixintr", "fixed DA3 intrinsics"),
             ("1115-abl-onestage", "no coarse-to-fine BA")]
GROUPS = [("colmap_fails", "(a)"), ("colmap_better", "(b)"), ("da3_good", "(c)")]  # explained in the caption


def tab_ablation():
    cases = json.load(open(ROOT / "manifests" / "1113-da3ba-explore.json"))["cases"]
    groups = pd.DataFrame([{**{k: c[k] for k in KEY}, "group": c["group"]} for c in cases])
    groups["n_views"] = groups.n_views.astype(str)
    rows = []
    base = pd.concat([pd.read_csv(ROOT / "results" / r / "metrics.csv") for r in ["1027-main", "1113-main"]])
    base = base[base.method.isin(["colmap", "colmap_radial", "da3"])].assign(variant=lambda x: x.method.map(LABEL))
    rows.append(base)
    for run, name in ABLATIONS:
        p = ROOT / "results" / run / "metrics.csv"
        if p.exists():
            m = pd.read_csv(p)
            rows.append(m[m.method == "da3ba"].assign(variant=name))
    d = pd.concat(rows, ignore_index=True)
    d["n_views"] = d.n_views.astype(str)
    d = d.merge(groups, on=KEY)
    lines = [r"\begin{tabular}{l cccc c}", r"\toprule",
             r" & \multicolumn{4}{c}{AUC@5} & AUC@30 \\", r"\cmidrule(lr){2-5}\cmidrule(lr){6-6}",
             "Method & " + " & ".join(g[1] for g in GROUPS) + r" & all & all \\"]
    order = ["COLMAP", "COLMAP-R", "DA3"] + [n for _, n in ABLATIONS]
    for v in order:
        x = d[d.variant == v]
        if x.empty:
            continue
        if v in ("COLMAP", "DA3-BA"):
            lines.append(r"\midrule")
        vals = [100 * x[x.group == g].posefail_auc05.mean() for g, _ in GROUPS]
        vals += [100 * x.posefail_auc05.mean(), 100 * x.posefail_auc30.mean()]
        label = v if v in ("COLMAP", "COLMAP-R", "DA3", "DA3-BA") else "\\quad " + v
        lines.append(f"{label} & " + " & ".join(f"{u:.1f}" for u in vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_ablation.tex").write_text("\n".join(lines) + "\n")
    print("ablation cases per variant:", d.groupby("variant").size().to_dict())


def tab_dense():
    methods = ["colmap_radial", "da3", "da3ba", "hybrid_da3ba", "hybrid_da3ba_fill"]
    cols = [("fscore_pc@0.02", "F@2"), ("fscore_pc@0.05", "F@5"), ("precision_pc@0.05", "P@5"), ("recall_pc@0.05", "R@5")]
    sub = load(["1027-main", "1113-main"], methods).query("sampling != 'all'")
    allv = load(["1028-allviews", "1113-allviews"], methods)
    lines = [r"\begin{tabular}{l cccc cccc}", r"\toprule",
             f" & \\multicolumn{{4}}{{c}}{{Subsets ({len(sub) // len(methods)})}} & \\multicolumn{{4}}{{c}}{{All views ({len(allv) // len(methods)})}} \\\\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             "Method & " + " & ".join(c[1] for c in cols) * 1 + " & " + " & ".join(c[1] for c in cols) + r" \\", r"\midrule"]
    cells = {m: [] for m in methods}
    for d in (sub, allv):
        for c, _ in cols:
            for m, v in zip(methods, bold_best([100 * d[d.method == m][c].mean() for m in methods])):
                cells[m].append(v)
    lines += [f"{LABEL[m]} & " + " & ".join(cells[m]) + r" \\" for m in methods]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_dense.tex").write_text("\n".join(lines) + "\n")


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "report" / "tables"))
    args = ap.parse_args()
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    for f in [tab_repro, tab_pose, tab_geometry, tab_main, tab_scene, tab_efficiency, tab_nested, tab_ablation,
              tab_dense]:
        f()
    print("tables ->", OUT)


if __name__ == "__main__":
    main()
