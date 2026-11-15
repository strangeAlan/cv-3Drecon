"""Report figures (static PDFs for the LaTeX report) from the result files of 1027-main + 1113-main (subsets),
1028-allviews + 1113-allviews (all views).

fig_fscore_views.pdf   F-score at 2 / 5 cm vs number of views, one panel per sampling scheme
fig_pose_views.pdf     pose AUC@5 vs number of views, one panel per sampling scheme
fig_prec_recall.pdf    precision vs recall at 5 cm of every subset reconstruction
fig_ba_scatter.pdf     per-case AUC@5 before vs after the BA refinement (DA3 -> DA3-BA, VGGT -> VGGT-BA)
fig_trajectory.pdf     camera centres of delivery_area (all views), top view, aligned to the GT
(runtime: see tab_efficiency in make_tables.py)

Colors: categorical slots 1-6 of the dataviz reference palette in their fixed order (blue, orange, aqua,
yellow, magenta, green), assigned per method and never re-used; every method also has its own marker and
the legend/direct labels, so identity never relies on color alone.

Usage: python scripts/make_figures.py [--out report/figures]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABEL = {"colmap": "COLMAP", "vggt": "VGGT", "da3": "DA3", "colmap_radial": "COLMAP-R", "da3ba": "DA3-BA",
         "vggtba": "VGGT-BA"}
COLOR = {"colmap": "#2a78d6", "vggt": "#eb6834", "da3": "#1baf7a", "colmap_radial": "#eda100", "da3ba": "#e87ba4",
         "vggtba": "#008300"}
MARKER = {"colmap": "o", "vggt": "s", "da3": "^", "colmap_radial": "D", "da3ba": "v", "vggtba": "P"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SAMPLING = {"neighbor": "Neighbor views (high overlap)", "random": "Random views (low overlap)"}
KEY = ["scene", "sampling", "n_views", "seed"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})


def load(methods):
    """Subset cases (5/10/20 views) that every method has; failed reconstructions count as 0."""
    d = pd.concat([pd.read_csv(ROOT / "results" / r / "metrics.csv") for r in ["1027-main", "1113-main"]])
    d = d[d.method.isin(methods)].copy()
    d = d[d.n_views.astype(str) != "all"]
    d["n_views"] = d.n_views.astype(int)
    for c in [c for c in d if c.startswith(("fscore_pc", "precision_pc", "recall_pc"))]:
        d[c] = d[c].fillna(0.0)
    d = d.drop_duplicates(KEY + ["method"], keep="last")
    return d[d.groupby(KEY).method.transform("nunique") == len(methods)]


PREVIEW = None  # directory for PNG previews (--preview)


def save(fig, out, name):
    fig.savefig(out / name, bbox_inches="tight")
    if PREVIEW is not None:
        fig.savefig(PREVIEW / name.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def mean_se(g, col):
    m = g[col].mean()
    se = g[col].std() / np.sqrt(g[col].count())
    return m, se.fillna(0)


def style_axis(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def curves(d, methods, rows, out, name, height=1.3):
    """rows: [(column, y label)]; one column of panels per sampling scheme, x = number of views."""
    samplings = list(SAMPLING)
    fig, axes = plt.subplots(len(rows), 2, figsize=(3.3, height * len(rows) + 0.45), sharex=True, sharey="row",
                             squeeze=False)
    views = sorted(d.n_views.unique())
    for j, s in enumerate(samplings):
        for i, (col, ylabel) in enumerate(rows):
            ax = axes[i, j]
            style_axis(ax)
            for m in methods:
                g = d[(d.method == m) & (d.sampling == s)].groupby("n_views")
                mu, se = mean_se(g, col)
                ax.fill_between(mu.index, mu - se, mu + se, color=COLOR[m], alpha=0.13, linewidth=0)
                ax.plot(mu.index, mu, color=COLOR[m], marker=MARKER[m], markersize=3.5, linewidth=1.2,
                        markeredgecolor="white", markeredgewidth=0.5, label=LABEL[m])
            ax.set_ylim(0, 1)
            ax.set_xticks(views)
            if i == 0:
                ax.set_title(SAMPLING[s], color=INK)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == len(rows) - 1:
                ax.set_xlabel("Number of input views")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(methods), 3), frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    nrow_leg = int(np.ceil(len(methods) / 3))
    fig.tight_layout(rect=(0, 0.045 * nrow_leg + 0.01, 1, 1), h_pad=0.6, w_pad=0.8)
    save(fig, out, name)


def fig_prec_recall(d, methods, out):
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    style_axis(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    for m in methods:
        x = d[d.method == m]
        ax.scatter(x["recall_pc@0.05"], x["precision_pc@0.05"], s=9, color=COLOR[m], marker=MARKER[m],
                   alpha=0.55, linewidths=0.3, edgecolors="white", label=LABEL[m])
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall (completeness) @ 5 cm")
    ax.set_ylabel("Precision (accuracy) @ 5 cm")
    ax.legend(frameon=False, loc="lower right", markerscale=1.5)
    fig.tight_layout()
    save(fig, out, "fig_prec_recall.pdf")


def fig_ba_scatter(out):
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.75), sharey=True)
    for ax, (before, after) in zip(axes, [("da3", "da3ba"), ("vggt", "vggtba")]):
        d = load([before, after]).pivot_table(index=KEY, columns="method", values="posefail_auc05").reset_index()
        ax.plot([0, 1], [0, 1], color=INK2, linewidth=0.6, linestyle=(0, (3, 2)))
        for s, mk in [("neighbor", "o"), ("random", "^")]:
            x = d[d.sampling == s]
            ax.scatter(x[before], x[after], s=10, marker=mk, color=COLOR[after], alpha=0.6, linewidths=0.3,
                       edgecolors="white", label=s)
        better = (d[after] > d[before] + 0.005).mean()
        ax.text(0.97, 0.04, f"better in {100 * better:.0f}%", ha="right", va="bottom", transform=ax.transAxes,
                fontsize=7, color=INK2)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{LABEL[before]} AUC@5")
        ax.set_ylabel(f"after BA" if before == "da3" else "")
        ax.set_title(f"{LABEL[before]} $\\rightarrow$ {LABEL[after]}")
        ax.grid(color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=mk, linestyle="", color=INK2, markersize=4, label=f"{s} views")
               for s, mk in [("neighbor", "o"), ("random", "^")]]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.6)
    save(fig, out, "fig_ba_scatter.pdf")


def _centers(E):
    E = np.asarray(E, np.float64)
    return -np.einsum("nji,nj->ni", E[:, :3, :3], E[:, :3, 3])


def _umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / xs.var(0).sum()
    return R, mu_d - s * R @ mu_s, s


def fig_trajectory(out, scene="delivery_area"):
    """Camera centres (Sim(3)-aligned to the GT centres) seen from above: the plane of the two largest
    principal directions of the GT centres."""
    methods = [("colmap", "1028-allviews"), ("colmap_radial", "1113-allviews"), ("da3", "1028-allviews"),
               ("da3ba", "1113-allviews")]
    fig, axes = plt.subplots(1, len(methods), figsize=(6.9, 1.75), sharex=True, sharey=True)
    for ax, (m, run) in zip(axes, methods):
        cdir = ROOT / "runs" / run / m / scene / "vall_s0"
        r = np.load(cdir / "exports" / "mini_npz" / "results.npz")
        g = np.load(cdir / "exports" / "gt_meta.npz", allow_pickle=True)
        P, Q = _centers(r["extrinsics"]), _centers(g["extrinsics"])
        R, t, s = _umeyama(P, Q)
        P = s * P @ R.T + t
        mu = Q.mean(0)
        _, _, Vt = np.linalg.svd(Q - mu)
        p2, q2 = (P - mu) @ Vt[:2].T, (Q - mu) @ Vt[:2].T
        for a, b in zip(q2, p2):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=INK2, linewidth=0.4)
        ax.scatter(q2[:, 0], q2[:, 1], s=9, color="#b4b2a9", label="ground truth", linewidths=0)
        ax.scatter(p2[:, 0], p2[:, 1], s=10, marker=MARKER[m], color=COLOR[m], linewidths=0.3, edgecolors="white",
                   label=LABEL[m])
        err = np.linalg.norm(P - Q, axis=1)
        info = json.load(open(cdir / "info.json"))
        ax.set_title(f"{LABEL[m]}: {sum(info['registered'])}/{len(info['registered'])} views\n"
                     f"median error {100 * np.median(err):.0f} cm, max {100 * err.max():.0f} cm", fontsize=7)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.4)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="", color="#b4b2a9", markersize=4, label="ground-truth centre"),
               Line2D([], [], color=INK2, linewidth=0.6, label="error")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.5)
    save(fig, out, "fig_trajectory.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "report" / "figures"))
    ap.add_argument("--preview", help="also write PNG previews into this directory")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    global PREVIEW
    if args.preview:
        PREVIEW = Path(args.preview)
        PREVIEW.mkdir(parents=True, exist_ok=True)
    geo = ["colmap", "colmap_radial", "vggt", "da3"]
    d = load(geo)
    curves(d, geo, [("fscore_pc@0.02", "F-score @ 2 cm"), ("fscore_pc@0.05", "F-score @ 5 cm")], out,
           "fig_fscore_views.pdf")
    fig_prec_recall(d, geo, out)
    pose = ["colmap", "colmap_radial", "vggt", "vggtba", "da3", "da3ba"]
    curves(load(pose), pose, [("posefail_auc05", "Pose AUC@5")], out, "fig_pose_views.pdf", height=1.6)
    fig_ba_scatter(out)
    fig_trajectory(out)
    print("figures ->", out)


if __name__ == "__main__":
    main()
