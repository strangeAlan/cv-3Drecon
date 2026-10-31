"""Report figures from results/<run>/metrics.csv (static PDFs for the LaTeX report).

fig_fscore_views.pdf   F-score at 2/5/10 cm vs number of views, one panel per sampling scheme
fig_prec_recall.pdf    precision vs recall at 5 cm (accuracy/completeness trade-off)
(runtime: see tab_efficiency in make_tables.py; 1027-main time_s is inference-only for VGGT/DA3)

Colors: the first three categorical slots of the dataviz reference palette (validated for all
pairs in light mode); every method also has its own marker and a direct label, so identity
never relies on color alone.

Usage: python scripts/make_figures.py <run_name> [--out report/figures]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["colmap", "vggt", "da3"]
LABEL = {"colmap": "COLMAP", "vggt": "VGGT", "da3": "DA3"}
COLOR = {"colmap": "#2a78d6", "vggt": "#eb6834", "da3": "#1baf7a"}
MARKER = {"colmap": "o", "vggt": "s", "da3": "^"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SAMPLING = {"neighbor": "Neighbor views (high overlap)", "random": "Random views (low overlap)"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})


def load(run):
    d = pd.read_csv(ROOT / "results" / run / "metrics.csv")
    d = d[d.method.isin(METHODS)].copy()
    d = d[d.n_views.astype(str) != "all"]
    d["n_views"] = d.n_views.astype(int)
    for c in [c for c in d if c.startswith(("fscore_pc", "precision_pc", "recall_pc"))]:
        d[c] = d[c].fillna(0.0)  # failed reconstructions count as 0
    key = ["scene", "sampling", "n_views", "seed"]  # keep cases evaluated for every method
    full = d.groupby(key).method.transform("nunique") == d.method.nunique()
    return d[full]


def mean_se(g, col):
    m = g[col].mean()
    se = g[col].std() / np.sqrt(g[col].count())
    return m, se.fillna(0)


def style_axis(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def fig_fscore(d, out):
    samplings = [s for s in SAMPLING if s in set(d.sampling)]
    thresholds = ["0.02", "0.05", "0.10"]
    fig, axes = plt.subplots(len(thresholds), len(samplings), figsize=(3.3, 1.25 * len(thresholds) + 0.3),
                             sharex=True, sharey="row", squeeze=False)
    views = sorted(d.n_views.unique())
    for j, s in enumerate(samplings):
        for i, th in enumerate(thresholds):
            ax = axes[i, j]
            style_axis(ax)
            for m in METHODS:
                g = d[(d.method == m) & (d.sampling == s)].groupby("n_views")
                mu, se = mean_se(g, f"fscore_pc@{th}")
                ax.fill_between(mu.index, mu - se, mu + se, color=COLOR[m], alpha=0.15, linewidth=0)
                ax.plot(mu.index, mu, color=COLOR[m], marker=MARKER[m], markersize=3.5, linewidth=1.2,
                        markeredgecolor="white", markeredgewidth=0.5, label=LABEL[m])
            ax.set_ylim(0, 1)
            ax.set_xticks(views)
            if i == 0:
                ax.set_title(SAMPLING[s], color=INK)
            if j == 0:
                ax.set_ylabel(f"F-score @ {float(th) * 100:.0f} cm")
            if i == len(thresholds) - 1:
                ax.set_xlabel("Number of input views")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.6, w_pad=0.8)
    fig.savefig(out / "fig_fscore_views.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_prec_recall(d, out):
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    style_axis(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    for m in METHODS:
        x = d[d.method == m]
        ax.scatter(x["recall_pc@0.05"], x["precision_pc@0.05"], s=9, color=COLOR[m], marker=MARKER[m],
                   alpha=0.55, linewidths=0.3, edgecolors="white", label=LABEL[m])
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall (completeness) @ 5 cm")
    ax.set_ylabel("Precision (accuracy) @ 5 cm")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "fig_prec_recall.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_runtime(d, out):
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    style_axis(ax)
    ok = d[~d.get("failed", pd.Series(False, index=d.index)).fillna(False).astype(bool)]
    for m in METHODS:
        g = ok[ok.method == m].groupby("n_views")["time_s"]
        med = g.median()
        ax.plot(med.index, med, color=COLOR[m], marker=MARKER[m], markersize=3.5, linewidth=1.2,
                markeredgecolor="white", markeredgewidth=0.5, label=LABEL[m])
        ax.annotate(f"{med.iloc[-1]:.0f} s" if med.iloc[-1] >= 10 else f"{med.iloc[-1]:.1f} s",
                    (med.index[-1], med.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=7, color=INK2)
    ax.set_yscale("log")
    ax.set_xticks(sorted(ok.n_views.unique()))
    ax.set_xlabel("Number of input views")
    ax.set_ylabel("Runtime per scene (s, median)")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0, 0.62))
    fig.tight_layout()
    fig.savefig(out / "fig_runtime.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default=str(ROOT / "report" / "figures"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = load(args.run)
    fig_fscore(d, out)
    fig_prec_recall(d, out)
    print(f"{len(d)} rows, figures -> {out}")


if __name__ == "__main__":
    main()
