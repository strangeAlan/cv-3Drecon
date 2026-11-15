"""Qualitative figure: each method's final point cloud (aligned exactly as in evaluate.py, including the
ICP refinement), seen from one input camera and colored by its distance to the GT cloud.

Points are projected with the GT pose and the GT (distorted) camera of that view and z-buffered;
pixels without a point stay white, so holes in a reconstruction show as white areas.

Usage: python scripts/make_qualitative.py 1027-main courtyard:nb_v10_s0 office:nb_v10_s0
       -> report/figures/fig_qualitative.pdf
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import evaluate as ev  # noqa: E402  (preloads libusb through common)
import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from common import ETH3D_ROOT, ROOT, eth3d_dataset  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

METHODS = ["colmap", "vggt", "da3"]
LABEL = {"colmap": "COLMAP", "vggt": "VGGT", "da3": "DA3"}
WIDTH = 600          # rendered image width (px)
ERR_MAX = 0.10       # color scale: 0 .. 10 cm
CMAP = "viridis"

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "font.size": 8,
                     "pdf.fonttype": 42})


def final_points(ds, run, method, scene, case):
    cdir = ROOT / "runs" / run / method / scene / case
    info = json.load(open(cdir / "info.json"))
    if info.get("failed"):
        return np.empty((0, 3)), np.empty(0)
    res = dict(np.load(cdir / "exports" / "mini_npz" / "results.npz"))
    gt = dict(np.load(cdir / "exports" / "gt_meta.npz", allow_pickle=True))
    pts, _ = ev.prediction_points(ds, scene, res, gt, info, cdir)
    ref = ev.gt_reference_cloud(ds, scene)
    tree = cKDTree(ref)
    pts, _ = ev.icp_sim3(pts, tree, ref)
    err, _ = tree.query(pts, k=1, workers=8)
    return pts, err


def render(pts, values, E, cam, scale):
    """Z-buffered splat of points into the (downscaled) distorted GT camera; NaN = no point."""
    H, W = int(round(cam.height * scale)), int(round(cam.width * scale))
    c = pts @ E[:3, :3].T + E[:3, 3]
    front = c[:, 2] > 0.1
    c, values = c[front], values[front]
    uv = np.asarray(cam.img_from_cam(c)) * scale
    u, v = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
    # 2x2 splats; all splat pixels sorted far to near, so the nearest point wins every pixel
    uu = np.concatenate([u + du for du in (0, 1) for _ in (0, 1)])
    vv = np.concatenate([v + dv for _ in (0, 1) for dv in (0, 1)])
    zz, val = np.tile(c[:, 2], 4), np.tile(values, 4)
    ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
    order = np.argsort(-zz[ok])
    img = np.full((H, W), np.nan)
    img[vv[ok][order], uu[ok][order]] = val[ok][order]
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("cases", nargs="+", help="scene:case_name")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="label:run:method columns inserted after COLMAP (e.g. COLMAP-RADIAL:1113-main:colmap_radial)")
    ap.add_argument("--out", default=str(ROOT / "report" / "figures" / "fig_qualitative.pdf"))
    ap.add_argument("--preview", help="also save a PNG here (for checking the figure without a PDF viewer)")
    args = ap.parse_args()
    ds = eth3d_dataset()
    cams = {}
    rows = []
    for spec in args.cases:
        scene, case = spec.split(":")
        cams.setdefault(scene, ev.gt_cameras(scene))
        data = ds.get_data(scene)
        by_name = {os.path.basename(p): i for i, p in enumerate(data.image_files)}
        names = json.load(open(ROOT / "runs" / args.run / "da3" / scene / case / "info.json"))["case"]["images"]
        name = names[len(names) // 2]  # one of the input views
        cam, E = cams[scene][name], data.extrinsics[by_name[name]]
        scale = WIDTH / cam.width
        rgb = cv2.cvtColor(cv2.imread(str(ETH3D_ROOT / scene / "images" / "dslr_images" / name)), cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (WIDTH, int(round(cam.height * scale))), interpolation=cv2.INTER_AREA)
        panels = [("Input view", rgb)]
        columns = [(LABEL["colmap"], args.run, "colmap")] + [tuple(e.split(":")) for e in args.extra] + \
                  [(LABEL[m], args.run, m) for m in METHODS[1:]]
        for label, run, m in columns:
            pts, err = final_points(ds, run, m, scene, case)
            panels.append((label, render(pts, err, E, cam, scale)))
            print(f"{scene} {case} {label}: {len(pts)} points, median error {np.median(err) * 100 if len(err) else np.nan:.1f} cm")
        rows.append((f"{scene}, {case.split('_')[1][1:]} views", panels))

    ncol = len(rows[0][1])
    fig, axes = plt.subplots(len(rows), ncol, figsize=(6.9, 6.9 / ncol * 0.69 * len(rows) + 0.4), squeeze=False)
    for i, (title, panels) in enumerate(rows):
        for j, (label, im) in enumerate(panels):
            ax = axes[i, j]
            if im.ndim == 3:
                ax.imshow(im)
            else:
                cm = plt.get_cmap(CMAP).copy()
                cm.set_bad("white")
                sm = ax.imshow(np.ma.masked_invalid(np.clip(im, 0, ERR_MAX) * 100), cmap=cm, vmin=0,
                               vmax=ERR_MAX * 100, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
            if i == 0:
                ax.set_title(label)
            if j == 0:
                ax.set_ylabel(title)
    fig.tight_layout(rect=(0, 0.08, 1, 1), h_pad=0.3, w_pad=0.3)
    cax = fig.add_axes([0.35, 0.05, 0.4, 0.018])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Distance to ground truth (cm); white = no point", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    fig.savefig(args.out, bbox_inches="tight", dpi=300)
    if args.preview:
        fig.savefig(args.preview, bbox_inches="tight", dpi=150)
    print("->", args.out)


if __name__ == "__main__":
    main()
