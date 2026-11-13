"""How well do COLMAP camera models represent the ETH3D lenses?

Each model is fitted (least squares on a 61 x 41 pixel grid) to the GT THIN_PRISM_FISHEYE calibration of
every scene; the residual is the pixel distance between the model's and the GT projection of the same ray.
SIMPLE_RADIAL is COLMAP's default; DA3-BA and colmap_radial use RADIAL.

fig_camera_model.pdf    residual over the image for SIMPLE_RADIAL and RADIAL (one scene)
results/camera_model_fit.csv   RMS / max residual and fitted focal error per scene and model

Usage: python scripts/make_camera_model_fig.py [--scene courtyard]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import evaluate as ev  # noqa: E402  (preloads libusb through common)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pycolmap  # noqa: E402
from common import ROOT  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

SCENES = ["courtyard", "delivery_area", "electro", "facade", "kicker", "office", "pipes", "playground", "relief",
          "relief_2", "terrains"]
MODELS = {"SIMPLE_RADIAL": lambda f, W, H: [f, W / 2, H / 2, 0],
          "RADIAL": lambda f, W, H: [f, W / 2, H / 2, 0, 0],
          "OPENCV": lambda f, W, H: [f, f, W / 2, H / 2, 0, 0, 0, 0]}
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 8, "pdf.fonttype": 42})


def fit(gt, model, nx=61, ny=41):
    W, H = gt.width, gt.height
    u, v = np.meshgrid(np.linspace(0, W, nx), np.linspace(0, H, ny))
    uv = np.stack([u.ravel(), v.ravel()], 1)
    X = np.concatenate([np.asarray(gt.cam_from_img(uv))[:, :2], np.ones((len(uv), 1))], 1)

    def res(q):
        return (np.asarray(pycolmap.Camera(model=model, width=W, height=H, params=q).img_from_cam(X)) - uv).ravel()

    r = least_squares(res, MODELS[model](gt.params[0], W, H), method="lm", max_nfev=5000)
    return r.x, np.linalg.norm(r.fun.reshape(-1, 2), axis=1).reshape(ny, nx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="courtyard")
    args = ap.parse_args()
    rows, maps = [], {}
    for scene in SCENES:
        gt = list(ev.gt_cameras(scene).values())[0]
        for model in MODELS:
            params, err = fit(gt, model)
            rows.append({"scene": scene, "model": model, "rms_px": float(np.sqrt((err ** 2).mean())),
                         "max_px": float(err.max()), "focal_err_pct": 100 * (params[0] / gt.params[0] - 1)})
            if scene == args.scene:
                maps[model] = err
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "results" / "camera_model_fit.csv", index=False)
    print(d.groupby("model")[["rms_px", "max_px", "focal_err_pct"]].agg(["mean", "min", "max"]).round(2).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.35))
    for ax, model in zip(axes, ["SIMPLE_RADIAL", "RADIAL"]):
        im = ax.imshow(maps[model], cmap="viridis", vmin=0, vmax=30, extent=(0, 6048, 4032, 0))
        r = d[(d.scene == args.scene) & (d.model == model)].iloc[0]
        name = {"SIMPLE_RADIAL": "1-parameter radial (default)", "RADIAL": "2-parameter radial"}[model]
        ax.set_title(f"{name}\nRMS {r.rms_px:.1f} px, max {r.max_px:.0f} px", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(rect=(0, 0, 0.88, 1), w_pad=0.4)
    cax = fig.add_axes([0.89, 0.12, 0.02, 0.72])
    cb = fig.colorbar(im, cax=cax, extend="max")
    cb.set_label("Residual (px)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    fig.savefig(ROOT / "report" / "figures" / "fig_camera_model.pdf", bbox_inches="tight")
    fig.savefig(Path(sys.argv[0]).resolve().parents[1] / "results" / "fig_camera_model_preview.png", dpi=150,
                bbox_inches="tight")


if __name__ == "__main__":
    main()
