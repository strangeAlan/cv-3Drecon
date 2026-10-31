"""Shared helpers: ETH3D scene access (via the DA3 benchmark loader), manifests,
and the per-case output layout that every method writes.

Every method writes, for one case (scene, view count, seed):
    <case_dir>/exports/mini_npz/results.npz   depth [N,H,W], extrinsics [N,3|4,4] (w2c, OpenCV),
                                              intrinsics [N,3,3] at depth resolution
    <case_dir>/exports/gt_meta.npz            GT extrinsics/intrinsics/image_files of the same N views
    <case_dir>/info.json                      timing, registered views, method notes
This is the format the DA3 benchmark consumes, so all methods share one evaluation path.
"""
import ctypes
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

# open3d needs libusb-1.0.so.0, which this server lacks; preload the copy shipped by
# the libusb-package wheel so the dynamic linker resolves it by soname.
try:
    import libusb_package

    ctypes.CDLL(str(Path(libusb_package.__file__).parent / "libusb-1.0.so"), mode=ctypes.RTLD_GLOBAL)
except (ImportError, OSError):
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "Depth-Anything-3" / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "vggt"))

ETH3D_ROOT = ROOT / "data" / "eth3d"


def eth3d_dataset():
    from depth_anything_3.bench.datasets.eth3d import ETH3D

    ETH3D.data_root = str(ETH3D_ROOT)
    return ETH3D()


def load_config(path):
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def case_name(n_views, seed, sampling="random"):
    # random subsets keep the original names used by the 1027 runs
    prefix = "" if sampling == "random" or n_views == "all" else "nb_"
    return f"{prefix}v{n_views}_s{seed}"


def neighbor_subset(extrinsics, k, rng):
    """Random anchor view plus its k-1 nearest views (camera distance, normalized by the
    median pairwise distance, plus optical-axis angle / 60 deg). Uses GT poses only to
    choose the input set, i.e. to control view overlap; no method sees them."""
    R, t = extrinsics[:, :3, :3], extrinsics[:, :3, 3]
    centers = -np.einsum("nji,nj->ni", R, t)
    axes = R[:, 2, :]  # optical axis in world coordinates
    dist = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
    scale = np.median(dist[np.triu_indices(len(dist), 1)])
    angle = np.degrees(np.arccos(np.clip(axes @ axes.T, -1, 1)))
    anchor = rng.randrange(len(extrinsics))
    score = dist[anchor] / scale + angle[anchor] / 60.0
    return sorted(np.argsort(score, kind="stable")[:k].tolist())


def make_manifest(cfg):
    """Fixed view subsets per (scene, sampling, view count, seed); identical for all methods.

    sampling: "random" = uniform subset (overlap left to chance, often low),
              "neighbor" = anchor + nearest views (overlap controlled, high).
    """
    ds = eth3d_dataset()
    cases = []
    for scene in cfg["scenes"]:
        data = ds.get_data(scene)
        files = data.image_files
        n = len(files)
        for k in cfg["view_counts"]:
            k_eff = n if k == "all" else int(k)
            if k_eff > n:
                continue
            for sampling in cfg.get("sampling", ["random"]):
                if k_eff == n and sampling != cfg.get("sampling", ["random"])[0]:
                    continue  # the full set does not depend on the sampling
                for seed in cfg["seeds"]:
                    if k_eff == n and seed != cfg["seeds"][0]:
                        continue  # nor on the seed
                    rng = random.Random(seed)
                    if k_eff == n:
                        idx = list(range(n))
                    elif sampling == "random":
                        idx = sorted(rng.sample(range(n), k_eff))
                    else:
                        idx = neighbor_subset(data.extrinsics, k_eff, rng)
                    cases.append({
                        "scene": scene,
                        "n_views": "all" if k_eff == n else k_eff,
                        "sampling": "all" if k_eff == n else sampling,
                        "seed": seed,
                        "indices": idx,
                        "images": [os.path.basename(files[i]) for i in idx],
                    })
    if cfg.get("order") == "seed_major":  # complete seeds first, so a partial run is still balanced
        n_order = {k: i for i, k in enumerate(cfg["view_counts"])}
        cases.sort(key=lambda c: (c["seed"], n_order.get(c["n_views"], -1), c["sampling"], c["scene"]))
    return cases


def case_gt(ds, case):
    data = ds.get_data(case["scene"])
    idx = case["indices"]
    return {
        "image_files": [data.image_files[i] for i in idx],
        "extrinsics": data.extrinsics[idx],
        "intrinsics": data.intrinsics[idx],
    }


def case_dir(run_dir, method, case):
    return Path(run_dir) / method / case["scene"] / case_name(case["n_views"], case["seed"],
                                                             case.get("sampling", "random"))


def save_case(out_dir, depth, extrinsics, intrinsics, gt, info):
    """gt is the GT dict restricted to the same views as the predictions."""
    out_dir = Path(out_dir)
    (out_dir / "exports" / "mini_npz").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "exports" / "mini_npz" / "results.npz",
                        depth=depth.astype(np.float32),
                        extrinsics=extrinsics.astype(np.float32),
                        intrinsics=intrinsics.astype(np.float32))
    np.savez_compressed(out_dir / "exports" / "gt_meta.npz",
                        extrinsics=gt["extrinsics"], intrinsics=gt["intrinsics"],
                        image_files=np.array(gt["image_files"], dtype=object))
    with open(out_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)


def is_done(out_dir):
    return (Path(out_dir) / "info.json").exists()


GRID_LONG = 1600  # common depth grid (= COLMAP dense image size) for point clouds


def resample_depth(depth, ixt, hw, max_ratio=1.1):
    """Bilinear depth resampling to grid hw = (h, w), scaling the intrinsics.

    Pixels whose source 2x2 neighbourhood spans a depth discontinuity (max/min >
    max_ratio) or touches an invalid pixel are dropped, so no points are interpolated
    between foreground and background.
    """
    h0, w0 = depth.shape
    h, w = hw
    d = np.where(np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32)
    up = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
    kernel = np.ones((2, 2), np.uint8)
    hi = cv2.dilate(d, kernel)
    lo = cv2.erode(np.where(d > 0, d, np.inf).astype(np.float32), kernel)
    ok = (hi <= max_ratio * lo) & np.isfinite(lo)
    # a target pixel's bilinear footprint starts at the source pixel up-left of its center
    ys = np.clip(np.floor((np.arange(h) + 0.5) * h0 / h - 0.5).astype(int), 0, h0 - 1)
    xs = np.clip(np.floor((np.arange(w) + 0.5) * w0 / w - 0.5).astype(int), 0, w0 - 1)
    up = up * ok[np.ix_(ys, xs)]
    K = ixt.copy().astype(np.float64)
    K[0] *= w / w0
    K[1] *= h / h0
    return up, K


def pc_grid(orig_hw):
    """Common evaluation grid: longer side GRID_LONG px, aspect of the original image."""
    h, w = orig_hw
    s = GRID_LONG / max(h, w)
    return int(round(h * s)), int(round(w * s))


def umeyama(src, dst):
    """Least-squares similarity dst ~ s R src + t (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / xs.var(0).sum()
    return R, mu_d - s * R @ mu_s, s


class GpuMemSampler:
    """Peak GPU memory of this process (all devices), sampled every `interval` s through NVML.
    Process-level, so it covers PyTorch and COLMAP's CUDA kernels alike (includes the CUDA context).

        with GpuMemSampler() as mem:
            ...
        mem.peak_gb
    """

    def __init__(self, interval=0.1):
        import threading

        import pynvml

        pynvml.nvmlInit()
        self.nvml = pynvml
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        self.interval, self.peak_bytes, self.pid = interval, 0, os.getpid()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self):
        used = 0
        for h in self.handles:
            for p in self.nvml.nvmlDeviceGetComputeRunningProcesses(h):
                if p.pid == self.pid and p.usedGpuMemory:
                    used += p.usedGpuMemory
        self.peak_bytes = max(self.peak_bytes, used)

    def _run(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self._sample()

    @property
    def peak_gb(self):
        return self.peak_bytes / 2**30
