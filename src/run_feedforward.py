"""Run a feed-forward model (VGGT or DA3) on every case of a manifest.

Usage: python src/run_feedforward.py configs/1027-prelim.yaml --method vggt|da3
"""
import argparse
import json
import time

import numpy as np
import torch

from common import ROOT, GpuMemSampler, case_dir, case_gt, eth3d_dataset, is_done, load_config, save_case


class VGGTRunner:
    def __init__(self, cfg):
        from vggt.models.vggt import VGGT

        self.cfg = cfg
        self.model = VGGT()
        self.model.load_state_dict(torch.load(ROOT / cfg["checkpoint"], map_location="cpu"))
        self.model = self.model.cuda().eval()

    @torch.no_grad()
    def __call__(self, image_files):
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        torch.cuda.synchronize()
        t0 = time.time()  # end-to-end: image reading + preprocessing + inference (model already loaded)
        images = load_and_preprocess_images(image_files, mode=self.cfg["preprocess_mode"]).cuda()
        torch.cuda.synchronize()
        self.read_s = time.time() - t0  # the official loader decodes the 24 MP JPEGs one by one (PIL)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = self.model(images)
        ext, ixt = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        depth = pred["depth"][0, ..., 0].float().cpu().numpy()
        return depth, ext[0].float().cpu().numpy(), ixt[0].float().cpu().numpy(), elapsed


class DA3Runner:
    def __init__(self, cfg):
        from depth_anything_3.api import DepthAnything3

        self.cfg = cfg
        self.model = DepthAnything3.from_pretrained(str(ROOT / cfg["checkpoint"])).cuda()

    def __call__(self, image_files):
        torch.cuda.synchronize()
        t0 = time.time()
        pred = self.model.inference(image_files, ref_view_strategy=self.cfg["ref_view_strategy"],
                                    process_res=self.cfg["process_res"])
        torch.cuda.synchronize()
        elapsed = time.time() - t0  # end-to-end: DA3 reads and preprocesses the images itself
        return pred.depth, pred.extrinsics, pred.intrinsics, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--method", choices=["vggt", "da3"], required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_dir = ROOT / "runs" / cfg["run_name"]
    cases = json.load(open(ROOT / "manifests" / f"{cfg['run_name']}.json"))["cases"]
    ds = eth3d_dataset()
    runner = (VGGTRunner if args.method == "vggt" else DA3Runner)(cfg[args.method])
    runner(case_gt(ds, cases[0])["image_files"])  # warm-up (CUDA kernels, cuDNN autotune), not timed

    for case in cases:
        out = case_dir(run_dir, args.method, case)
        if is_done(out):
            continue
        gt = case_gt(ds, case)
        torch.cuda.empty_cache()  # cached blocks of an earlier, larger case would inflate the NVML peak
        torch.cuda.reset_peak_memory_stats()
        with GpuMemSampler() as mem:
            depth, ext, ixt, elapsed = runner(gt["image_files"])
        info = {
            "method": args.method,
            "case": {k: case.get(k, "random") for k in ("scene", "n_views", "sampling", "seed", "images")},
            "registered": [True] * len(gt["image_files"]),
            "time_s": {"end_to_end": elapsed},
            "time_breakdown_s": ({"image_reading": runner.read_s, "inference": elapsed - runner.read_s}
                                 if hasattr(runner, "read_s") else None),
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
            "peak_gpu_gb": mem.peak_gb,  # NVML, process level (same measure as COLMAP)
            "depth_hw": list(depth.shape[1:]),
        }
        save_case(out, depth, ext, ixt, gt, info)
        print(f"{args.method} {case['scene']} {case['n_views']} views: {elapsed:.1f}s, "
              f"{info['peak_mem_gb']:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
