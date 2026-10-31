"""Precompute the whole-scene GT reference clouds used by evaluate.py (one process per scene)."""
import sys
import time
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def build(scene):
    import evaluate as ev
    from common import eth3d_dataset

    t0 = time.time()
    pts = ev.gt_reference_cloud(eth3d_dataset(), scene)
    return scene, len(pts), time.time() - t0


if __name__ == "__main__":
    from common import eth3d_dataset

    scenes = sys.argv[1:] or [s for s in eth3d_dataset().SCENES]
    with get_context("spawn").Pool(len(scenes)) as pool:
        for scene, n, dt in pool.imap_unordered(build, scenes):
            print(f"{scene}: {n / 1e6:.1f}M points, {dt:.0f}s", flush=True)
