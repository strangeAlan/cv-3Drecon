"""Download one archive of the DA3-BENCH dataset (default: eth3d.zip) from hf-mirror,
verify its sha256 against the LFS oid, and unzip it.

Direct connection only (never the 10808 proxy). Segmented parallel download with
per-segment resume, since a single stream from hf-mirror runs at ~4 MB/s.

Usage: python scripts/download_da3bench.py [--file eth3d.zip] [--dest /home/hsy/datasets/da3-bench]
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "depth-anything/DA3-BENCH"
MIRROR = "https://hf-mirror.com"
ENV = {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}


def curl(args, **kw):
    return subprocess.run(["curl", "-sS", "--noproxy", "*"] + args, env=ENV, **kw)


def remote_meta(name):
    for _ in range(10):
        r = curl(["-m", "60", f"{MIRROR}/api/datasets/{REPO}/tree/main"], capture_output=True, text=True)
        try:
            for f in json.loads(r.stdout):
                if f["path"] == name:
                    return f["size"], f["lfs"]["oid"]
        except (json.JSONDecodeError, TypeError):
            pass
        time.sleep(10)
    raise RuntimeError("cannot read file listing from mirror")


def fetch_segment(url, part, start, end):
    want = end - start + 1
    for _ in range(30):
        have = part.stat().st_size if part.exists() else 0
        if have == want:
            return True
        if have > want:
            part.unlink()
            have = 0
        with open(part, "ab") as fh:
            curl(["-L", "-m", "3600", "-r", f"{start + have}-{end}", url], stdout=fh)
        time.sleep(3)
    return part.exists() and part.stat().st_size == want


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="eth3d.zip")
    ap.add_argument("--dest", default="/home/hsy/datasets/da3-bench")
    ap.add_argument("--segments", type=int, default=8)
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / args.file
    size, oid = remote_meta(args.file)
    print(f"{args.file}: {size / 1e9:.2f} GB, sha256 {oid}", flush=True)

    if not (out.exists() and out.stat().st_size == size):
        url = f"{MIRROR}/datasets/{REPO}/resolve/main/{args.file}"
        step = -(-size // args.segments)
        ranges = [(i, i * step, min(size, (i + 1) * step) - 1) for i in range(args.segments)]
        parts = [dest / f"{args.file}.part{i}" for i, _, _ in ranges]
        t0 = time.time()
        with ThreadPoolExecutor(args.segments) as ex:
            ok = list(ex.map(lambda r: fetch_segment(url, parts[r[0]], r[1], r[2]), ranges))
        if not all(ok):
            raise SystemExit(f"segments failed: {[i for i, o in enumerate(ok) if not o]} (rerun to resume)")
        with open(out, "wb") as fh:
            for p in parts:
                with open(p, "rb") as src:
                    while chunk := src.read(1 << 24):
                        fh.write(chunk)
        for p in parts:
            p.unlink()
        print(f"downloaded in {(time.time() - t0) / 60:.1f} min", flush=True)

    h = hashlib.sha256()
    with open(out, "rb") as fh:
        while chunk := fh.read(1 << 24):
            h.update(chunk)
    if h.hexdigest() != oid:
        raise SystemExit(f"sha256 mismatch: {h.hexdigest()}")
    print("sha256 ok", flush=True)

    with zipfile.ZipFile(out) as z:
        z.extractall(dest)
    print(f"extracted to {dest}", flush=True)


if __name__ == "__main__":
    main()
