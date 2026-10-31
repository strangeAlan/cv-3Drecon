"""Write the fixed view-subset manifest for a config: manifests/<run_name>.json"""
import argparse
import json

from common import ROOT, load_config, make_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cases = make_manifest(cfg)
    out = ROOT / "manifests" / f"{cfg['run_name']}.json"
    with open(out, "w") as f:
        json.dump({"config": args.config, "cases": cases}, f, indent=1)
    print(f"{len(cases)} cases -> {out}")
    for c in cases:
        print(f"  {c['scene']:<14} {str(c['n_views']):>4} views  seed {c['seed']}")


if __name__ == "__main__":
    main()
