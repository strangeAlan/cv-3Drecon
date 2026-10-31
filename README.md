# cv-3Drecon

Course project (Image Processing & Computer Vision): classical multi-view 3D
reconstruction (COLMAP) vs. feed-forward 3D foundation models (VGGT, Depth Anything 3)
on ETH3D, as a function of the number of input views.

## Layout

| Path | Content |
|---|---|
| `configs/` | one YAML per run (scenes, view counts, seeds, method settings) |
| `manifests/` | fixed view subsets per run, shared by all methods |
| `src/` | `make_manifest.py`, `run_colmap.py`, `run_feedforward.py`, `evaluate.py`, `common.py` |
| `runs/<run>/` | raw outputs (not in git) |
| `results/<run>/` | compact metrics and figures |
| `notes/LOG.md` | experiment log |
| `report/` | proposal and final report |
| `third_party/vggt` | official VGGT code, commit `a288dd0` (sha1-verified via GitHub API) |
| `third_party/Depth-Anything-3` | symlink to `/home/hsy/references/Depth-Anything-3` (code + ETH3D benchmark) |
| `checkpoints/` | symlinks to local VGGT-1B and DA3NESTED-GIANT-LARGE weights |
| `data/eth3d` | symlink to `/home/hsy/datasets/da3-bench/eth3d` (DA3-BENCH preprocessed ETH3D) |

## Environment

Python 3.11.17 (standalone build in `env/python3.11`, from the npmmirror mirror of
python-build-standalone) with a venv in `.venv`:

```bash
env/python3.11/bin/python3 -m venv .venv
.venv/bin/pip install -r env/requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

All downloads use direct domestic mirrors (aliyun PyPI, hf-mirror), not the proxy.

## Running

```bash
PY=.venv/bin/python
$PY src/make_manifest.py configs/1027-prelim.yaml
$PY src/run_colmap.py configs/1027-prelim.yaml
$PY src/run_feedforward.py configs/1027-prelim.yaml --method vggt
$PY src/run_feedforward.py configs/1027-prelim.yaml --method da3
$PY src/evaluate.py configs/1027-prelim.yaml
```

Data: `python3 scripts/download_da3bench.py` (eth3d.zip, 15.1 GB, sha256-checked).

## Evaluation

Main metrics (`src/evaluate.py`, details and reasons in `notes/LOG.md`, 10/27):
- cameras: relative pose AUC@5/30 over all view pairs, pairs with an unregistered image count as failures;
- geometry: point-cloud precision / recall / F-score at 2, 5, 10 cm against a GT cloud built from the
  back-projected GT depth of all views (the DA3 GT mesh is incomplete); per-pixel Sim(3) alignment plus
  similarity ICP; COLMAP scored on its stereo-fusion cloud, VGGT/DA3 on back-projected depth maps.
The unmodified DA3 benchmark protocol (TSDF, F-score at 0.25 m) is kept only as a reproduction check
(`1027-repro`).

Runs used in the report: `1027-repro`, `1027-main` (180 subsets: 11 scenes x {5,10,20} views x
{neighbor, random} x 3 seeds), `1028-allviews`, `1028-efficiency`, `1027-hybridtest` (exploratory).

## Report

```bash
PY=.venv/bin/python
$PY scripts/summarize.py 1027-main            # results/<run>/summary.{csv,txt}
$PY scripts/nested_views.py 1027-main         # same 5 views inside 10/20-view reconstructions
$PY scripts/make_figures.py 1027-main         # report/figures/fig_fscore_views.pdf, fig_prec_recall.pdf
$PY scripts/make_qualitative.py 1027-main courtyard:nb_v10_s0 office:nb_v10_s0
$PY scripts/make_tables.py                    # report/tables/*.tex
bash report/build.sh report                   # report/build/report.pdf (TinyTeX)
```
