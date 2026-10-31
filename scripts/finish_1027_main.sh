#!/bin/bash
# Wait until the 1027-main sweep (COLMAP shards + GPU1 queue) is done, then evaluate, summarize and plot.
cd /home/hsy/cv-3Drecon
busy() { ps -eo args | grep -E "^(/bin/bash scripts/launch_1027_main_gpu1.sh|\.venv/bin/python src/run_(colmap|feedforward)\.py configs/1027-main\.yaml)" > /dev/null; }
while busy; do sleep 60; done
echo "sweep finished $(date -u)"
.venv/bin/python src/evaluate.py configs/1027-main.yaml --methods colmap vggt da3 --workers 24 --no-da3-protocol \
    > runs/1027-main/eval.log 2>&1
.venv/bin/python scripts/summarize.py 1027-main > results/1027-main/summary.txt 2>&1
.venv/bin/python scripts/make_figures.py 1027-main >> results/1027-main/summary.txt 2>&1
echo "evaluation finished $(date -u)"
