#!/bin/bash
# After the 1028 GPU queues: evaluate 1028-allviews (CPU; runs alongside the match statistics).
cd /home/hsy/cv-3Drecon
while ps -eo args | grep -q "^/bin/bash scripts/launch_1028.sh"; do sleep 60; done
mkdir -p results/1028-allviews
.venv/bin/python src/evaluate.py configs/1028-allviews.yaml --methods colmap vggt da3 --workers 24 --no-da3-protocol \
    > runs/1028-allviews/eval.log 2>&1
.venv/bin/python scripts/summarize.py 1028-allviews > results/1028-allviews/summary.txt 2>&1
echo "1028-allviews evaluated $(date -u)"
