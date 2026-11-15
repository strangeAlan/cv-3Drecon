#!/bin/bash
# After the 1115 efficiency runs on GPU1: clean DA3-BA / VGGT-BA runtime with all views (GPU1 alone).
cd /home/hsy/cv-3Drecon
while pgrep -f "run_da3ba.py configs/1115-efficiency" > /dev/null; do sleep 30; done
mkdir -p runs/1115-allviews-time
for c in 1115-allviews-time 1115-allviews-time-vggtba; do
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python src/run_da3ba.py configs/$c.yaml > runs/1115-allviews-time/sfm_$c.log 2>&1
done
echo "$(date -u '+%m-%d %H:%M') alltime done"
