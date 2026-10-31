#!/bin/bash
# After the 1028 GPU queues: matching statistics for the 1027-main cases on both GPUs.
cd /home/hsy/cv-3Drecon
while ps -eo args | grep -q "^/bin/bash scripts/launch_1028.sh"; do sleep 60; done
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/match_stats.py configs/1027-main.yaml --shard 0/2 > runs/1027-main/match_stats_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/match_stats.py configs/1027-main.yaml --shard 1/2 > runs/1027-main/match_stats_1.log 2>&1 &
wait
echo "match stats finished $(date -u)"
