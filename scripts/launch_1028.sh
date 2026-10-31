#!/bin/bash
# 1028 runs: all-views (11 scenes) and efficiency (3 scenes x 5/10/20) with end-to-end time and NVML memory.
cd /home/hsy/cv-3Drecon
gpu0() {
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python src/run_colmap.py configs/1028-allviews.yaml --shard 0/2 > runs/1028-allviews/colmap_shard0.log 2>&1
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python src/run_colmap.py configs/1028-efficiency.yaml > runs/1028-efficiency/colmap.log 2>&1
}
gpu1() {
  for m in vggt da3; do
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python src/run_feedforward.py configs/1028-allviews.yaml --method $m > runs/1028-allviews/infer_$m.log 2>&1
  done
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python src/run_colmap.py configs/1028-allviews.yaml --shard 1/2 > runs/1028-allviews/colmap_shard1.log 2>&1
}
gpu0 & gpu1 & wait
echo "1028 runs finished $(date -u)"
