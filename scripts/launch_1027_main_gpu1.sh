#!/bin/bash
# GPU1 queue for the 1027-main sweep: wait for the running VGGT inference, then DA3, then COLMAP shard 1/2.
cd /home/hsy/cv-3Drecon
while ps -p "$1" > /dev/null 2>&1; do sleep 10; done
CUDA_VISIBLE_DEVICES=1 .venv/bin/python src/run_feedforward.py configs/1027-main.yaml --method vggt >> runs/1027-main/infer_vggt.log 2>&1
CUDA_VISIBLE_DEVICES=1 .venv/bin/python src/run_feedforward.py configs/1027-main.yaml --method da3 > runs/1027-main/infer_da3.log 2>&1
CUDA_VISIBLE_DEVICES=1 .venv/bin/python src/run_colmap.py configs/1027-main.yaml --shard 1/2 > runs/1027-main/colmap_shard1.log 2>&1
