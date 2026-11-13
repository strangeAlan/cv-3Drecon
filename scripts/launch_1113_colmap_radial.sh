#!/bin/bash
# COLMAP baseline with the RADIAL camera model: 1113-main (181 cases) then 1113-allviews (11), sharded over both GPUs.
cd /home/hsy/cv-3Drecon
mkdir -p runs/1113-main runs/1113-allviews
for i in 0 1; do
  ( CUDA_VISIBLE_DEVICES=$i .venv/bin/python src/run_colmap.py configs/1113-main.yaml --shard $i/2 > runs/1113-main/colmap_radial_shard$i.log 2>&1
    CUDA_VISIBLE_DEVICES=$i .venv/bin/python src/run_colmap.py configs/1113-allviews.yaml --shard $i/2 > runs/1113-allviews/colmap_radial_shard$i.log 2>&1 ) &
done
wait
echo "colmap_radial done $(date -u)"
