#!/bin/bash
# DA3-BA and VGGT-BA sparse stage on 1113-main and 1113-allviews (GPU only for SIFT; shares the GPUs with
# the colmap_radial run).
cd /home/hsy/cv-3Drecon
for i in 0 1; do
  ( for c in 1113-main 1113-allviews 1113-main-vggtba 1113-allviews-vggtba; do
      run=$(grep '^run_name' configs/$c.yaml | awk '{print $2}')
      CUDA_VISIBLE_DEVICES=$i .venv/bin/python src/run_da3ba.py configs/$c.yaml --shard $i/2 > runs/$run/sfm_${c}_shard$i.log 2>&1
    done ) &
done
wait
echo "BA sparse stage done $(date -u)"
