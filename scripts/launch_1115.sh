#!/bin/bash
# 11/15: DA3-BA ablations on the exploration set (GPU0) and clean runtime of DA3-BA / VGGT-BA (GPU1, alone),
# then evaluation of the ablations.
cd /home/hsy/cv-3Drecon
PY=.venv/bin/python
( for a in triangulate simpleradial fixintr onestage; do
    mkdir -p runs/1115-abl-$a
    CUDA_VISIBLE_DEVICES=0 $PY src/run_da3ba.py configs/1115-abl-$a.yaml > runs/1115-abl-$a/sfm.log 2>&1
  done ) &
( mkdir -p runs/1115-efficiency
  for c in 1115-efficiency 1115-efficiency-vggtba; do
    CUDA_VISIBLE_DEVICES=1 $PY src/run_da3ba.py configs/$c.yaml > runs/1115-efficiency/sfm_$c.log 2>&1
  done ) &
wait
for a in triangulate simpleradial fixintr onestage; do
  $PY src/evaluate.py configs/1115-abl-$a.yaml --methods da3ba --no-da3-protocol --workers 24 > runs/1115-abl-$a/eval.log 2>&1
done
echo "$(date -u '+%m-%d %H:%M') 1115 done"
