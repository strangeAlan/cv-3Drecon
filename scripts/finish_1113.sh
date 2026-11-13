#!/bin/bash
# Unattended end of the 11/13 runs: wait for colmap_radial and the DA3-BA/VGGT-BA sparse stage, evaluate,
# summarize to results/1113-summary.md; then run the DA3-BA dense stage on all cases, evaluate, summarize again.
cd /home/hsy/cv-3Drecon
PY=.venv/bin/python
EV="$PY src/evaluate.py --no-da3-protocol --workers 24"
waitfor() { while pgrep -f "$1" > /dev/null; do sleep 60; done; }
log() { echo "$(date -u '+%m-%d %H:%M') $*"; }

waitfor "evaluate.py configs/1113-da3ba-explore"
$EV configs/1113-da3ba-explore.yaml --methods da3ba da3ba_ds vggtba vggtba_ds hybrid_da3ba hybrid_da3ba_fill \
    > runs/1113-da3ba-explore/eval.log 2>&1
$PY scripts/summarize_1113.py; log "exploration evaluated"

waitfor "launch_1113_ba.sh"
waitfor "launch_1113_colmap_radial.sh"
for r in main allviews; do
  $EV configs/1113-$r.yaml --methods colmap_radial da3ba da3ba_ds vggtba vggtba_ds > runs/1113-$r/eval_sparse.log 2>&1
done
$PY scripts/summarize_1113.py; log "stage 1 (sparse methods + colmap_radial) summarized"

for i in 0 1; do
  ( for r in main allviews; do
      CUDA_VISIBLE_DEVICES=$i $PY src/run_da3ba.py configs/1113-$r.yaml --shard $i/2 --dense > runs/1113-$r/dense_shard$i.log 2>&1
    done ) &
done
wait
for r in main allviews; do
  $EV configs/1113-$r.yaml --methods colmap_radial da3ba da3ba_ds vggtba vggtba_ds hybrid_da3ba hybrid_da3ba_fill \
      > runs/1113-$r/eval_all.log 2>&1
done
$PY scripts/summarize_1113.py; log "stage 2 (dense stage) summarized, all done"
