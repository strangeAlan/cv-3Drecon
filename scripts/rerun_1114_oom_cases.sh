#!/bin/bash
# Re-run the 6 COLMAP-RADIAL cases that failed with CUDA OOM on GPU1 at 11/14 03:54-03:56 (GPU shared with a
# graph-mining job). The failed outputs were moved to runs/_oom_failed_20261114/; run_colmap.py skips all
# finished cases, so each shard-1 call redoes exactly the moved cases. finish_1113.sh (PID given) was paused
# with SIGSTOP and is resumed when both re-runs end, so its later evaluations see the new results.
cd /home/hsy/cv-3Drecon
FINISH_PID=$1
CUDA_VISIBLE_DEVICES=0 .venv/bin/python src/run_colmap.py configs/1113-main.yaml --shard 1/2 \
    > runs/1113-main/colmap_radial_rerun_1114.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 .venv/bin/python src/run_colmap.py configs/1113-allviews.yaml --shard 1/2 \
    > runs/1113-allviews/colmap_radial_rerun_1114.log 2>&1 &
P1=$!
wait $P0 $P1
echo "$(date -u '+%m-%d %H:%M') re-run done, resuming finish_1113.sh ($FINISH_PID)"
kill -CONT "$FINISH_PID"
