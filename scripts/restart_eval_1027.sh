#!/bin/bash
# Restart evaluation after the open3d 0.19 pin: stop the running (bogus) evaluation,
# delete fused clouds produced with open3d 0.20, then evaluate the repro run in the
# background. COLMAP / DA3 inference jobs are left running.
set -u
cd /home/hsy/cv-3Drecon

for pid in $(ps -eo pid,args | grep "[s]rc/evaluate.py" | awk '{print $1}'); do
    kill "$pid" && echo "stopped evaluate.py pid $pid"
done
sleep 1

find runs -path "*exports/fuse" -type d -prune -exec rm -rf {} +
rm -f runs/1027-repro/eval_vggt.log
rm -rf results/1027-repro
echo "remaining fused clouds: $(find runs -name pcd.ply | wc -l)"

.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); import common, open3d; print('open3d', open3d.__version__)"

nohup .venv/bin/python src/evaluate.py configs/1027-repro.yaml --methods vggt da3 \
    > runs/1027-repro/eval.log 2>&1 &
echo "repro evaluation started (pid $!), log: runs/1027-repro/eval.log"
