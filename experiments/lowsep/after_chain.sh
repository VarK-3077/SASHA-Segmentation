#!/bin/bash
# Runs after the main lowsep chain: waits for every given PID to exit (the chain and any other
# GPU job on the box), then times the screen alone on the test slides (levels 2-6) and rebuilds
# the report with the timing table.
# Usage: bash experiments/lowsep/after_chain.sh <pid> [<pid> ...]. Log: /data2/venkatavks/lowsep/timing.log
cd ~/SASHA-Segmentation
export PYTHONPATH=. TORCH_HOME=/data2/venkatavks/torch_cache OMP_NUM_THREADS=2
PY=~/envs/sasha-seg/bin/python
for p in "$@"; do while kill -0 $p 2>/dev/null; do sleep 30; done; done
until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 30; done
date; $PY -m experiments.lowsep.time_screen --device cuda:0
date; $PY -m experiments.lowsep.report > /dev/null
date; echo "timing done"
