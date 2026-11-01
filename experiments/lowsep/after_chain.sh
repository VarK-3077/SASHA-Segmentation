#!/bin/bash
# Runs after the main lowsep chain: waits for its PID to exit, then times the screen alone on
# the test slides (levels 2-6) and rebuilds the report with the timing table.
# Usage: bash experiments/lowsep/after_chain.sh <chain pid> cuda:0. Log: /data2/venkatavks/lowsep/timing.log
cd ~/SASHA-Segmentation
export PYTHONPATH=. TORCH_HOME=/data2/venkatavks/torch_cache OMP_NUM_THREADS=2
PY=~/envs/sasha-seg/bin/python
while kill -0 $1 2>/dev/null; do sleep 30; done
date; $PY -m experiments.lowsep.time_screen --device ${2:-cuda:0}
date; $PY -m experiments.lowsep.report > /dev/null
date; echo "timing done"
