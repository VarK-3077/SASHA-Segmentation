#!/bin/bash
# Final step: waits for the given PID (the sweep), re-evaluates every finished run with the
# current evaluate.py (adds the budget frontier to runs scored before it existed), then rebuilds
# the report. Log: /data2/venkatavks/lowsep/reeval.log
cd ~/SASHA-Segmentation
export PYTHONPATH=. OMP_NUM_THREADS=2
PY=~/envs/sasha-seg/bin/python
R=/data2/venkatavks/lowsep
while kill -0 $1 2>/dev/null; do sleep 30; done
for d in $R/runs/*/; do
  t=$(basename $d)
  [ -f $d/best.pt ] && $PY -m experiments.lowsep.evaluate --tag $t --device cuda:0 2>&1 | grep -v Warning
done
date; $PY -m experiments.lowsep.report > /dev/null
date; echo "reeval done"
