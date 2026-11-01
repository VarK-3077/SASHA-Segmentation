#!/bin/bash
# Whole low-level separability study on one GPU box, detached: unit labels, screen tokens at
# levels 6, 5, 4, 3 (cheapest first), then for each level two heads (context on / off) x two
# seeds, two trainings at a time, each followed by its test evaluation, then the report.
# Log: /data2/venkatavks/lowsep/chain.log. Usage: bash experiments/lowsep/chain.sh cuda:0
cd ~/SASHA-Segmentation
export PYTHONPATH=. TORCH_HOME=/data2/venkatavks/torch_cache HF_HOME=/data2/venkatavks/hf_cache OMP_NUM_THREADS=2
PY=~/envs/sasha-seg/bin/python
DEV=${1:-cuda:0}
R=/data2/venkatavks/lowsep
set -e
mkdir -p $R
date; $PY -m experiments.lowsep.labels 10
for L in 6 5 4 3; do date; $PY -m experiments.lowsep.extract --level $L --device $DEV --workers 10; done
run() {
  tag=L$1_$([ $2 = 1 ] && echo ctx || echo noctx)_s$3
  [ -f $R/runs/$tag/test.json ] && return 0
  $PY -m experiments.lowsep.train --level $1 --ctx $2 --seed $3 --device $DEV --workers 2 > $R/train_$tag.log 2>&1 \
    && $PY -m experiments.lowsep.evaluate --tag $tag --device $DEV >> $R/train_$tag.log 2>&1
}
for L in 6 5 4 3; do
  for s in 0 1; do
    run $L 1 $s & run $L 0 $s & wait
  done
done
date; $PY -m experiments.lowsep.report > /dev/null
date; echo "lowsep chain done"
