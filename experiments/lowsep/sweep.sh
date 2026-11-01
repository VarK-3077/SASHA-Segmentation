#!/bin/bash
# Regularisation sweep after the first pass showed the 256-wide head overfitting: for every
# level, narrower context heads with more dropout / weight decay and fewer steps, plus a narrow
# no-context head, two seeds each, three trainings at a time, each evaluated on test. Waits for
# the given PID (the timing step) so the timing runs alone. Log: /data2/venkatavks/lowsep/sweep.log
cd ~/SASHA-Segmentation
export PYTHONPATH=. TORCH_HOME=/data2/venkatavks/torch_cache OMP_NUM_THREADS=2
PY=~/envs/sasha-seg/bin/python
R=/data2/venkatavks/lowsep
DEV=${2:-cuda:0}
while kill -0 $1 2>/dev/null; do sleep 30; done
run() {  # level ctx d drop wd name seed
  tag=L$1_$([ $2 = 1 ] && echo ctx || echo noctx)$6_s$7
  [ -f $R/runs/$tag/test.json ] && return 0
  $PY -m experiments.lowsep.train --level $1 --ctx $2 --d $3 --drop $4 --wd $5 --name $6 --seed $7 --steps 1500 --eval_every 150 \
      --device $DEV --workers 2 > $R/train_$tag.log 2>&1 \
    && $PY -m experiments.lowsep.evaluate --tag $tag --device $DEV >> $R/train_$tag.log 2>&1
}
for L in 6 5 4 3; do
  for s in 0 1; do
    run $L 1 128 0.3 0.01 _d128 $s & run $L 1 64 0.5 0.05 _d64 $s & run $L 0 64 0.5 0.05 _d64 $s & wait
  done
done
date; $PY -m experiments.lowsep.report > /dev/null
date; echo "sweep done"
