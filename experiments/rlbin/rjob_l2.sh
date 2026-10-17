#!/bin/bash
# Level-2 chain, detached: waits for the Kaiko extraction to finish, builds the level-2 cache
# and low-res head, runs the binning agent at level 2 with plain and oracle-bucket state, then
# evaluates both and draws the level-2 frontier. Log: /data2/venkatavks/rlbin/run_l2_chain.log
cd ~/SASHA-Segmentation
export PYTHONPATH=. OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/envs/sasha-seg/bin/python
DEV=${1:-cuda:1}
R=/data2/venkatavks/rlbin
set -e
until grep -q "l2 extraction done" $R/extract_l2.log; do sleep 60; done
date; echo "extraction done, building cache_l2"
[ -f $R/cache_l2/test_001.npz ] || $PY -m experiments.rlbin.cache_l2 $DEV
date; echo "cache_l2 done, launching l2 runs"
for cfg in "l2_none --state none" "l2_oracle --state oracle"; do
  set -- $cfg; tag=$1; shift
  setsid nohup bash -c "$PY -m experiments.rlbin.train --device $DEV --level 2 --lr 1e-4 --ent 0.02 --episodes 8 --updates 400 --balanced --tag $tag $* \
    && $PY -m experiments.rlbin.evaluate --device $DEV --tag $tag --criterion balanced \
    && $PY -m experiments.rlbin.evaluate --device $DEV --tag $tag --criterion best" > $R/run_$tag.log 2>&1 < /dev/null &
done
wait
date; echo "l2 runs done, frontier"
$PY -m experiments.rlbin.frontier --level 2
date; echo "l2 chain done"
