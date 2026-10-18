#!/bin/bash
# Error bars for the clone-only level-2 policies: behaviour-clone with seeds 1 and 2 for the
# label-free state variants, then score each after a single gentle PPO update (lr 1e-5), which
# is how the seed-0 rows `l2_bc_*` were effectively picked. Usage: rjob_bcseeds.sh cuda:1
cd ~/SASHA-Segmentation
export PYTHONPATH=. OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/envs/sasha-seg/bin/python
DEV=${1:-cuda:1}
R=/data2/venkatavks/rlbin
for seed in 1 2; do
  for st in none pred; do
    tag=l2_bcs${seed}_$st
    $PY -m experiments.rlbin.bc --device $DEV --level 2 --state $st --epochs 20 --seed $seed --tag bc_$tag > $R/bc_$tag.log 2>&1
    $PY -m experiments.rlbin.train --device $DEV --level 2 --lr 1e-5 --ent 0.001 --episodes 8 --updates 1 --eval_every 1 --balanced --state $st --seed $seed --init $R/ckpt/bc_$tag.pt --tag $tag > $R/run_$tag.log 2>&1
    $PY -m experiments.rlbin.evaluate --device $DEV --tag $tag --criterion balanced >> $R/run_$tag.log 2>&1
  done
done
date; echo "bc seeds done"
