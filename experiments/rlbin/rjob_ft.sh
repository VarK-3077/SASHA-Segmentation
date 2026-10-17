#!/bin/bash
# Fine-tune the behaviour-cloned level-3 policies with PPO at SASHA's conservative settings
# (lr 1e-5, entropy 0.001), one run per state variant, starting once the given GPU has room.
# Usage: rjob_ft.sh cuda:1 1   (device, nvidia-smi index). Log: run_l3_ft_<state>.log
cd ~/SASHA-Segmentation
export PYTHONPATH=. OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/envs/sasha-seg/bin/python
DEV=${1:-cuda:1}
IDX=${2:-1}
R=/data2/venkatavks/rlbin
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $IDX)" -lt 12000 ]; do sleep 120; done
date; echo "gpu $IDX has room, launching fine-tunes"
for st in oracle pred none; do
  tag=l3_ft_$st
  setsid nohup bash -c "$PY -m experiments.rlbin.train --device $DEV --lr 1e-5 --ent 0.001 --episodes 16 --balanced --state $st --init $R/ckpt/bc_l3_$st.pt --tag $tag \
    && $PY -m experiments.rlbin.evaluate --device $DEV --tag $tag --criterion balanced \
    && $PY -m experiments.rlbin.evaluate --device $DEV --tag $tag --criterion best" > $R/run_$tag.log 2>&1 < /dev/null &
done
wait
date; echo "fine-tunes done"
