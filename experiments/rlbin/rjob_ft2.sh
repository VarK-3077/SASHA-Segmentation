#!/bin/bash
# Generic clone-then-fine-tune chain: behaviour-clone the oracle routing for each state
# variant (if the checkpoint is missing), then, once the GPU has room, fine-tune each with PPO
# at lr 1e-5 / entropy 0.001 and evaluate. Usage: rjob_ft2.sh <device> <nvidia-smi index>
# <level 3|2> <mode neg|lr> <gate MiB>. Tags: l<level><mode>_ft_<state> (mode omitted for neg).
cd ~/SASHA-Segmentation
export PYTHONPATH=. OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/envs/sasha-seg/bin/python
DEV=$1; IDX=$2; LVL=$3; MODE=$4; GATE=${5:-14000}
R=/data2/venkatavks/rlbin
if [ "$MODE" = neg ]; then PFX=l$LVL; else PFX=l$LVL$MODE; fi
if [ "$LVL" = 3 ]; then EPS=16; UPS=600; BCEP=6; else EPS=8; UPS=200; BCEP=20; fi
for st in oracle pred none; do
  [ -f $R/ckpt/bc_${PFX}_$st.pt ] || $PY -m experiments.rlbin.bc --device $DEV --level $LVL --mode $MODE --state $st --epochs $BCEP --tag bc_${PFX}_$st > $R/bc_${PFX}_$st.log 2>&1
done
date; echo "bc done for $PFX"
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $IDX)" -lt $GATE ]; do sleep 120; done
date; echo "gpu $IDX has room, launching fine-tunes for $PFX"
for st in oracle pred none; do
  tag=${PFX}_ft_$st
  setsid nohup bash -c "$PY -m experiments.rlbin.train --device $DEV --level $LVL --mode $MODE --lr 1e-5 --ent 0.001 --episodes $EPS --updates $UPS --balanced --state $st --init $R/ckpt/bc_${PFX}_$st.pt --tag $tag \
    && $PY -m experiments.rlbin.evaluate --device $DEV --tag $tag --criterion balanced \
    && $PY -m experiments.rlbin.evaluate --device $DEV --tag $tag --criterion best" > $R/run_$tag.log 2>&1 < /dev/null &
done
wait
date; echo "fine-tunes done for $PFX"
