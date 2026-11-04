#!/bin/bash
# Everything the RL screening runs read, on the box that holds /data2/venkatavks/lowsep:
# unit tables for the ViT-L screen, offline baselines, then the faster screen (Kaiko ViT-S/16):
# its tokens at levels 6, 5 and 3, its supervised screen heads (the configurations that won the
# lowsep sweep, seed 0), and its unit tables. Log: /data2/venkatavks/rlscreen/prep_chain.log
# Usage: bash experiments/rlscreen/prep_chain.sh cuda:0
cd ~/SASHA-Segmentation
export PYTHONPATH=. TORCH_HOME=/data2/venkatavks/torch_cache HF_HOME=/data2/venkatavks/hf_cache OMP_NUM_THREADS=2
PY=~/envs/sasha-seg/bin/python
DEV=${1:-cuda:0}
set -e
mkdir -p /data2/venkatavks/rlscreen
date; LOWSEP_BACKBONE=kaiko_vitl14 $PY -m experiments.rlscreen.prep --device $DEV
for spec in "coarse 6" "coarse 5" "full 6"; do
  set -- $spec
  date; $PY -m experiments.rlscreen.evaluate --design $1 --g_level $2 --backbone kaiko_vitl14 --taus 0.5 0.3 --device $DEV
done
export LOWSEP_BACKBONE=kaiko_vits16
for L in 6 5 3; do date; $PY -m experiments.lowsep.extract --level $L --device $DEV --workers 10; done
for L in 6 5 3; do
  for cfg in "1 128 0.3 0.01 _d128" "1 64 0.5 0.05 _d64" "0 64 0.5 0.05 _d64"; do
    set -- $cfg
    $PY -m experiments.lowsep.train --level $L --ctx $1 --d $2 --drop $3 --wd $4 --name $5 --seed 0 --steps 1500 --eval_every 150 --device $DEV --workers 2
  done
done
date; $PY -m experiments.rlscreen.prep --device $DEV
for spec in "coarse 6" "full 6"; do
  set -- $spec
  date; $PY -m experiments.rlscreen.evaluate --design $1 --g_level $2 --backbone kaiko_vits16 --taus 0.5 0.3 --device $DEV
done
date; echo "prep chain done"
