#!/bin/bash
# Coarse design with a level-3 context head trained under unit dropout (unread units zeroed):
# train the heads (same settings as the picked full-screen heads, plus --unit_drop 0.8), score the
# global view's screened set with them, and rerun the coarse baselines on those outputs.
# Usage (isp-a100): bash experiments/rlscreen/udrop_chain.sh
cd ~/SASHA-Segmentation
export PYTHONPATH=. OMP_NUM_THREADS=4
PY=~/envs/sasha-seg/bin/python
L=/data2/venkatavks/lowsep/runs
common="--level 3 --ctx 1 --batch 4 --eval_every 150 --workers 2 --unit_drop 0.8"
LOWSEP_BACKBONE=kaiko_vitl14 $PY -m experiments.lowsep.train $common --d 128 --drop 0.3 --wd 0.01 --steps 1500 --name _d128_udrop > $L/udrop_vitl.log 2>&1 &
LOWSEP_BACKBONE=kaiko_vitl14 $PY -m experiments.lowsep.train $common --d 128 --drop 0.3 --wd 0.01 --steps 3000 --name _d128_udrop3k > $L/udrop_vitl3k.log 2>&1 &
LOWSEP_BACKBONE=kaiko_vits16 $PY -m experiments.lowsep.train $common --d 64 --drop 0.5 --wd 0.05 --steps 1500 --name _d64_udrop > $L/udrop_vits.log 2>&1 &
wait
for spec in kaiko_vitl14:L3_ctx_d128_udrop_s0 kaiko_vitl14:L3_ctx_d128_udrop3k_s0 kaiko_vits16:L3_kaiko_vits16_ctx_d64_udrop_s0; do
  bb=${spec%%:*}; head=${spec#*:}
  for g in 6 5; do
    LOWSEP_BACKBONE=$bb $PY -m experiments.rlscreen.p3drop --head $head --g_level $g
    RLSCREEN_P3=$head $PY -m experiments.rlscreen.evaluate --design coarse --g_level $g --backbone $bb --taus 0.5 0.3 > /dev/null
  done
done
date; echo "udrop chain done"
