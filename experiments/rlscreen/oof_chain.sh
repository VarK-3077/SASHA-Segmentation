#!/bin/bash
# vivi GPU 0: out-of-fold copies of the heads the slide tables use (5 folds each, same settings as
# the picked heads), the out-of-fold tables for the train slides, then the sequential stop policy
# trained on them (and, as a control, on the full-data tables).
# Usage: bash experiments/rlscreen/oof_chain.sh   (waits for the token copy to finish)
cd /data1/venkatavks/SASHA-Segmentation
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 LOWSEP_ROOT=/var/tmp/venkatavks/lowsep RLSCREEN_ROOT=/var/tmp/venkatavks/rlscreen OMP_NUM_THREADS=4
PY=~/envs/sasha-seg/bin/python
L=$LOWSEP_ROOT/runs
R=$RLSCREEN_ROOT/runs
until grep -q copy-done /var/tmp/venkatavks/copy_tokens.log; do sleep 30; done
mkdir -p $L/oof_logs
for k in 0 1 2 3 4; do
  $PY -m experiments.lowsep.train --level 6 --ctx 0 --d 64 --drop 0.5 --wd 0.05 --steps 1500 --eval_every 150 --workers 2 --name _d64 --fold $k > $L/oof_logs/L6_f$k.log 2>&1 &
  $PY -m experiments.lowsep.train --level 3 --ctx 0 --d 256 --drop 0.1 --wd 0.0001 --steps 3000 --eval_every 300 --workers 2 --fold $k > $L/oof_logs/L3u_f$k.log 2>&1 &
  $PY -m experiments.lowsep.train --level 3 --ctx 1 --d 128 --drop 0.3 --wd 0.01 --steps 1500 --eval_every 150 --workers 2 --name _d128 --fold $k > $L/oof_logs/L3c_f$k.log 2>&1 &
  wait
done
$PY -m experiments.rlscreen.oof --heads L6_full L3_unit L3_full > $L/oof_logs/oof.log 2>&1
export CUDA_VISIBLE_DEVICES=
run() { tag=$1; shift; mkdir -p $R/$tag; "$@" --tag $tag > $R/$tag/log.txt 2>&1; }
RLSCREEN_OOF=1 run seq_oof_s0 $PY -m experiments.rlscreen.seqscreen --lam_min 0.1 --lam_max 30 --seed 0 &
RLSCREEN_OOF=1 run seq_oof_s1 $PY -m experiments.rlscreen.seqscreen --lam_min 0.1 --lam_max 30 --seed 1 &
RLSCREEN_OOF=1 run seq_oof_tz05 $PY -m experiments.rlscreen.seqscreen --lam_min 0.1 --lam_max 30 --tz 0.5 &
run seq_inf_wide $PY -m experiments.rlscreen.seqscreen --lam_min 0.1 --lam_max 30 &
wait
echo "oof chain done"
