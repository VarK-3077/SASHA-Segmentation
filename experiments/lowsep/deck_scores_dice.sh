#!/bin/bash
# Same as deck_scores.sh but with the screen heads picked on val by ideal-segmenter Dice at the
# 20 % budget instead of by AUC; outputs under /data2/venkatavks/lowsep/deck_dice.
# Log: /data2/venkatavks/lowsep/deck_scores_dice.log
cd ~/SASHA-Segmentation
export PYTHONPATH=. TORCH_HOME=/data2/venkatavks/torch_cache OMP_NUM_THREADS=4 CAMELYON16_MIRROR=/data2/venkatavks/c16_extra
PY=~/envs/sasha-seg/bin/python
D=${OUT:-/data2/venkatavks/lowsep/deck_dice}
set -e
mkdir -p $D
[ -f $D/screen_ideal.json ] || $PY -m experiments.lowsep.screen_ideal --out_root $D --select dice $EXTRA
export CUDA_VISIBLE_DEVICES=
for d in $D/ideal_L3 $D/ideal_L4 $D/ideal_L5 $D/ideal_L6; do
  [ -f $d/clean.json ] || $PY -m experiments.pfa_scannet.clean_sweep --maps $d --thrs 0.5 --seeds 0.9 --areas 0 --workers 2 --out $d/clean.json > $d/clean.log 2>&1 &
done
wait
for d in $D/ideal_L3 $D/ideal_L4 $D/ideal_L5 $D/ideal_L6; do
  [ -f $d/small_probe.json ] || $PY -m experiments.pfa_scannet.small_probe --detector_maps $d --decoder_maps $d --out $d/small_probe.json > $d/small_probe.log 2>&1 &
done
wait
date; echo "deck scores dice done"
