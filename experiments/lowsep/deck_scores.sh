#!/bin/bash
# Scores for the 2 Nov deck: ideal-segmenter maps for the L3-L6 screens (screen_ideal.py), then
# clean_sweep (per-slide / pooled / per-lesion Dice, FP components) and small_probe (found lesions
# per size bin) on those maps and on the two pipelined L3 cascades, with the token head's
# validation-chosen rules. Log: /data2/venkatavks/lowsep/deck_scores.log
cd ~/SASHA-Segmentation
export PYTHONPATH=. TORCH_HOME=/data2/venkatavks/torch_cache OMP_NUM_THREADS=4 CAMELYON16_MIRROR=/data2/venkatavks/c16_extra
PY=~/envs/sasha-seg/bin/python
D=/data2/venkatavks/lowsep/deck
C=/data2/venkatavks/lowsep/cascade
set -e
mkdir -p $D
[ -f $D/screen_ideal.json ] || $PY -m experiments.lowsep.screen_ideal --out_root $D
export CUDA_VISIBLE_DEVICES=
for d in $D/ideal_L3 $D/ideal_L4 $D/ideal_L5 $D/ideal_L6; do
  [ -f $d/clean.json ] || $PY -m experiments.pfa_scannet.clean_sweep --maps $d --thrs 0.5 --seeds 0.9 --areas 0 --workers 2 --out $d/clean.json > $d/clean.log 2>&1 &
done
wait
for d in $D/ideal_L3 $D/ideal_L4 $D/ideal_L5 $D/ideal_L6 $C/cascadepipe_L3_b0.1 $C/cascadepipe_L3_b0.2; do
  [ -f $d/small_probe.json ] || $PY -m experiments.pfa_scannet.small_probe --detector_maps $d --decoder_maps $d --out $d/small_probe.json > $d/small_probe.log 2>&1 &
done
wait
date; echo "deck scores done"
