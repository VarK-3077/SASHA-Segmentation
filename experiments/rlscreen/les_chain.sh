#!/bin/bash
# vivi, CPU only, after train maps: outcome tables, rules, and RL stoppers on real segmenter outcomes
# with the lesion-aware quality; L6 -> L3 with painting.
cd /data1/venkatavks/SASHA-Segmentation
export PYTHONPATH=. CUDA_VISIBLE_DEVICES= RLSCREEN_ROOT=/var/tmp/venkatavks/rlscreen LOWSEP_ROOT=/var/tmp/venkatavks/lowsep RLSCREEN_OOF=1 OMP_NUM_THREADS=2
PY=~/envs/sasha-seg/bin/python
R=$RLSCREEN_ROOT/runs
M="$PY -m experiments.rlscreen.stoprl"
until grep -q "train maps done\|infer failed" /var/tmp/venkatavks/train_maps.log; do sleep 60; done
grep -q "infer failed" /var/tmp/venkatavks/train_maps.log && { echo "les chain stopped: train maps failed"; exit 1; }
$PY -m experiments.rlscreen.seg_tables --maps /var/tmp/venkatavks/real_train /var/tmp/venkatavks/real/full_l2
mkdir -p $R/rules
$M --design coarse --tz 0.5 --outcome seg --cluster 1 --rules_only --rules_tag _seg_z05 > $R/rules/seg_z05.txt 2>&1 &
$M --design coarse --tz 0.7 --outcome seg --cluster 1 --rules_only --rules_tag _seg_z07 > $R/rules/seg_z07.txt 2>&1 &
$M --design coarse --tz 0.3 --outcome seg --cluster 1 --rules_only --rules_tag _seg_z03 > $R/rules/seg_z03.txt 2>&1 &
wait
run() { tag=$1; shift; mkdir -p $R/$tag; $M --tag $tag --design coarse --outcome seg --cluster 1 --updates 1200 "$@" > $R/$tag/log.txt 2>&1; }
run les_z05_s0 --tz 0.5 --seed 0 &
run les_z05_s1 --tz 0.5 --seed 1 &
run les_z05_s2 --tz 0.5 --seed 2 &
run les_z03_s0 --tz 0.3 --seed 0 &
wait
echo "les chain done"
