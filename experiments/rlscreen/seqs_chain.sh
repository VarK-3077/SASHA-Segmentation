#!/bin/bash
# vivi, CPU only: L6 -> L3 at zoom 0.5 with painting: rules including escalation, then RL stoppers.
cd /data1/venkatavks/SASHA-Segmentation
export PYTHONPATH=. CUDA_VISIBLE_DEVICES= RLSCREEN_ROOT=/var/tmp/venkatavks/rlscreen RLSCREEN_OOF=1 OMP_NUM_THREADS=3
PY=~/envs/sasha-seg/bin/python
R=$RLSCREEN_ROOT/runs
M="$PY -m experiments.rlscreen.stoprl"
run() { tag=$1; shift; mkdir -p $R/$tag; $M --tag $tag --design coarse --tz 0.5 "$@" > $R/$tag/log.txt 2>&1; }
mkdir -p $R/rules
$M --design coarse --tz 0.5 --rules_only > $R/rules/coarse_z05.txt 2>&1
run seqs_s0 --cluster 1 --seed 0 &
run seqs_s1 --cluster 1 --seed 1 &
run seqs_s2 --cluster 1 --seed 2 &
run seqs_nocluster --seed 0 &
run seqs_long --cluster 1 --updates 1200 &
wait
echo "seqs chain done"
