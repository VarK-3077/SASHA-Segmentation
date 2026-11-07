#!/bin/bash
# vivi, CPU only: optimal stop + core critic. eps 0 = plain p_core >= 0.99 rule, no critic.
cd /data1/venkatavks/SASHA-Segmentation
export PYTHONPATH=. CUDA_VISIBLE_DEVICES= RLSCREEN_ROOT=/var/tmp/venkatavks/rlscreen RLSCREEN_OOF=1 OMP_NUM_THREADS=3
PY=~/envs/sasha-seg/bin/python
R=$RLSCREEN_ROOT/runs
run() { tag=$1; shift; mkdir -p $R/$tag; $PY -m experiments.rlscreen.qstop --tag $tag "$@" > $R/$tag/log.txt 2>&1; }
run qs_full_e0 --design full --eps 0 &
run qs_full_e01 --design full --eps 0.01 &
run qs_full_e02 --design full --eps 0.02 &
run qs_full_e05 --design full --eps 0.05 &
run qs_full_e02_b1 --design full --eps 0.02 --batch 0.01 &
run qs_full_e02_s1 --design full --eps 0.02 --seed 1 &
wait
echo "qstop chain done"
