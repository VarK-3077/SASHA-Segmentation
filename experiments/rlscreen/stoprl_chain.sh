#!/bin/bash
# vivi, CPU only: rules on both rankings (with core painting), then RL stoppers and RL core selection.
# Usage: bash experiments/rlscreen/stoprl_chain.sh
cd /data1/venkatavks/SASHA-Segmentation
export PYTHONPATH=. CUDA_VISIBLE_DEVICES= RLSCREEN_ROOT=/var/tmp/venkatavks/rlscreen RLSCREEN_OOF=1 OMP_NUM_THREADS=3
PY=~/envs/sasha-seg/bin/python
R=$RLSCREEN_ROOT/runs
M="$PY -m experiments.rlscreen.stoprl"
run() { tag=$1; shift; mkdir -p $R/$tag; $M --tag $tag "$@" > $R/$tag/log.txt 2>&1; }
mkdir -p $R/rules
$M --design full --rules_only > $R/rules/full.txt 2>&1 &
$M --design coarse --rules_only > $R/rules/coarse.txt 2>&1 &
wait
run stop_full_s0 --design full --seed 0 &
run stop_full_s1 --design full --seed 1 &
run stop_full_long --design full --updates 1200 &
run stop_full_nobc --design full --bc_episodes 0 &
wait
run core_full_rl --design full --tcs 2 0.99 0.9 &
run core_full_nofp --design full --tcs 2 0.99 0.9 --fp 0 &
run core_coarse_rl --design coarse --tcs 2 0.99 0.9 &
run stop_coarse_s0 --design coarse &
wait
echo "stoprl chain done"
