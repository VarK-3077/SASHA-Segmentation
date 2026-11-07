#!/bin/bash
# Real runs on vivi GPU 1 of the settings chosen on segmenter outcomes: the RL stopper trained on them and
# the escalation rule.
cd /data1/venkatavks/SASHA-Segmentation
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 TORCH_HOME=/data2/venkatavks/torch_cache HF_HOME=/data2/venkatavks/hf_cache TOK_LEVEL=2 TOK_BACKBONE=kaiko_vitl14
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CAMELYON16_MIRROR=/var/tmp/venkatavks/c16_test
export LOWSEP_ROOT=/var/tmp/venkatavks/lowsep RLSCREEN_ROOT=/var/tmp/venkatavks/rlscreen
PY=~/envs/sasha-seg/bin/python
R=/var/tmp/venkatavks/real
rr() { name=$1; shift; [ -f $R/$name/DONE ] && return; date; $PY -m experiments.rlscreen.run_real --out_dir $R/$name "$@" < /dev/null > $R/$name.log 2>&1; }
rr les_rl_s0_l08 --design coarse --g_level 6 --actor seq --tag les_z05_s0 --lam 0.849 --tau_zoom 0.5 --paint 0.99 --cluster 1
rr les_rl_s2_l08 --design coarse --g_level 6 --actor seq --tag les_z05_s2 --lam 0.849 --tau_zoom 0.5 --paint 0.99 --cluster 1
rr esc_08_06_z07 --design coarse --g_level 6 --actor seq --stages 0.8 0.6 --tau_zoom 0.7
rr esc_08_06_z07_paint --design coarse --g_level 6 --actor seq --stages 0.8 0.6 --tau_zoom 0.7 --paint 0.99
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2
for d in $R/les_rl_*/ $R/esc_*/; do
  [ -f $d/clean.json ] || $PY -m experiments.pfa_scannet.clean_sweep --maps $d --thrs 0.4 --seeds 0.9 --areas 0 --workers 3 --out $d/clean.json > $d/clean.log 2>&1 &
done
wait
$PY -m experiments.rlscreen.real_vivi_report $R
date; echo "real les done"
