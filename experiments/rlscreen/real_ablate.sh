#!/bin/bash
# Real fine-to-coarse feedback ablation on vivi GPU 1 (settings from ablate.py, out-of-fold train only):
# L6 -> L3, screen 0.6, zoom 0.5. One run at a time, cold cache per slide; then per-slide scoring and
# paired bootstraps against rows 2 and 3.
cd /data1/venkatavks/SASHA-Segmentation
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 TORCH_HOME=/data2/venkatavks/torch_cache HF_HOME=/data2/venkatavks/hf_cache TOK_LEVEL=2 TOK_BACKBONE=kaiko_vitl14
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CAMELYON16_MIRROR=/var/tmp/venkatavks/c16_test
export LOWSEP_ROOT=/var/tmp/venkatavks/lowsep RLSCREEN_ROOT=/var/tmp/venkatavks/rlscreen
PY=~/envs/sasha-seg/bin/python
R=/var/tmp/venkatavks/real_ablate
CR=$RLSCREEN_ROOT/ablate_critic.pt
mkdir -p $R
B="--design coarse --g_level 6 --actor seq --tau 0.6 --tau_zoom 0.5"
rr() { name=$1; shift; [ -f $R/$name/DONE ] && return; date; $PY -m experiments.rlscreen.run_real --out_dir $R/$name $B "$@" < /dev/null > $R/$name.log 2>&1; }
rr r1_fixed --rule fixed --follow 0 --paint 2
rr r2_paint --rule fixed --follow 0 --paint 0.99
rr r3_escalate --rule esc --stages 0.8 0.7 --minpos 4 --minfrac 0.05 --mode last --follow 0 --paint 0.99
rr r4_trace --rule fixed --follow 1 --paint 0.99
rr r5_interior --rule fixed --follow 0 --paint 0.99 --critic $CR --eps 0.005
rr r6_all --rule esc --stages 0.8 0.75 0.7 --minpos 4 --minfrac 0.05 --mode last --follow 1 --paint 0.99 --critic $CR --eps 0.005
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1
$PY -m experiments.rlscreen.ablate_score --root $R --runs r1_fixed r2_paint r3_escalate r4_trace r5_interior r6_all --refs r3_escalate r2_paint --workers 4 > $R/report.md 2>&1
date; echo "real ablate done"
