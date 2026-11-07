#!/bin/bash
# Every real-segmenter timing on one machine: vivi GPU 1, test slides on NVMe, one run at a time, cold
# page cache per slide. Full level-2 head, supervised L3 cascade, L6->L3 rule, hit-rate rule, PPO stopper.
# Then clean_sweep on every map set and the table. Usage: bash experiments/rlscreen/real_vivi.sh
cd /data1/venkatavks/SASHA-Segmentation
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 TORCH_HOME=/data2/venkatavks/torch_cache HF_HOME=/data2/venkatavks/hf_cache TOK_LEVEL=2 TOK_BACKBONE=kaiko_vitl14
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CAMELYON16_MIRROR=/var/tmp/venkatavks/c16_test
export LOWSEP_ROOT=/var/tmp/venkatavks/lowsep RLSCREEN_ROOT=/var/tmp/venkatavks/rlscreen
PY=~/envs/sasha-seg/bin/python
R=/var/tmp/venkatavks/real
CK=/data2/venkatavks/pfa_ckpt/named/tokhead_kaiko_vitl14_stageA_input_l2.pt
mkdir -p $R
until grep -q mirror-done /var/tmp/venkatavks/vivi_disk.log; do sleep 30; done
rr() { name=$1; shift; [ -f $R/$name/DONE ] && return; date; $PY -m experiments.rlscreen.run_real --out_dir $R/$name "$@" < /dev/null > $R/$name.log 2>&1; }
date
[ -f $R/full_l2/DONE ] || { $PY -m experiments.tokhead.infer --ckpt $CK --out_dir $R/full_l2 --label full_l2_vivi > $R/full_l2.log 2>&1 && touch $R/full_l2/DONE; }
[ -f $R/cascade_L3_b0.2/DONE ] || $PY -m experiments.lowsep.cascade_pipe --level 3 --budget 0.2 --seg_ckpt $CK --out_dir $R/cascade_L3_b0.2 > $R/cascade_L3_b0.2.log 2>&1
rr coarse_z07 --design coarse --g_level 6 --actor teacher --tau 0.5 --tau_zoom 0.7
rr coarse_z07_paint --design coarse --g_level 6 --actor teacher --tau 0.5 --tau_zoom 0.7 --paint 0.99
rr hit_paint --design full --actor seq --h 0.02 --paint 0.99
rr ppo_l02 --design full --actor seq --tag stop_full_s0 --lam 0.204 --paint 0.99
rr ppo_l04 --design full --actor seq --tag stop_full_s0 --lam 0.416 --paint 0.99
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2
for d in $R/*/; do
  [ -f $d/clean.json ] || $PY -m experiments.pfa_scannet.clean_sweep --maps $d --thrs 0.4 --seeds 0.9 --areas 0 --workers 3 --out $d/clean.json > $d/clean.log 2>&1 &
done
wait
$PY -m experiments.rlscreen.real_vivi_report $R
date; echo "real vivi done"
