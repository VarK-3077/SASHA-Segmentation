#!/bin/bash
# Pipelined L3 screen + segment cascade (persistent reader pools, segmentation overlapping the
# screen) at the same two validation budgets as cascade_chain.sh, then clean_sweep with the
# baseline's rules and the joint table. Same environment as the baseline timing.
# Usage: bash experiments/lowsep/cascade_pipe_chain.sh
cd ~/SASHA-Segmentation
export PYTHONPATH=. TORCH_HOME=/data2/venkatavks/torch_cache TOK_LEVEL=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CAMELYON16_MIRROR=/data2/venkatavks/c16_extra
PY=~/envs/sasha-seg/bin/python
R=/data2/venkatavks/lowsep/cascade
CK=/data2/venkatavks/pfa_ckpt/named/tokhead_kaiko_vitl14_stageA_input_l2.pt
set -e
for B in 0.1 0.2; do
  date; [ -f $R/cascadepipe_L3_b$B/DONE ] || $PY -m experiments.lowsep.cascade_pipe --level 3 --budget $B --seg_ckpt $CK --out_dir $R/cascadepipe_L3_b$B
done
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4
for d in $R/cascadepipe_L3_*; do
  [ -f $d/clean.json ] || $PY -m experiments.pfa_scannet.clean_sweep --maps $d --thrs 0.4 --seeds 0.9 --areas 0 --workers 6 --out $d/clean.json > $d/clean.log 2>&1 &
done
wait
date; $PY -m experiments.lowsep.cascade_report $R
date; echo "pipe chain done"
