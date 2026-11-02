#!/bin/bash
# Screen + segment vs full level-2 token head, one GPU box, detached. Validation choices first
# (head and thresholds), then the full token head on every test slide (tokhead/infer.py), then
# the four cascades (L3 / L4 screen x 10 / 20 % zoom budget), one GPU job at a time so the wall
# clocks are clean, then score_maps on every map set at the token head's validation-chosen
# operating point, then the table. Usage: bash experiments/lowsep/cascade_chain.sh <seg ckpt> "<thr> <open> <cands>"
cd ~/SASHA-Segmentation
export PYTHONPATH=. TORCH_HOME=/data2/venkatavks/torch_cache TOK_LEVEL=2 OMP_NUM_THREADS=2
PY=~/envs/sasha-seg/bin/python
R=/data2/venkatavks/lowsep/cascade
CK=$1
set -- $2; THR=$1; OPEN=$2; CANDS=$3
set -e
mkdir -p $R
date; [ -f /data2/venkatavks/lowsep/cascade_cfg.json ] || $PY -m experiments.lowsep.cascade_prep
date; [ -f $R/full_l2/DONE ] || $PY -m experiments.tokhead.infer --ckpt $CK --out_dir $R/full_l2 --workers 10 --label full_l2
for L in 3 4; do for B in 0.1 0.2; do
  date; [ -f $R/cascade_L${L}_b$B/DONE ] || $PY -m experiments.lowsep.cascade --level $L --budget $B --seg_ckpt $CK --out_dir $R/cascade_L${L}_b$B
done; done
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4
for d in $R/full_l2 $R/cascade_L*; do
  [ -f $d/scores.json ] || $PY -m experiments.pfa_scannet.score_maps --maps $d --thr $THR --open $OPEN --cands $CANDS --out $d/scores.json > $d/score.log 2>&1 &
done
wait
date; $PY -m experiments.lowsep.cascade_report $R
date; echo "cascade chain done"
