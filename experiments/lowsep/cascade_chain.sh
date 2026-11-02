#!/bin/bash
# Screen + segment cascades on one GPU box, detached. The full level-2 token head baseline is the
# pfa-analysis session's existing run (tokhead_l2), not repeated here; its timing settings are
# matched (solo, cold cache per slide, 16 loader workers, batch 64, OMP_NUM_THREADS=1). Steps:
# validation choices (screen head, thresholds), the four cascades (L3 / L4 screen x 10 / 20 % val
# zoom budget) one at a time, clean_sweep on every map set with the baseline's validation-chosen
# rules, then the table. Usage: bash experiments/lowsep/cascade_chain.sh
cd ~/SASHA-Segmentation
export PYTHONPATH=. TORCH_HOME=/data2/venkatavks/torch_cache TOK_LEVEL=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CAMELYON16_MIRROR=/data2/venkatavks/c16_extra
PY=~/envs/sasha-seg/bin/python
R=/data2/venkatavks/lowsep/cascade
CK=/data2/venkatavks/pfa_ckpt/named/tokhead_kaiko_vitl14_stageA_input_l2.pt
set -e
mkdir -p $R
date; [ -f /data2/venkatavks/lowsep/cascade_cfg.json ] || $PY -m experiments.lowsep.cascade_prep
for L in 3 4; do for B in 0.1 0.2; do
  date; [ -f $R/cascade_L${L}_b$B/DONE ] || $PY -m experiments.lowsep.cascade --level $L --budget $B --seg_ckpt $CK --out_dir $R/cascade_L${L}_b$B --workers 16
done; done
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4
for d in $R/cascade_L*; do
  [ -f $d/clean.json ] || $PY -m experiments.pfa_scannet.clean_sweep --maps $d --thrs 0.4 --seeds 0.9 --areas 0 --workers 3 --out $d/clean.json > $d/clean.log 2>&1 &
done
wait
date; $PY -m experiments.lowsep.cascade_report $R
date; echo "cascade chain done"
