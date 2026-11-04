#!/bin/bash
# Real-segmenter runs on the A100, one at a time so wall clocks are clean, then clean_sweep with
# the level-2 head's validation-chosen rules and the table. Each line of RUNS is
# "<out name> <run_real.py arguments>". Usage: bash experiments/rlscreen/real_chain.sh RUNS.txt
cd ~/SASHA-Segmentation
export PYTHONPATH=. TORCH_HOME=/data2/venkatavks/torch_cache TOK_LEVEL=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CAMELYON16_MIRROR=/data2/venkatavks/c16_extra
PY=~/envs/sasha-seg/bin/python
R=/data2/venkatavks/rlscreen/real
mkdir -p $R
while read -r name args; do
  [ -z "$name" ] || [[ $name == \#* ]] && continue
  date; [ -f $R/$name/DONE ] || $PY -m experiments.rlscreen.run_real --out_dir $R/$name $args < /dev/null
done < $1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4
for d in $R/*/; do
  [ -f $d/clean.json ] || $PY -m experiments.pfa_scannet.clean_sweep --maps $d --thrs 0.4 --seeds 0.9 --areas 0 --workers 3 --out $d/clean.json > $d/clean.log 2>&1 &
done
wait
date; $PY -m experiments.rlscreen.real_report
date; echo "real chain done"
