#!/bin/bash
# Found-lesion counts under each model's validation-chosen cleaned mask for the 2 Nov deck: the
# token head at input levels 0-3 and the two pipelined level-3 cascades.
# Output: /data2/venkatavks/lowsep/found/*.json. Log: /data2/venkatavks/lowsep/found_chain.log
cd ~/SASHA-Segmentation
export PYTHONPATH=. OMP_NUM_THREADS=1 CAMELYON16_MIRROR=/data2/venkatavks/c16_extra CUDA_VISIBLE_DEVICES=
PY=~/envs/sasha-seg/bin/python
D=/data2/venkatavks
O=$D/lowsep/found
mkdir -p $O
for spec in "l0 $D/tokhead_l0/out t0.5_seed0.98" "l1 $D/tokhead/out t0.5_seed0.98" "l2 $D/tokhead_l2/out t0.4_seed0.9" \
            "l3 $D/tokhead_l3/out t0.3" "pipe_b0.1 $D/lowsep/cascade/cascadepipe_L3_b0.1 t0.4_seed0.9" \
            "pipe_b0.2 $D/lowsep/cascade/cascadepipe_L3_b0.2 t0.4_seed0.9"; do
  set -- $spec
  [ -f $O/$1.json ] || $PY -m experiments.lowsep.found_counts --maps $2 --variant $3 --out $O/$1.json --workers 10
done
date; echo "found chain done"
