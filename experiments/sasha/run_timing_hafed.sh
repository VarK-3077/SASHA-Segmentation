#!/bin/bash
# Waits for the SASHA / Model-1 timing runs to end, re-times the three SASHA slides that overlapped a smoke test,
# then times the full high-res HAFED baseline on GPU 0.
cd ~/SASHA-Segmentation
export CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/tmp/venkatavks/oslib
W=/tmp/venkatavks/timing_ours
PY=~/envs/sasha-seg/bin/python
until [ -f $W/ALL_DONE ]; do sleep 60; done
mkdir -p $W/overlapped
for s in test_087 test_088 test_089; do
  mv $W/full_0.2/sasha/$s.json $W/overlapped/ 2>/dev/null
  find $W/full_0.2/sasha_work -name "$s*" -delete
done
$PY experiments/sasha/time_infer.py --mode sasha --config config/camelyon_seg_0.2.yml --save_dir $W/full_0.2 || touch $W/HAFED_FAILED
$PY experiments/sasha/time_infer.py --mode hafed --save_dir $W/full_hafed || touch $W/HAFED_FAILED
touch $W/HAFED_DONE
