#!/bin/bash
# Timed inference of SASHA (20 % and 10 % budgets) and Model-1 over the 129 CAMELYON16 test slides, GPU 0.
cd ~/SASHA-Segmentation
export CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/tmp/venkatavks/oslib
W=/tmp/venkatavks/timing_ours
PY=~/envs/sasha-seg/bin/python
$PY experiments/sasha/time_infer.py --mode sasha --config config/camelyon_seg_0.2.yml --save_dir $W/full_0.2 || touch $W/FAILED
$PY experiments/sasha/time_infer.py --mode model1 --save_dir $W/full_m1 || touch $W/FAILED
$PY experiments/sasha/time_infer.py --mode sasha --config config/camelyon_seg_0.1.yml --save_dir $W/full_0.1 || touch $W/FAILED
touch $W/ALL_DONE
