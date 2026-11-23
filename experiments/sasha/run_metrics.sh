#!/bin/bash
# Slide-level predictions of SASHA-0.2, SASHA-0.1 and HAFED from the raw test slides (same code path as the timing
# runs). The machine may be shared while this runs, so the times it writes are not used.
cd ~/SASHA-Segmentation
export CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/tmp/venkatavks/oslib
W=/tmp/venkatavks/timing_ours
PY=~/envs/sasha-seg/bin/python
T=experiments/sasha/time_infer.py
$PY $T --mode sasha --config config/camelyon_seg_0.2.yml --save_dir $W/metrics_0.2
$PY $T --mode sasha --config config/camelyon_seg_0.1.yml --save_dir $W/metrics_0.1
$PY $T --mode hafed --save_dir $W/metrics_hafed
$PY experiments/sasha/cls_metrics.py $W/metrics_0.1/sasha $W/metrics_0.2/sasha $W/metrics_hafed/hafed > $W/cls_metrics.md 2>&1
touch $W/METRICS_DONE
