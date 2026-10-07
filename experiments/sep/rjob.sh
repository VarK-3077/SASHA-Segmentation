#!/bin/bash
# Runs the whole separability study detached on the server: labels -> sample -> features
# per level -> probes -> report. Log goes to /data2/venkatavks/sep/run.log.
cd ~/SASHA-Segmentation
export PYTHONPATH=. HF_HOME=/data2/venkatavks/hf_cache
PY=~/envs/sasha-seg/bin/python
DEV=${1:-cuda:1}
set -e
[ -f /data2/venkatavks/sep/lesions.csv ] || $PY -m experiments.sep.build_labels 16
[ -f /data2/venkatavks/sep/crops_L0.npz ] || $PY -m experiments.sep.sample
for L in 3 2 1 0; do $PY -m experiments.sep.extract --level $L --device $DEV; done
$PY -m experiments.sep.probe --device $DEV
$PY -m experiments.sep.report
