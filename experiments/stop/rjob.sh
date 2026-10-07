#!/bin/bash
# Runs the stopping-ceiling study detached: heads -> rollouts (train/val/test) -> analysis.
cd ~/SASHA-Segmentation
export PYTHONPATH=.
PY=~/envs/sasha-seg/bin/python
DEV=${1:-cuda:1}
set -e
[ -f /data2/venkatavks/stop/heads.pt ] || $PY -m experiments.stop.heads
$PY -m experiments.stop.rollout --device $DEV
$PY -m experiments.stop.analyze
