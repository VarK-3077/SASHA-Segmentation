#!/bin/bash
# Runs the RL binning study detached: per-slide cache -> PPO training -> test evaluation.
cd ~/SASHA-Segmentation
export PYTHONPATH=.
PY=~/envs/sasha-seg/bin/python
DEV=${1:-cuda:0}
TAG=${2:-l3}
set -e
$PY -m experiments.rlbin.cache $DEV
$PY -m experiments.rlbin.train --device $DEV --tag $TAG
$PY -m experiments.rlbin.evaluate --device $DEV --tag $TAG
