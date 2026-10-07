#!/bin/bash
# Second pass of the separability study: adds the backbone size ladder (Kaiko ViT-S..L,
# DINOv2 ViT-S) at levels 3, 2, 1 on top of the existing features, then re-probes.
cd ~/SASHA-Segmentation
export PYTHONPATH=. HF_HOME=/data2/venkatavks/hf_cache TORCH_HOME=/data2/venkatavks/torch_cache
PY=~/envs/sasha-seg/bin/python
DEV=${1:-cuda:1}
BB="dinov2_s kaiko_vits16 kaiko_vits8 kaiko_vitb16 kaiko_vitb8 kaiko_vitl14"
set -e
for L in 3 2 1; do $PY -m experiments.sep.extract --level $L --device $DEV --backbones $BB; done
$PY -m experiments.sep.probe --device $DEV --levels 3 2 1 --backbones $BB
$PY -m experiments.sep.report
