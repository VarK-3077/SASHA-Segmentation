#!/bin/bash
# After the variants: (1) the same 10 slides through every pipeline, interleaved, to measure repeatability;
# (2) PFA-ScanNet fp32 with TF32 left on (PyTorch default) and bf16 with 24 loader workers, 5 slides each.
cd ~/SASHA-Segmentation
export CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/tmp/venkatavks/oslib OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
W=/tmp/venkatavks/timing_ours
R=$W/repeat
PY=~/envs/sasha-seg/bin/python
T=experiments/sasha/time_infer.py
CK=/tmp/venkatavks/pfa/run1/ckpt/it_0150000.pt
until [ -f $W/VARIANTS_DONE ]; do sleep 60; done
for s in test_005 test_018 test_027 test_040 test_052 test_066 test_079 test_092 test_105 test_121; do
  $PY -m experiments.pfa_scannet.time_infer --ckpt $CK --out_dir $R/pfa_fp32 --slides $s
  $PY $T --mode sasha --config config/camelyon_seg_0.2.yml --save_dir $R/sasha_0.2 --slides $s
  $PY $T --mode hafed --save_dir $R/hafed --slides $s
  $PY -m experiments.pfa_scannet.time_infer --ckpt $CK --out_dir $R/pfa_bf16 --dtype bf16 --slides $s
  $PY $T --mode model1 --save_dir $R/model1 --slides $s
  $PY $T --mode sasha --config config/camelyon_seg_0.2.yml --parallel_reads --save_dir $R/sasha_0.2_par --slides $s
done
S5="test_005 test_027 test_052 test_079 test_105"
$PY -m experiments.pfa_scannet.time_infer --ckpt $CK --out_dir $R/pfa_fp32_tf32 --tf32 --slides $S5
$PY -m experiments.pfa_scannet.time_infer --ckpt $CK --out_dir $R/pfa_bf16_w24 --dtype bf16 --workers 24 --slides $S5
touch $W/REPEAT_DONE
