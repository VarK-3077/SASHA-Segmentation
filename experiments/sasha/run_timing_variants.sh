#!/bin/bash
# Follow-up to the timing audit, GPU 0: SASHA with the 16 crops of a tile read by 16 threads (20 % and 10 %),
# HAFED without the unused low-res reads, then FROC/Dice of the fp32 PFA-ScanNet maps the timed run wrote.
cd ~/SASHA-Segmentation
export CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/tmp/venkatavks/oslib
W=/tmp/venkatavks/timing_ours
PY=~/envs/sasha-seg/bin/python
T=experiments/sasha/time_infer.py
# another user's training shares the machine at times: start only once GPU 0 and the CPUs have been idle for 5 min
idle=0
until [ $idle -ge 5 ]; do
  mem=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
  load=$(cut -d" " -f1 /proc/loadavg | cut -d. -f1)
  if [ "$mem" -lt 500 ] && [ "$load" -lt 4 ]; then idle=$((idle+1)); else idle=0; fi
  sleep 60
done
date > $W/VARIANTS_STARTED
$PY $T --mode sasha --config config/camelyon_seg_0.2.yml --parallel_reads --save_dir $W/par_0.2 || touch $W/VARIANTS_FAILED
$PY $T --mode sasha --config config/camelyon_seg_0.1.yml --parallel_reads --save_dir $W/par_0.1 || touch $W/VARIANTS_FAILED
$PY $T --mode hafed --hr_only --save_dir $W/hafed_hr_only || touch $W/VARIANTS_FAILED
for D in fp32 bf16; do
  $PY -m experiments.pfa_scannet.score_maps --maps /tmp/venkatavks/pfa/timing/full_$D --out /tmp/venkatavks/pfa/timing/full_$D/_scores.json > /tmp/venkatavks/pfa/timing/full_$D/_scores.txt 2>&1
done
touch $W/VARIANTS_DONE
