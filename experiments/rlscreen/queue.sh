#!/bin/bash
# Runs the training jobs listed in a file, K at a time on one GPU: each line is "<tag> <train.py
# arguments>"; every job is trained, then evaluated offline on the test slides. Jobs whose
# offline report already exists are skipped, so the queue can be restarted.
# Usage: bash experiments/rlscreen/queue.sh JOBS.txt cuda:0 3. Logs: /data2/venkatavks/rlscreen/runs/<tag>/log.txt
cd ~/SASHA-Segmentation
export PYTHONPATH=. OMP_NUM_THREADS=2
PY=~/envs/sasha-seg/bin/python
JOBS=$1; DEV=${2:-cuda:0}; K=${3:-3}
R=${RLSCREEN_ROOT:-/data2/venkatavks/rlscreen}
job() {
  tag=$1; shift
  [ -f $R/offline_$tag.json ] && return 0
  mkdir -p $R/runs/$tag
  design=$(sed -n 's/.*--design \([a-z]*\).*/\1/p' <<< "$*"); g=$(sed -n 's/.*--g_level \([0-9]\).*/\1/p' <<< "$*"); bb=$(sed -n 's/.*--backbone \([a-z0-9_]*\).*/\1/p' <<< "$*")
  { [ -f $R/runs/$tag/final.pt ] || $PY -m experiments.rlscreen.train --tag $tag --device $DEV "$@" > $R/runs/$tag/log.txt 2>&1; } \
    && $PY -m experiments.rlscreen.evaluate --tag $tag --design $design --g_level ${g:-6} --backbone ${bb:-kaiko_vitl14} --device $DEV >> $R/runs/$tag/log.txt 2>&1
}
while read -r line; do
  [ -z "$line" ] || [[ $line == \#* ]] && continue
  job $line &
  while [ $(jobs -rp | wc -l) -ge $K ]; do sleep 20; done
done < $JOBS
wait
date; echo "queue $JOBS done"
