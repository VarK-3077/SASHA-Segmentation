#!/bin/bash
# Seed study chain on vivi: train (both GPUs, rl3_seeds.py), exact tables on cross-fitted train and on test
# (seeds_eval.py, CPU), then the timing window (time_seeds.sh). Holds /var/tmp/venkatavks/queue/SEEDS_TRAINING
# while training. Waits for any TIMING_HOLD to clear before training. Usage: chain_seeds.sh DEVICES... (e.g. cuda:0 cuda:1)
source /var/tmp/venkatavks/rlpolicy/env_vivi_rlp.sh
Q=/var/tmp/venkatavks/queue
SEEDS="0 1 2 3 4"
E=$R/eval
mkdir -p $E
log() { echo "$(date '+%F %T') $*" >> $R/seeds_chain.log; }
while [ -f $Q/TIMING_HOLD ] || [ -f $Q/RLPOLICY_YIELD ]; do sleep 60; done
if [ ! -f $R/search/SEEDS_DONE ] && [ ! -f $R/search/SEEDS_SMOKE_OK ]; then
  log "smoke start"
  $PY -m experiments.rlpolicy.rl3_seeds --seeds 9 --devices "$@" --workers 6 --updates 1 --episodes 4 > $R/search/seeds_smoke.log 2>&1 || { log "smoke FAILED"; exit 1; }
  $PY -m experiments.rlpolicy.seeds_eval --split train --seeds 9 --out $E/seeds_smoke.md --workers 24 > $E/seeds_smoke.log 2>&1 || { log "smoke eval FAILED"; exit 1; }
  rm -rf $R/search/seeds_s9 $R/search/SEEDS_DONE; touch $R/search/SEEDS_SMOKE_OK
  log "smoke ok"
fi
if [ ! -f $R/search/SEEDS_DONE ]; then
  echo "nov23-presentation seed training $(date)" > $Q/SEEDS_TRAINING
  log "training start (devices $*)"
  $PY -m experiments.rlpolicy.rl3_seeds --seeds $SEEDS --devices "$@" --workers 30 > $R/search/seeds.log 2>&1 || { log "training FAILED"; rm -f $Q/SEEDS_TRAINING; exit 1; }
  rm -f $Q/SEEDS_TRAINING
  log "training done"
fi
[ -f $E/seeds_train.md ] || $PY -m experiments.rlpolicy.seeds_eval --split train --seeds $SEEDS --out $E/seeds_train.md --workers 24 --from_pkl > $E/seeds_train.log 2>&1 || { log "train tables FAILED"; exit 1; }
log "train tables done"
[ -f $E/seeds_test.md ] || $PY -m experiments.rlpolicy.seeds_eval --split test --seeds $SEEDS --out $E/seeds_test.md --workers 24 --from_pkl > $E/seeds_test.log 2>&1 || { log "test tables FAILED"; exit 1; }
log "test tables done"
bash $R/time_seeds.sh $SEEDS
log "timing done"
touch $R/SEEDS_CHAIN_DONE
