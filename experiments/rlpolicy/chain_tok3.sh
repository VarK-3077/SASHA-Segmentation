#!/bin/bash
# Trains the six token-level L3 heads of one tag (five folds and the all-train head, three at a time), then
# predicts every slide and prints the out-of-fold comparison. Usage: chain_tok3.sh TAG [train args...]
source /media/internal_8T/venkatavks/rlp/env_prince_rlp.sh
TAG=$1; shift
L=$R/tok3/$TAG; mkdir -p $L
while [ ! -f $R/sub/meta.npz ]; do sleep 20; done
run() { [ -f $L/f$1/DONE ] || $PY -m experiments.rlpolicy.tok3_train --tag $TAG --fold $1 "${@:2}" > $L/train_f$1.log 2>&1; }
for trio in "0 1 2" "3 4 -1"; do
  for k in $trio; do run $k "$@" & done
  wait
done
$PY -m experiments.rlpolicy.tok3_predict --tag $TAG > $L/predict.log 2>&1
$PY -m experiments.rlpolicy.tok3_eval --tag $TAG > $L/oof_eval.txt 2>&1
touch $L/CHAIN_DONE
