#!/bin/bash
# RL vs supervised for SASHA's slide classification, end to end on vivi GPU 1 (CUDA device 1 only):
#  1 low-res token cache of every slide (lr_tokens.py) and offline rollouts of the random rule for 5 seeds + RL
#  2 the three tile classifiers (tile_mlp.py: linear, mlp, juiced), chosen on validation
#  3 a 2-slide smoke run of every timed variant
#  4 after the seed study's own timing (SEEDS_CHAIN_DONE), one quiet timing window with time_infer.py: SASHA as
#    released, SASHA with random / uncertain / greedy picks, and the three tile classifiers, all 129 test slides
#  5 cls_report.py (Accuracy, AUC, F1, ECE, time, compressibility, paired bootstrap) and rule_seeds.py
# Never signals other processes; waits for TIMING_HOLD / RLPOLICY_YIELD; finished steps are skipped on a relaunch.
cd ~/SASHA-Segmentation
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4
# the authors' bundled openslide wrapper loads libopenslide.so.0; the venv ships openslide 4 as .so.1 (the earlier
# timing runs pointed LD_LIBRARY_PATH at the same kind of link)
mkdir -p /var/tmp/venkatavks/oslib
ln -sf /data1/venkatavks/envs/sasha-seg/lib/python3.11/site-packages/openslide_bin/libopenslide.so.1 /var/tmp/venkatavks/oslib/libopenslide.so.0
export LD_LIBRARY_PATH=/var/tmp/venkatavks/oslib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
PY=~/envs/sasha-seg/bin/python
Q=/var/tmp/venkatavks/queue
C=/var/tmp/venkatavks/sasha_cls
T="$PY experiments/sasha/time_infer.py"
mkdir -p $C/offline $C/bundles $C/timed $C/smoke
log() { echo "$(date '+%F %T') $*" >> $C/chain.log; }
TEST=$($PY -c "import json; print(' '.join(json.load(open('dataset_csv/camelyon16/splits/split_4.json'))['test_names']))")
wait_free() { while [ -f $Q/TIMING_HOLD ] || [ -f $Q/RLPOLICY_YIELD ]; do sleep 60; done; }

wait_free
if [ ! -f $C/lr_tokens/DONE ]; then log "token cache start"; $PY -m experiments.sasha.lr_tokens --out $C/lr_tokens --device cuda:0 > $C/lr_tokens.log 2>&1 || { log "token cache FAILED"; exit 1; }; log "token cache done"; fi
for s in 0 1 2 3 4; do
  [ -f $C/offline/random/s$s/rollout_random.pkl ] || $PY -m experiments.sasha.rule_rollouts --rule_seed $s -- --policy random --device cuda:0 --out_dir $C/offline/random/s$s > $C/offline/random_s$s.log 2>&1 || log "offline random s$s FAILED"
done
[ -f $C/offline/rl/s0/rollout_rl.pkl ] || $PY -m experiments.sasha.rule_rollouts --rule_seed 0 -- --policy rl --device cuda:0 --out_dir $C/offline/rl/s0 > $C/offline/rl.log 2>&1 || log "offline rl FAILED"
log "offline rollouts done"
for k in linear mlp juiced; do
  [ -f $C/bundles/$k/cfg.json ] || $PY -m experiments.sasha.tile_mlp --kind $k --tokens $C/lr_tokens --out $C/bundles/$k --device cuda:0 > $C/bundles/$k.log 2>&1 || { log "train $k FAILED"; exit 1; }
done
log "tile classifiers trained"

SMOKE="test_001 test_002"
if [ ! -f $C/smoke/OK ]; then
  rm -rf $C/smoke; mkdir -p $C/smoke
  for p in rl random uncertain greedy; do $T --mode sasha --policy $p --slides $SMOKE --save_dir $C/smoke/sasha_$p > $C/smoke/sasha_$p.log 2>&1 || { log "smoke sasha $p FAILED"; exit 1; }; done
  for k in linear mlp juiced; do $T --mode tilemlp --mlp_bundle $C/bundles/$k --slides $SMOKE --save_dir $C/smoke/$k > $C/smoke/$k.log 2>&1 || { log "smoke $k FAILED"; exit 1; }; done
  touch $C/smoke/OK; log "smoke ok"
fi

while [ ! -f /var/tmp/venkatavks/rlpolicy/SEEDS_CHAIN_DONE ]; do sleep 120; done
arl() { for d in /proc/[0-9]*; do c=$(tr '\0' ' ' < $d/cmdline 2>/dev/null); w=$(readlink $d/cwd 2>/dev/null)
  case "$c$w" in *chain_cls*) ;; *common.verifier_server*|*common.trainer*|*vllm*|*VLLM*|*/data3/venkatavks/arl_*) echo x; return ;; esac; done; }
gpu_mb() { $PY -c "
import torch
print(max(int((t - f) / 2**20) for f, t in (torch.cuda.mem_get_info(i) for i in range(torch.cuda.device_count()))))" 2>/dev/null || echo 99999; }
ok=0
while [ $ok -lt 3 ]; do
  load=$(cut -d' ' -f1 /proc/loadavg); a=$(arl); g=$(CUDA_VISIBLE_DEVICES=0,1 gpu_mb)
  if [ -z "$a" ] && [ ! -f $Q/TIMING_HOLD ] && [ ! -f $Q/RLPOLICY_YIELD ] && [ "$g" -le 1536 ] && awk "BEGIN{exit !($load <= 4)}"; then ok=$((ok + 1)); else ok=0; fi
  log "check: arl ${a:-none}, load $load, gpu max ${g} MB, idle streak $ok"
  [ $ok -lt 3 ] && sleep 120
done
echo "sasha classification timing (nov23-presentation) $(date)" > $Q/TIMING_HOLD
unset OMP_NUM_THREADS   # the timed rows run with the authors' defaults, as the earlier timing runs did
trap 'rm -f $Q/TIMING_HOLD' EXIT
row() { name=$1; shift; [ -f $C/timed/$name.DONE ] && return; log "start $name, $(uptime)"; "$@" --slides $TEST --save_dir $C/timed/$name < /dev/null > $C/timed/$name.log 2>&1 && touch $C/timed/$name.DONE || log "$name FAILED"; log "end $name, $(uptime)"; }
row sasha_rl $T --mode sasha --policy rl
row sasha_random $T --mode sasha --policy random
row sasha_uncertain $T --mode sasha --policy uncertain
row sasha_greedy $T --mode sasha --policy greedy
for k in linear mlp juiced; do row tile_$k $T --mode tilemlp --mlp_bundle $C/bundles/$k; done
rm -f $Q/TIMING_HOLD
trap - EXIT
log "timing done, scoring"
$PY -m experiments.sasha.cls_report --out $C/cls_report.md --runs "SASHA-0.2 as released=$C/timed/sasha_rl/sasha" \
  "random picks=$C/timed/sasha_random/sasha_random" "uncertainty picks=$C/timed/sasha_uncertain/sasha_uncertain" \
  "greedy picks=$C/timed/sasha_greedy/sasha_greedy" "linear tile classifier (CLS, max)=$C/timed/tile_linear/tilemlp" \
  "MLP tile classifier (CLS, max)=$C/timed/tile_mlp/tilemlp" "juiced MLP (token mean+max, top-k)=$C/timed/tile_juiced/tilemlp" > $C/cls_report.log 2>&1
$PY -m experiments.sasha.rule_seeds --out $C/rule_seeds.md --dirs "random picks=$C/offline/random" "SASHA RL=$C/offline/rl" >> $C/cls_report.log 2>&1
log "done"; touch $C/CHAIN_DONE
