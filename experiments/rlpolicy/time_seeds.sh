#!/bin/bash
# One timing window on vivi GPU 1 for the seed study: the supervised rule, the teacher (consensus veto rule) and the
# PPO and clone networks of every seed (rl3_seeds.py), one run at a time, cold page cache per slide, L2 token head.
# Waits for an idle machine on 3 checks 2 min apart (no ARL/Lean process, load <= 4, every GPU <= 1.5 GB, no
# TIMING_HOLD or RLPOLICY_YIELD), holds TIMING_HOLD during the runs, never signals other processes, then scores the
# real maps (seeds_eval --real). Finished rows are skipped on a relaunch. Usage: time_seeds.sh SEED [SEED ...]
source /var/tmp/venkatavks/rlpolicy/env_vivi_rlp.sh
export RLSCREEN_ROOT=/var/tmp/venkatavks/rlcascade/oofroot
Q=/var/tmp/venkatavks/queue
T=$R/timing/seeds
mkdir -p $T
log() { echo "$(date '+%F %T') $*" >> $T/gate.log; }
arl() { for d in /proc/[0-9]*; do c=$(tr '\0' ' ' < $d/cmdline 2>/dev/null); w=$(readlink $d/cwd 2>/dev/null)
  case "$c$w" in *time_seeds*) ;; *common.verifier_server*|*common.trainer*|*vllm*|*VLLM*|*/data3/venkatavks/arl_*) echo x; return ;; esac; done; }
gpu_mb() { $PY -c "
import torch
print(max(int((t - f) / 2**20) for f, t in (torch.cuda.mem_get_info(i) for i in range(torch.cuda.device_count()))))" 2>/dev/null || echo 99999; }
ok=0
while [ $ok -lt 3 ]; do
  load=$(cut -d' ' -f1 /proc/loadavg); a=$(arl); g=$(gpu_mb)
  if [ -z "$a" ] && [ ! -f $Q/TIMING_HOLD ] && [ ! -f $Q/RLPOLICY_YIELD ] && [ "$g" -le 1536 ] && awk "BEGIN{exit !($load <= 4)}"; then ok=$((ok + 1)); else ok=0; fi
  log "check: arl ${a:-none}, load $load, gpu max ${g} MB, hold $([ -f $Q/TIMING_HOLD ] && echo yes || echo no), idle streak $ok"
  [ $ok -lt 3 ] && sleep 120
done
echo "rlpolicy timing seeds (nov23-presentation) $(date)" > $Q/TIMING_HOLD
trap 'rm -f $Q/TIMING_HOLD' EXIT
export CUDA_VISIBLE_DEVICES=1
row() { name=$1; shift; [ -f $T/$name/DONE ] && return; log "start $name, $(uptime)"; "$@" --out_dir $T/$name < /dev/null > $T/$name.log 2>&1 || log "$name FAILED"; log "end $name, $(uptime)"; }
row rule $PY -m experiments.rlcascade.run_real_rl --model rule
row teacher $PY -m experiments.rlpolicy.run_staged --theta $R/timing/theta_top16_k05.json --tok3 t1
RUNS="rule=$T/rule teacher=$T/teacher"
for s in "$@"; do
  row ppo_s$s $PY -m experiments.rlpolicy.run_staged --policy seeds_s$s@ppo --tok3 t1
  row clone_s$s $PY -m experiments.rlpolicy.run_staged --policy seeds_s$s@clone --tok3 t1
  RUNS="$RUNS ppo_s$s=$T/ppo_s$s clone_s$s=$T/clone_s$s"
done
rm -f $Q/TIMING_HOLD
trap - EXIT
log "timing done, scoring"
export CUDA_VISIBLE_DEVICES=
$PY -m experiments.rlpolicy.seeds_eval --seeds "$@" --real $RUNS --out $T/seeds_real.md --workers 20 >> $T/score.log 2>&1
log "done"; touch $T/TIMING_DONE
