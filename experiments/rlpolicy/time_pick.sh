#!/bin/bash
# Real timing on vivi GPU 1 of the supervised rule (re-timed in this window) and one rlpolicy policy, one run at a
# time, cold page cache per slide, as rlcascade/gate_timing2.sh does it: waits for an idle vivi on 3 checks 2 min
# apart (no ARL/Lean process, load <= 4, every GPU <= 1.5 GB used, no TIMING_HOLD, no RLPOLICY_YIELD), holds
# TIMING_HOLD during the runs, never signals other processes, then scores the real maps into timing.md.
# Usage: time_pick.sh NAME TOK3TAG <run_staged arguments: --theta FILE.json | --policy SEARCH@RUN [--tok6 TAG]>
source /var/tmp/venkatavks/rlpolicy/env_vivi_rlp.sh
export RLSCREEN_ROOT=/var/tmp/venkatavks/rlcascade/oofroot
Q=/var/tmp/venkatavks/queue
NAME=$1; TAG=$2; shift 2
T=$R/timing/$NAME
mkdir -p $T
log() { echo "$(date '+%F %T') $*" >> $T/gate.log; }
arl() { for d in /proc/[0-9]*; do c=$(tr '\0' ' ' < $d/cmdline 2>/dev/null); w=$(readlink $d/cwd 2>/dev/null)
  case "$c$w" in *time_pick*) ;; *common.verifier_server*|*common.trainer*|*vllm*|*VLLM*|*/data3/venkatavks/arl_*) echo x; return ;; esac; done; }
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
echo "rlpolicy timing $NAME $(date)" > $Q/TIMING_HOLD
trap 'rm -f $Q/TIMING_HOLD' EXIT
export CUDA_VISIBLE_DEVICES=1
if [ ! -f $T/rule/DONE ]; then log "start rule, $(uptime)"; $PY -m experiments.rlcascade.run_real_rl --model rule --out_dir $T/rule < /dev/null > $T/rule.log 2>&1 || log "rule FAILED"; log "end rule, $(uptime)"; fi
if [ ! -f $T/pick/DONE ]; then log "start pick, $(uptime)"; $PY -m experiments.rlpolicy.run_staged "$@" --tok3 $TAG --out_dir $T/pick < /dev/null > $T/pick.log 2>&1 || log "pick FAILED"; log "end pick, $(uptime)"; fi
rm -f $Q/TIMING_HOLD
trap - EXIT
log "timing done, scoring"
export CUDA_VISIBLE_DEVICES=
$PY -m experiments.rlcascade.score_real --runs rule=$T/rule $NAME=$T/pick --out $T/timing.md >> $T/gate.log 2>&1
log "done"; touch $T/TIMING_DONE
