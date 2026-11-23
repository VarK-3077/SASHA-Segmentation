#!/bin/bash
# Times the teacher row of the seed-study window (time_seeds.sh) after its first attempt failed, straight after that
# window ends: waits for time_seeds to release TIMING_HOLD and for its scoring to exit, takes TIMING_HOLD at once (so the
# next queued window cannot start in between), times the teacher with the same harness, then re-scores the whole window
# into timing/seeds/seeds_real.md. Never signals other processes.
source /var/tmp/venkatavks/rlpolicy/env_vivi_rlp.sh
export RLSCREEN_ROOT=/var/tmp/venkatavks/rlcascade/oofroot
Q=/var/tmp/venkatavks/queue
T=$R/timing/seeds
log() { echo "$(date '+%F %T') $*" >> $T/gate.log; }
until grep -q "timing done, scoring" $T/gate.log 2>/dev/null && [ ! -f $Q/TIMING_HOLD ]; do sleep 5; done
while ps -u $USER -o cmd= | grep -q "[s]eeds_eval --seeds"; do sleep 5; done
echo "rlpolicy timing seeds teacher rerun (nov23-presentation) $(date)" > $Q/TIMING_HOLD
trap 'rm -f $Q/TIMING_HOLD' EXIT
sleep 30
log "teacher rerun: load $(cut -d' ' -f1 /proc/loadavg)"
export CUDA_VISIBLE_DEVICES=1
rm -rf $T/teacher
log "start teacher (rerun), $(uptime)"
$PY -m experiments.rlpolicy.run_staged --theta $R/timing/theta_top16_k05.json --tok3 t1 --out_dir $T/teacher < /dev/null > $T/teacher.log 2>&1 || log "teacher FAILED"
log "end teacher (rerun), $(uptime)"
rm -f $Q/TIMING_HOLD
trap - EXIT
export CUDA_VISIBLE_DEVICES=
RUNS="rule=$T/rule teacher=$T/teacher"
for s in 0 1 2 3 4; do RUNS="$RUNS ppo_s$s=$T/ppo_s$s clone_s$s=$T/clone_s$s"; done
$PY -m experiments.rlpolicy.seeds_eval --seeds 0 1 2 3 4 --real $RUNS --out $T/seeds_real.md --workers 20 >> $T/score.log 2>&1 && log "rescored" || log "rescore FAILED"
touch $T/TEACHER_RERUN_DONE
