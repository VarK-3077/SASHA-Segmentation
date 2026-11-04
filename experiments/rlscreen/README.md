# RL screening

Can an RL agent do what the supervised low-res screen does (`experiments/lowsep`: level-3 screen
of every unit, then the level-2 token head on the units it flags) as well or faster, deciding by
itself how much to look at? The unit is the segmenter's own 2048 px crop.

Two designs share one environment (`env.py`), played in batched rounds so reads and GPU work
stay batched:

- **coarse**: a cheap global pass at level 5 or 6 for every unit; each round the agent picks units
  to screen at level 3 (revealing their level-3 view), units to zoom (segment), or stops.
- **full**: the whole slide is screened at level 3 first (the supervised pipeline's input); each
  round the agent picks units to zoom or stops. This isolates whether RL can match a supervised
  head on the same information, and where it gains (lesion following, per-slide stopping).

A zoomed unit reveals its outcome: tumour fraction and how much tumour touches each border, the
signal for tracing a lesion into its neighbours. Training uses the ideal segmenter (the outcome
is ground truth); `run_real.py` uses the real token head. Quality J is tumour area recall, non-ITC
lesion recall, or their mean. Reward per round: S^J(t-1) (J(t) - J(t-1) - lam * cost), cost = the
round's A100 seconds (`costs.py`) over the seconds of segmenting the whole slide, so the penalty
is a price in hardware time, not a dataset constant. lam is drawn per episode and is an input
of the policy, so one policy covers every speed / quality point and each slide's stopping round
is its own decision; the final checkpoint is used, val is only logged.

Training (`train.py`): behaviour cloning of a hand-written teacher (`teacher.py`, also the
supervised baseline: screen / zoom where the supervised heads say tumour, then follow tumour
across borders), with DAgger relabelling, then PPO with per-unit clipped surrogates and a KL
penalty to the frozen cloned policy. Ablations in `jobs_wave1.txt`: global level 5 vs 6, light
state (head probabilities only), S = 1, area-only quality, no KL, no cloning at all. Wave 2
repeats the main ones with the Kaiko ViT-S/16 screen (`LOWSEP_BACKBONE=kaiko_vits16`).

## Run order

```
bash experiments/rlscreen/prep_chain.sh cuda:0          # box holding /data2/venkatavks/lowsep: tables, baselines, ViT-S tokens + heads
bash experiments/rlscreen/queue.sh experiments/rlscreen/jobs_wave1.txt cuda:0 3   # train + offline eval, 3 jobs at a time
bash experiments/rlscreen/real_chain.sh RUNS.txt        # chosen configurations with the real segmenter, timed alone
```
Outputs under `/data2/venkatavks/rlscreen/` (`units_<backbone>/`, `runs/<tag>/`, `offline_*.md`,
`real/`). `real_report.py` expects the authoritative level-2 scores at
`/data2/venkatavks/rlscreen/ref_l2_clean_test.json` (copy of
`outputs/pfa_scannet/whole_set_dice/tokhead_l2/clean_test.json`).
