# 24-hour autonomous run, 2026-10-18 01:30 IST to 2026-10-19 night

Audit log. Every experiment launched in this window is listed with its command, config,
log path and result. Mistakes and corrections are recorded as they happen, not cleaned up.
Code changes are committed to branch `worktree-rlbin` as they are made.

## Starting state

- RL binning at L3 (`experiments/rlbin`): three runs done (l3, l3b, l3c). Consistent test
  result on 49 tumour slides: Dice 0.56, oracle routing 0.78. Agent zooms core tiles or marks
  them core, zooms boundary tiles 17-48 % and small tiles 18-29 %. Reports:
  `/data2/venkatavks/rlbin/report_l3*.md`.
- Small-vs-normal study (agent worktree, `experiments/small`): core rows done, tail rows
  (stain, children, context, TTA, fine-tune) in flight on vivi, `/data2/venkatavks/small/gpu3.log`.
- Server at 01:30: both GPUs idle, /data2 97 GB free (small study holds 90 GB of token h5s,
  deletable), another session runs CPU-only pfa_scannet jobs.

## Plan

1. Diagnostic on L3: ground-truth bucket in the agent's state (`--state oracle`), and
   predicted bucket probabilities (`--state pred`). Separates perception from optimisation.
2. Three seeds of config l3c for error bars.
3. Kaiko ViT-L embeddings for every 1024 px tile (`extract_l2.py`), GPU 1, read-bound.
4. Branch on 1: L2 agent run if perception-limited; training-signal changes on L3 if not.
5. Frontier plot (quality vs zoom fraction) at L2 with the new low-res head.

## Timeline (server clock, IST; entries before 01:38 were first written with guessed times and corrected against `date`)

- 01:35 wrote `extract_l2.py`; launched on cuda:1, 24 workers. Log `/data2/venkatavks/rlbin/extract_l2.log`.
  Throughput 390 crops/s (no contention this time), so ~1 h for 1.26 M crops.
- 01:33 state features: `--state none|pred|oracle` adds 4 per-tile dims (zeros / predicted bucket
  probs / one-hot GT bucket). Side effect: the policy input grew from 388 to 392, so the l3/l3b/l3c
  checkpoints no longer load with the new code; their reports are already written.
- 01:34 `bucket_head.py` (ViT-S L3 embedding -> 4 buckets, balanced CE): val macro-recall 0.654,
  test recall normal 0.77 / core 0.75 / boundary 0.43 / small 0.62. Probabilities under
  `/data2/venkatavks/rlbin/pbucket/`. Written beside the cache, not into it, so the running
  extraction never reads a half-written cache file.
- 01:34 launched five L3 runs on cuda:0, config = l3c (lr 1e-4, ent 0.02, 16 episodes, balanced):
  `l3_oracle` (--state oracle), `l3_pred` (--state pred), `l3c_s1/s2/s3` (--state none, seeds 1-3).
  Each is followed by evaluate with criterion balanced and best. Logs `run_<tag>.log`,
  args `args_<tag>.json` in `/data2/venkatavks/rlbin/`.
  MISTAKE: the first launcher loop had no `&`, so only `l3_oracle` started and the loop sat
  waiting on it. Killed the loop (pid 2456260, the oracle job itself was setsid'd and survived),
  relaunched the other four at 01:36. Consequence: l3_oracle started ~10 min before the others
  and was launched from code commit a7cef54; its evaluate step imports the current code, which
  loads a7cef54 checkpoints unchanged (same layer shapes).
- 01:36 `cache_l2.py` written: regroups the L3 cache's 16 sub-patches into 4 per L2 child
  (asserted against the sep L2 fractions and coordinates), trains a Kaiko-L -> 4 sub-patch
  low-res head, writes `/data2/venkatavks/rlbin/cache_l2/`. Policy/env made level-agnostic
  (embedding dim and sub-patch count read from the cache).
- 01:37 `frontier.py` run at L3 (`/data2/venkatavks/rlbin/frontier_L3.{json,png}`). Tumour-slide
  Dice at zoom 0.02 / 0.1 / 0.5: random 0.02/0.19/0.47, uncertainty 0.34/0.55/0.67, greedy
  0.40/0.59/0.68, supervised router 0.49/0.62/0.68, ground-truth order 0.53/0.66/0.72; oracle
  routing (core marked, boundary+small zoomed) 0.776 at 0.02. The RL points (l3b, l3c) lie on
  the supervised-router curve at their zoom fraction (~0.08, Dice 0.56) on tumour slides, and
  far above every curve on all slides (0.79-0.83 vs <=0.75) because they leave normal slides
  empty. First bug: boolean negation in the GT-order lexsort, fixed (c4c725d).
  Reading: marking core without zooming is what lifts oracle routing above the GT-order curve
  (0.776 vs 0.53 at 2 %), so the core action is worth a lot at low zoom fractions, and no
  learned run uses it on the right tiles.
- 01:38 five L3 runs at 10-15 s per update with all five on cuda:0 (29 of 32 GB used, so no
  sixth process goes there). Expected finish ~03:30.
- 01:39 `rjob_l2.sh` launched detached on cuda:1: waits for the extraction, runs `cache_l2.py`
  (asserts on the sub-patch regrouping, trains the Kaiko-L low-res head), then two L2 runs
  `l2_none` and `l2_oracle` (8 episodes/update, 400 updates, otherwise l3c config), evaluates
  both, draws `frontier_L2`. Log `/data2/venkatavks/rlbin/run_l2_chain.log`.
