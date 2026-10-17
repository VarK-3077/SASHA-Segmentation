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
- 01:50 small-vs-normal study (agent worktree `agent-a35f15a848390276d`, `experiments/small`,
  commits 1822911 + 653b13b) reported complete: 36 methods on the same 24,049 normal + 892 small
  test crops; report `/data2/venkatavks/small/report.md`. Best: Kaiko-L patch tokens of the four
  level-1 children per 1024 px crop, per-token MLP, max pooling: small recall 0.735 @99 % spec,
  0.872 @95 %, AUC 0.962 (pooled Kaiko-L baseline 0.576 / 0.725 / 0.917). Levers in order:
  visible tumour area (sets the floor; <1,000 µm² crops stay at 0.2-0.55 for every method),
  token-level scoring (+0.07-0.09), finer read of the same footprint (+0.09 with tokens),
  backbone size (+0.15 S->g); training tricks, context, TTA ≤ +0.02; Macenko stain −0.08;
  light fine-tuning does not beat frozen tokens. Its 90 GB of token h5s under
  `/data2/venkatavks/small/feats/` are deletable if space gets tight.
- 02:12 extraction finished (1.26 M crops, 477 crops/s). `cache_l2.py` ran clean: both
  alignment asserts (child sub-patch fractions average to the L2 fraction; child coordinates
  equal the sep L2 coordinates) passed on all 383 slides. Kaiko-L -> 4 sub-patch low-res head:
  best val BCE 0.0260 at epoch 3 (ViT-S L3 head was 0.0335 on 16 sub-patches; not the same
  granularity, so only roughly comparable).
- 02:14 `l2_none` and `l2_oracle` launched on cuda:1 by the chain.
- 02:26 MISTAKE: `l2_oracle` died with CUDA out-of-memory in the PPO update (two runs at
  15-16 GB each on a 32 GB card). Cause: the update replays fixed 96-step chunks, and at level 2
  a slide has up to 14.7k tiles, so one chunk's activations are ~15x the level-3 size. Fix in
  `ppo.py`: chunk length = 96,000 // tiles (min 4), so every forward pass sees the same number
  of tile rows regardless of slide size; `expandable_segments` set for the allocator. Both L2
  runs killed and restarted from scratch at 02:30 with their old logs, jsonl, args and
  checkpoints removed first, so the balanced checkpoint pick cannot read stale rows. The five
  L3 runs on cuda:0 were untouched (they imported the old ppo.py before the fix and run fine
  at level-3 sizes).
- 03:19 `l3_oracle` test (balanced pick = update 480, val tumour 0.812 / clean 1.000): Dice all
  0.859, tumour 0.678, normal clean 0.953, zoom 0.033; boundary zoomed 0.37, small zoomed 0.42,
  core marked 0.71. Oracle routing: 0.916 / 0.776 / 0.988 at zoom 0.02. Reading: with the
  ground-truth bucket in the state the agent gains +0.11 tumour Dice over the plain runs
  (0.565) but still zooms under half of the boundary/small tiles, so the remaining 0.10 to the
  lookup policy is optimisation, not perception. Val-to-test gap on tumour Dice (0.81 -> 0.68)
  is large: 9 val tumour slides.
- 03:20 `bc.py`: behaviour cloning from the oracle routing (states along random oracle
  roll-outs, loss = -log of the policy mass on the allowed set), 6 epochs, ~20 s each. Val after
  BC: state oracle Dice-all 0.908 / tumour 0.745 / zoom 0.03 (boundary 0.98, small 1.00 zoomed,
  but core marked only 0.03: it stops before marking cores, PPO must fix that); state pred
  0.817 / 0.602; state none 0.809 / 0.581. Checkpoints `ckpt/bc_l3_{oracle,pred,none}.pt`,
  logs `bc_l3_*.log`.
- 03:21 launched `l3_bc_oracle`, `l3_bc_pred`, `l3_bc_none` on cuda:1 (l3c config, `--init`
  from the BC checkpoints) next to the two L2 runs; GPU 1 memory 10 GB + 3 x ~6 GB.
