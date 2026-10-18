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
- 03:34 all five L3 diagnostics scored (balanced pick; Dice all / tumour / normal clean / zoom):
  l3_oracle 0.859 / 0.678 / 0.953 / 0.03; l3_pred 0.857 / 0.626 / 0.976 / 0.09;
  l3c_s1 0.851 / 0.607 / 0.976 / 0.03; l3c_s2 0.814 / 0.613 / 0.918 / 0.05;
  l3c_s3 0.808 / 0.527 / 0.953 / 0.05; with l3c (seed 0) 0.828 / 0.565: plain state over four
  seeds = tumour 0.578 ± 0.034, all 0.825 ± 0.017. Predicted buckets +0.05 tumour Dice (about
  1.4 sd), ground-truth buckets +0.10. Boundary zoom rate stays 0.25-0.37 in every run.
- 03:35 level-2 bucket head (Kaiko-L 1024-d -> 4 buckets): val macro-recall 0.666, test recall
  normal 0.96 / core 0.76 / boundary 0.62 / small 0.40; probabilities under `pbucket_l2/`.
- 03:36 BC at level 2, 6 epochs (10 s each): val tumour Dice oracle 0.669, pred 0.759, none
  0.763, mostly by marking cores (zoom 0.00-0.01). Loss still falling at epoch 6 at this size,
  so re-run with 20 epochs before using the checkpoints as PPO warm starts.
- 03:38 three level-2 PPO runs from the 20-epoch BC checkpoints launched on cuda:0:
  `l2_bc_oracle`, `l2_bc_pred`, `l2_bc_none` (8 episodes/update, 200 updates, l3c config).
  20-epoch BC val tumour Dice: oracle 0.819 (boundary 0.96 / small 1.00 zoomed), pred 0.712,
  none 0.741.
- 04:00 PPO at lr 1e-4 / entropy 0.02 moves away from the BC warm starts: `l3_bc_oracle` val
  tumour 0.745 at init -> 0.547 at update 250 with boundary zoom 0.14; `l3_bc_none` collapsed
  to 0.30; `l2_bc_oracle` collapsed to marking everything core by update 10. Likely causes:
  the entropy bonus (0.02 over a 2N+1 action space) penalises the near-deterministic cloned
  policy, and lr 1e-4 is 10x SASHA's. The balanced checkpoint pick will therefore fall on an
  early update, which makes those rows effectively a test of the BC policy itself.
  `rjob_ft.sh` queued: the same three L3 runs fine-tuned at lr 1e-5 / entropy 0.001 (SASHA's
  values), tags `l3_ft_*`, starting when GPU 1 drops below 12 GB used.
- 04:27 level-2 plain runs scored (balanced pick): `l2_none` 0.822 / 0.479 / clean 1.000 /
  zoom 0.026; `l2_oracle` 0.780 / 0.400 / 0.976 / 0.023. Both below the L3 plain runs on tumour
  slides: 4x more candidates, 400 updates of 8 episodes, and the same risk-averse basin.
  Level-2 baselines on the same scorers: low-res mask only (Kaiko-L head, no zoom) 0.852 all /
  0.588 tumour; supervised router at zoom 0.03: 0.775 / 0.704; zoom 0.10: 0.772 / 0.719;
  oracle routing 0.921 / 0.792 at zoom 0.012. `frontier_L2.{json,png}` written by the chain.
  Reading: the level-2 low-res head alone beats every RL agent so far on tumour slides with
  zero zooms, and the environment never lets the agent use that mask (untouched = negative).
  Decision: add an environment mode where untouched tiles keep the low-res mask and the agent
  has an explicit normal action to suppress it (zoom / core / normal / stop), which is the
  deployment semantics and the action space originally asked for. Existing mode kept as is.
- 04:36 `--mode lr` implemented (env, policy with 3 per-tile actions, metrics, baselines, BC,
  frontier). Status one-hot width depends on the mode so every 'neg' checkpoint, including the
  ones still training, loads unchanged with the new code. Smoke test (2 updates) and an
  end-to-end evaluate pass at L3 in lr mode. Baselines in lr mode at L3: low-res mask only
  0.688 / 0.314, supervised router at zoom 0.10: 0.742 / 0.631, oracle routing (now also
  marking normal tiles normal) 0.916 / 0.776.
- 04:22 `rjob_ft.sh` fired: GPU 1 dropped below 12 GB when the two L2 chain runs ended, so the
  three `l3_ft_*` fine-tunes started while the three `l3_bc_*` runs were still training there
  (six processes on cuda:1, no OOM so far). Not what the gate was meant to wait for; harmless
  apart from slower updates.
- 05:00 scored (balanced pick; Dice all / tumour / clean / zoom; boundary / small zoomed):
  `l3_bc_oracle` (BC then PPO at lr 1e-4) 0.862 / 0.664 / 0.965 / 0.05; 0.15 / 0.28 — PPO at
  lr 1e-4 moved it off the cloned routing (val tumour 0.745 at init -> pick at update 510).
  `l3_bc_pred` 0.804 / 0.494; `l3_bc_none` 0.831 / 0.617 (pick at update 90, i.e. close to BC).
  **`l3_ft_oracle` (BC then PPO at lr 1e-5 / entropy 0.001): 0.900 / 0.751 / 0.976 / 0.028;
  boundary zoomed 0.95, small zoomed 0.97, core zoomed 0.42 + marked 0.42.** That is within
  0.02-0.03 of the oracle routing (0.916 / 0.776 / 0.988 at 0.02) with the same reads. So the
  PPO machinery reproduces and holds the lookup policy once it starts near it and is updated
  gently; the earlier failures were the optimiser settings, not the objective.
  In flight: `l3_ft_pred` val tumour 0.837 and `l3_ft_none` 0.754 at update 140 (zoom 0.27).
- 04:59 `rjob_ft2.sh` (generic clone-then-fine-tune): L3 lr-mode chain on cuda:1 cloned the three
  state variants (val tumour after BC: oracle 0.733, pred 0.606, none 0.570) and launched
  `l3lr_ft_{oracle,pred,none}` at 05:00. MISTAKE: the L2 chain was started in the same ssh
  command after a `setsid` that did not fork, so it never ran; relaunched on its own at 05:00
  (`run_ft2_l2.log`); it reuses the existing `bc_l2_*` checkpoints and waits for GPU 0 to
  drop below 14 GB (the three `l2_bc_*` runs are still there).
- 05:42 `l2_bc_none` scored (balanced pick = update 10, i.e. essentially the 20-epoch BC policy
  on Kaiko-L level-2 embeddings, no bucket labels in the state): 0.889 / 0.699 / clean 0.988 /
  zoom 0.010; boundary zoomed 0.44, small 0.18, core marked 0.39. Best label-free tumour Dice so
  far, at 1 % of tiles zoomed (plain L3 runs: 0.58 at 4-8 %).
- 06:02 `l2_bc_oracle` scored (pick = update 0, i.e. the BC policy with ground-truth buckets in
  the state): 0.899 / 0.726 / 0.988 / zoom 0.016; boundary 0.94 / small 0.86 zoomed, core
  marked 0.52. The L2 fine-tune chain fired at 05:45 (GPU 0 under 14 GB); `l2_ft_*` at update
  80 of 200 by 06:02.
- 06:33 `l2lr_none` (L2, lr mode, plain PPO) scored: the balanced pick (update 40) takes no
  action at all, so the row equals the low-res mask alone: 0.852 / 0.588 / 0.988 / zoom 0.
  In lr mode the plain agent learns that the default mask is hard to improve on without risk.
  `l3_ft_pred` (BC then gentle PPO, predicted buckets, neg mode; pick = update 410): 0.797 /
  0.677 / clean 0.859 / zoom 0.051; boundary 0.66 / small 0.37 zoomed, core zoomed 0.64.
  Tumour Dice +0.10 over the plain runs, but 14 % of normal slides now carry false positives,
  so Dice-all is lower (0.797 vs 0.83).
- 06:38 `rjob_bcseeds.sh` on cuda:1: clone-only level-2 policies with seeds 1 and 2 for states
  none and pred, each scored after one gentle PPO update (tags `l2_bcs<seed>_<state>`), to put
  error bars on the `l2_bc_none` 0.699 row.
- 06:43 `l3_ft_none` (BC then gentle PPO, embedding-only state, neg mode; pick = update 470):
  0.807 / 0.660 / clean 0.882 / zoom 0.142; boundary 0.75 / small 0.63 zoomed, core zoomed
  0.66. Same pattern as `l3_ft_pred`: +0.08-0.10 tumour Dice over the plain runs, paid with
  more zooms and 12 % of normal slides no longer clean.
- 06:54 clone-only level-2 seeds (one gentle update, pick = update 0): state none seeds 1/2 ->
  0.888 / 0.696 and 0.883 / 0.726 (with seed 0: tumour 0.707 ± 0.013, Dice-all 0.887 ± 0.003,
  zoom 0.7-1.0 %); state pred seeds 1/2 -> 0.870 / 0.641 and 0.894 / 0.713 (seed 0 still
  training in `l2_bc_pred`). The clone-only level-2 policy is stable across seeds.
- 06:55 queued `l3_ft_none` PPO seeds 1 and 2 on cuda:1 (same BC checkpoint, different PPO
  seed) for error bars on the 0.660 row.
- 07:15 `l3lr_ft_oracle` (lr mode, BC then gentle PPO, GT buckets; pick = update 170): 0.887 /
  0.693 / 0.988 / zoom 0.020; boundary 0.81 / small 0.87 zoomed. Lower on tumour slides than
  its neg-mode twin (0.751): in lr mode the untouched boundary tiles keep a low-res mask that
  the agent then has to override, and the clone did not learn to mark enough cores (0.46).
- 07:35 `l3lr_ft_pred` (lr mode, BC then gentle PPO, predicted buckets; pick = update 10):
  0.808 / 0.552 / 0.941 / zoom 0.008. In lr mode the fine-tune did not improve on the clone,
  and the row is below the neg-mode twin (0.677).
- 07:38 `l3lr_ft_none` (lr mode, BC then gentle PPO, embedding-only; pick = update 570):
  0.826 / 0.625 / 0.929 / zoom 0.042; boundary 0.66 / small 0.35 zoomed. lr-mode fine-tunes
  land 0.03-0.06 below their neg-mode twins on tumour slides across all three states.
- 08:05 `l3_ft_none_s2` (PPO seed 2 from the same BC checkpoint; pick = update 230): 0.796 /
  0.652 / 0.871 / zoom 0.094 (seed 0: 0.807 / 0.660 / 0.882 / 0.142).
- 08:10 MISTAKE, bigger one: at 04:59 the two chains were started in one ssh command; the first
  `setsid` did not fork, so the L2 chain was not skipped but queued behind the L3 lr chain. I
  relaunched the L2 chain by hand at 05:00, and when the L3 lr chain finished (~07:40) the queued
  copy fired too: a second `rjob_ft2.sh ... 2 neg` passed the memory gate and launched three
  more `l2_ft_*` trainings with the same tags, truncating the running jobs' logs (binary
  garbage), interleaving their jsonl rows and overwriting their checkpoints. Nothing from
  `l2_ft_*` is trustworthy. Action: killed both sets and both chains, deleted every `l2_ft_*`
  log / jsonl / args / checkpoint / report, relaunched the L2 chain once (`run_ft2_l2.log`;
  the first attempt's chain log kept as `run_ft2_l2_attempt1.log`). Cost: ~2 h of GPU 0.
  Rule from now on: one chain per ssh command, and tags must never be reused.
- 08:22 `l3lr_none` (L3, lr mode, plain PPO; pick = update 530): 0.758 / 0.518 / 0.882 / zoom
  0.009, almost no actions (boundary zoomed 0.08). Below the lr-mode fine-tunes.
- 08:43 `l2_ft_oracle` (level 2, BC then gentle PPO, GT buckets; pick = update 170): 0.899 /
  0.749 / 0.976 / zoom 0.051; boundary 0.96 / small 0.95 zoomed, core marked 0.53 (oracle
  routing at L2: 0.921 / 0.792 / 0.988 / 0.012). `l3_ft_none_s1` (pick = update 260): 0.770 /
  0.688 / 0.812 / zoom 0.280. The three `l3_ft_none` seeds: tumour 0.667 ± 0.015 but zoom
  0.09-0.28 and normal-clean 0.81-0.88, i.e. the fine-tune's operating point is not stable
  across seeds even when its tumour Dice is.
- 04:37 launched `l3lr_none` (L3, lr mode, l3c config, 600 updates) and `l2lr_none` (L2, lr
  mode, 8 episodes, 400 updates) on cuda:0.
