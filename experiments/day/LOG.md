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

## Timeline

- 01:35 wrote `extract_l2.py`; launching on cuda:1, 24 workers.
