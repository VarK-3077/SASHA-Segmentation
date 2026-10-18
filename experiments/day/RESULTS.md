# 24-hour run: results

Everything is on the 129 official CAMELYON16 test slides (49 tumour). Dice is over 512 px
sub-patches of the composed mask, so it is not comparable with pixel-level Dice elsewhere.
"zoom" is the fraction of a slide's tiles sent to the high-res scorer. Checkpoints are picked on
the 25 val slides (9 tumour) by 0.5·tumour Dice + 0.5·normal-slides-clean, then scored once on
test. Per-run commands, timestamps and mistakes are in `LOG.md`; raw reports in
`/data2/venkatavks/rlbin/report_<tag>_balanced.md`, summary table in `summary.md` there,
frontier plots `frontier_L3.png`, `frontier_L3_lr.png`, `frontier_L2.png`, `frontier_L2_lr.png`.

## 1. What limits the plain RL agent: half perception, half optimisation

Level 3 (2048 px tiles, ViT-S DINO embeddings), the environment from before this window
(untouched tiles count as negative), same PPO config, only the state changes:

| state given to the agent | Dice all | Dice tumour | normal slides clean | zoom |
|---|---|---|---|---|
| embedding only (4 seeds) | 0.825 ± 0.017 | 0.578 ± 0.034 | 0.95 | 0.04-0.08 |
| + predicted bucket probabilities | 0.857 | 0.626 | 0.976 | 0.085 |
| + ground-truth bucket (one-hot) | 0.859 | 0.678 | 0.953 | 0.033 |
| oracle routing (lookup on the labels) | 0.916 | 0.776 | 0.988 | 0.020 |

With the labels in its state the agent gains 0.10 but still zooms under half of the boundary
and small tiles; the remaining 0.10 to the lookup is PPO not finding a policy it can represent.

## 2. The optimiser was the problem, not the objective

Behaviour-clone the oracle routing (6 epochs, ~2 min), then PPO. Fine-tuning at the earlier
settings (lr 1e-4, entropy 0.02) walks away from the clone; at SASHA's values (lr 1e-5,
entropy 0.001) it stays and improves:

| level 3, state | clone only (val) | + PPO lr 1e-4 | + PPO lr 1e-5 (tumour Dice / clean / zoom) |
|---|---|---|---|
| ground-truth bucket | 0.745 | 0.664 | **0.751 / 0.976 / 0.028** (oracle routing: 0.776 / 0.988 / 0.020) |
| predicted bucket | 0.602 | 0.494 | 0.677 / 0.859 / 0.051 |
| embedding only | 0.581 | 0.617 | 0.660 / 0.882 / 0.142 (seed 2: 0.652 / 0.871 / 0.094) |

The machinery reproduces the lookup policy to within 0.03 when it can see the buckets. Without
labels the same recipe gives +0.08-0.10 tumour Dice over plain PPO, paid with 12-14 % of normal
slides no longer clean.

## 3. Level 2 with Kaiko ViT-L: the low-res head alone beats every level-3 agent

| method (level 2, 1024 px tiles) | Dice all | Dice tumour | normal clean | zoom |
|---|---|---|---|---|
| low-res head mask only, no zoom | 0.852 | 0.588 | 0.988 | 0 |
| supervised router (core ≥ 0.9, zoom most uncertain) | 0.775 | 0.704 | 0.812 | 0.03 |
| plain PPO, embedding only | 0.822 | 0.479 | 1.000 | 0.026 |
| plain PPO, ground-truth bucket | 0.780 | 0.400 | 0.976 | 0.023 |
| clone only, embedding only (3 seeds) | 0.887 ± 0.003 | 0.707 ± 0.013 | 0.98 | 0.007-0.010 |
| clone only, predicted bucket (3 seeds) | 0.880 | 0.679 ± 0.030 | 0.98 | 0.008 |
| clone only, ground-truth bucket | 0.899 | 0.726 | 0.988 | 0.016 |
| clone + PPO lr 1e-5, embedding only (3 seeds) | 0.865 ± 0.047 | 0.694 ± 0.042 | 0.95 | 0.05-0.17 |
| clone + PPO lr 1e-5, predicted bucket | 0.853 | 0.684 | 0.941 | 0.188 |
| clone + PPO lr 1e-5, ground-truth bucket | 0.899 | 0.749 | 0.976 | 0.051 |
| supervised router in deployment mode (untouched = low-res mask) | 0.800 | 0.733 | - | 0.01 |
| oracle routing | 0.921 | 0.792 | 0.988 | 0.012 |

The clone-only level-2 policy is the best label-free result at 1 % of tiles: tumour Dice 0.71,
normal slides 98 % clean, stable across seeds. Gentle PPO on top of it reaches 0.720 and 0.728 at 5-7 %
in two seeds but 0.635 at 17 % in a third, so it adds nothing reliable; with predicted buckets it only adds zooms. Plain PPO at level 2 is
worse than at level 3 (4x more candidates, same budget of updates, same risk-averse basin).
For tumour slides alone the simple router in deployment mode is as good (0.733 at 1 %), but it
cannot keep normal slides clean, which is where every agent wins.

## 4. The deployment semantics (untouched tiles keep the low-res mask) do not help the agent

New environment mode `lr`: untouched tiles keep the low-res head's mask and the agent has an
explicit `normal` action to suppress it. Level 3:

| level 3, lr mode | Dice all | Dice tumour | normal clean | zoom |
|---|---|---|---|---|
| plain PPO, embedding only | 0.758 | 0.518 | 0.882 | 0.009 |
| clone + PPO lr 1e-5, embedding only | 0.826 | 0.625 | 0.929 | 0.042 |
| clone + PPO lr 1e-5, predicted bucket | 0.808 | 0.552 | 0.941 | 0.008 |
| clone + PPO lr 1e-5, ground-truth bucket | 0.887 | 0.693 | 0.988 | 0.020 |

Every lr-mode row is 0.03-0.06 below its twin in the original mode on tumour slides: the agent
inherits the low-res mask's errors on boundary tiles and has to spend actions undoing them. At
level 2 the plain lr-mode agent learned to do nothing at all (its row equals the low-res mask).

## 5. Frontier

`frontier_L3.png` / `frontier_L2.png`: Dice against zoom fraction for random, greedy,
uncertainty, supervised-router and ground-truth visit orders on the same scorers, with every RL
run as a point. On tumour slides the plain agents sit on the supervised-router curve; the
clone-then-fine-tune agents sit on or above the ground-truth-order curve; the oracle routing
point (core marked without zooming) is above every curve, which is the value of the core
action. On all slides every agent is far above every curve, because leaving normal slides
empty is something the heuristics cannot do.

## 6. What this says for the project

- The binning question is answered at the ceiling: a PPO policy reproduces the oracle routing
  (0.751 vs 0.776 tumour Dice at the same 2-3 % of reads) once it starts from a clone and is
  updated gently. The remaining gap to the lookup is 0.03.
- Without labels the realistic ceiling in this setup is the clone of the supervised router on
  Kaiko ViT-L level-2 embeddings: 0.71 tumour Dice at 1 % of reads. PPO on top adds at most a
  few points on tumour slides and costs normal-slide cleanliness.
- The lever that moved tumour Dice most was not RL: level 2 and a pathology backbone moved the
  no-zoom baseline from 0.31 to 0.59 and the router from 0.63 to 0.70.
- Core marking without zooming is worth as much as the zooms themselves at low budgets.

## Caveats

- Scorers: the high-res scorer is the ViT-S sub-patch head from the stopping study at both
  levels; the level-2 low-res head is a fresh Kaiko-L MLP. Dice at 512 px sub-patch resolution.
- Checkpoint selection on 9 val tumour slides; val-to-test gaps of 0.1 on tumour Dice are
  common. The balanced criterion is reported throughout; the best-val-Dice criterion often picks
  a worse policy (e.g. l3_pred 0.492 vs 0.626).
- Single seeds unless stated. Plain L3 has 4 seeds, clone-only L2 has 3, L3 clone+fine-tune
  embedding-only has 2.
- The `oracle routing` row and the `ground-truth bucket` states use test labels and are
  ceilings, not methods.
- No per-read cost in the reward; the only constraint is the 50 % zoom cap, which never binds.
