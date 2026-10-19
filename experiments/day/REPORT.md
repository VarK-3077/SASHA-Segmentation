# Status report: RL-guided segmentation on top of SASHA

October 2026. Written for someone who knows SASHA and has not followed this project.

## 1. The problem and the narrow question we chose

SASHA classifies a whole-slide image by reading every tile at low resolution, letting an RL
agent pick 10-20 % of the tiles to read at high resolution, and feeding both to a MIL head. The
goal here is different: a segmentation mask for the whole slide, as fast as possible. The idea
is the same shape as SASHA, a cheap low-resolution pass over everything and an agent that
decides which tiles deserve an expensive high-resolution pass, but the output is a mask and
the data has pixel-level labels (CAMELYON16 lesion polygons, and the eventual multiphoton
data will have them too).

Rather than build the whole pipeline, the work was narrowed to one question: **can a WSI be
divided into the regions a segmenter needs to treat differently, and can an RL policy learn
to do the dividing?** The regions are four: normal tissue (leave it), tumour core (mark it
positive without a closer look), lesion boundary (needs high resolution to get the edge
right) and small lesion (needs high resolution to be found at all). The order of importance
is boundary, then small, then core.

Everything below is on CAMELYON16 with the repo's split 4: 242 train, 27 val and the 129
official test slides. Numbers are always test unless marked val.

## 2. The substrate every study shares

- **Tiles and levels.** SASHA's tiles are 2048 px at level 0, read as 256 px at pyramid level
  3. We also use "level 2" tiles, 1024 px read as 256 px at level 2, i.e. four children per
  SASHA tile. Every crop fed to a backbone is 256 px at its own level, so per-crop compute is
  the same at every level and only the number of crops changes.
- **Labels.** The lesion XMLs are rasterised once per slide at level 4 (3.6 µm per pixel),
  connected components give each lesion an area and a major axis, and every crop at every
  level is a block of that raster: its tumour fraction, the lesions it touches, and its
  bucket. Core is fraction at least 0.9; boundary is a partial crop touching a lesion larger
  than the crop; small is a partial crop whose lesions all fit inside it. The 20 training
  tumour slides with known non-exhaustive annotations never contribute normal crops.
- **Scorers.** A "low-res head" predicts the sub-patch labels of a tile from its low-res
  embedding; a "high-res head" predicts each 512 px sub-patch from its own level-1 embedding
  plus the tile embedding. Both are small MLPs trained fresh on the train split (BCE, early
  stop on val). At level 3 both use SASHA's ViT-S/16 DINO embeddings; at level 2 the low-res
  head uses Kaiko ViT-L/14 embeddings and the high-res head is the same ViT-S one.
- **Mask quality.** Dice between the composed mask and ground truth at 512 px sub-patch
  resolution (16 sub-patches per SASHA tile). This is coarser than pixel Dice and not
  comparable with numbers elsewhere in the project; it is the same metric for every method in
  this report. A slide with no tumour scores 1 if the mask is empty and 0 otherwise, so
  "Dice all" over 129 slides rewards leaving normal slides alone, and "Dice tumour" over the
  49 tumour slides measures segmentation.

## 3. Can the buckets be told apart from a single crop? (separability study)

Stratified crop samples per level, frozen backbone embeddings, a linear probe and a small
MLP on the four buckets, model selection on val, one test pass. The headline metric is recall
of each tumour bucket at 99 % specificity on normal crops.

Two things decide separability: the backbone's pretraining and the crop size, in that order.

| 1024 px crops, MLP probe, recall at 99 % specificity | core | boundary | small |
|---|---|---|---|
| ViT-S/16 DINO, lunit (SASHA's backbone) | 0.95 | 0.71 | 0.36 |
| Kaiko ViT-S/16, same size, pathology recipe | 0.98 | 0.76 | 0.49 |
| Kaiko ViT-L/14 | 0.98 | 0.81 | 0.55 |
| Midnight-12k ViT-g (1.1 B) | 0.99 | 0.82 | 0.60 |
| DINOv2 ViT-S, natural images | - | 0.43 | 0.16 |
| ResNet-50 ImageNet | 0.79 | 0.32 | 0.11 |

Core is solved by everything pathological. Boundary is good. Small lesions are the weak
class, and their misses go to normal, not to boundary. Crop size peaks at 1024 px for every
backbone: 2048 px (SASHA's) loses about 0.15 on small lesions, 256 px loses on every bucket
because a 58 µm crop has no architectural context. Natural-image backbones are unusable.
Going from 22 M to 1.1 B parameters buys 0.11 on small lesions; going from natural-image to
pathology pretraining at 22 M buys 0.33. Licences: Midnight-12k is MIT, the Kaiko ladder and
Phikon-v2 are non-commercial, the gated models (UNI2, Virchow2, H-optimus-0) were not tested.

## 4. Why small lesions are hard (small-vs-normal study, run by a separate agent)

Half of the 892 "small" test crops are slivers: less than half of their lesion lies inside
the crop, the rest is in the neighbour. Recall is a monotone function of the tumour area
visible in the crop and of nothing else: 0.08 under 500 µm², 0.86 above 10,000 µm² for the
pooled Kaiko-L probe. The 20 % of the bucket with under 1,000 µm² visible, a handful of
cells, stays at 0.2-0.5 recall for every method and is not separable from the crop alone.

What moves the rest, in order of effect: scoring the backbone's patch tokens instead of its
pooled embedding (+0.07-0.10 recall, all of it on the low-area crops; crop labels suffice, no
masks needed, a linear token head does not work), reading the same footprint at a finer level
(+0.09 with tokens), backbone size (+0.15 from ViT-S to ViT-g). Loss, balancing, label
cleaning, context from the parent tile and TTA each move it by 0.02 or less; Macenko stain
normalisation hurts by 0.08; fine-tuning the last four blocks does not beat frozen tokens.
Best row: Kaiko-L tokens of the four level-1 children per 1024 px crop, per-token MLP, max
pooling: small recall 0.735 at 99 % specificity, 0.872 at 95 %, from 0.576 / 0.725 pooled.

Transferable statement: whatever backbone the final system can afford, keep its spatial
tokens and score them before pooling, and treat sub-1,000 µm² slivers as the neighbour's
problem, which argues for a selector that sees neighbourhoods.

## 5. Is there anything to gain from stopping per slide? (stopping ceiling)

Before any agent, a label computation. Play a fixed visit order to the end of each slide,
record Dice after every step, and define J = Dice - λ·steps. The per-slide oracle stop is the
argmax of J with the labels in hand; compare it with the best single budget and with a
"stop when uncertainty drops" rule, both chosen on train rollouts. On the 49 test tumour
slides with the uncertainty-plus-neighbour-follow-up order and λ = 1e-4 Dice per tile:

| stopper | J | Dice at stop | tiles visited |
|---|---|---|---|
| per-slide oracle | 0.655 | 0.669 | 14 % ± 21 % |
| best fixed budget | 0.557 | 0.633 | 75 % |
| uncertainty threshold | 0.566 | | |
| visit everything | 0.546 | 0.648 | 100 % |

The oracle stop point has a standard deviation larger than its mean, which is the variance a
fixed budget cannot serve. Visiting everything is worse than stopping, because the high-res
head adds false positives. Visit order matters less than stopping: at the oracle stop,
random 0.614, uncertainty 0.645, adaptive 0.655, ground-truth order 0.687. The user then set
stopping aside to focus on the binning question, so no stopper was trained.

## 6. The RL binning agent

### Environment

One slide is an episode. The state is every tile's low-res embedding, a summary of its
low-res scores (mean, max, entropy, neighbourhood mean), its status, the high-res result
once zoomed, and episode progress; the policy is a per-tile MLP with a pooled slide vector
fed back into each tile's logits, SASHA-style pointer selection over all tiles plus a stop
action. Actions: zoom a tile (its sub-patches take the high-res head's prediction), mark it
core (all positive), stop. Whatever is untouched at the end is negative. Reward is the change
in Dice of the composed mask after each action, so the return is the final Dice; no per-read
cost, only a 50 % zoom cap that never binds. PPO with SASHA's clipping. Checkpoints are
chosen on val by half tumour Dice plus half normal-slides-clean, then scored once on test.
Baselines on the same scorers: low-res mask alone, zoom everything, a supervised router (core
where the low-res mean score is at least 0.9, zoom the most uncertain tiles up to a fraction),
and the oracle routing, the lookup that zooms exactly the boundary and small tiles and marks
cores, which uses the labels and is the ceiling.

### What happened, in order

**Plain PPO at level 3 learns something real and stops there.** Four seeds: Dice all
0.825 ± 0.017, tumour 0.578 ± 0.034, zoom 4-8 %. On all slides this beats the supervised
router at the same zoom (0.75), entirely by leaving normal slides empty, which the heuristics
cannot do. On tumour slides it sits on the router curve and far below the oracle routing's
0.776 at 2 % zoom. It zooms only 20-50 % of boundary tiles and 16-30 % of small tiles, takes
the sure gains on core tiles, and avoids the uncertain ones. Training curves oscillated
between a cautious and a bold policy every few updates.

**The failure is half perception, half optimiser.** With the ground-truth bucket in its
state the same agent reaches 0.678 but still zooms under half of the boundary and small
tiles; with predicted buckets, 0.626. The remaining 0.10 to the lookup is PPO not finding a
policy it can represent.

**Behaviour cloning plus gentle PPO reproduces the lookup.** Cloning the oracle routing
takes two minutes. PPO from that start at the first settings (lr 1e-4, entropy 0.02) walks
away from the clone within 100 updates; at SASHA's own values (lr 1e-5, entropy 0.001) it
stays and improves: with ground-truth buckets 0.751 tumour Dice, 0.976 of normal slides
clean, 2.8 % zoom, against the lookup's 0.776 / 0.988 / 2.0 %. The machinery was never the
problem; the optimiser settings were.

**Level 2 with Kaiko ViT-L moves the whole picture.** The level-2 low-res head alone, zero
zooms, gets tumour Dice 0.588 and all-slides 0.852, better than every level-3 agent. Plain
PPO at level 2 is worse than at level 3 (4x more candidates, same updates, same cautious
basin: 0.479). The clone of the router on level-2 embeddings, no labels at test time, gets
0.707 ± 0.013 at 1 % zoom over three seeds with 98 % of normal slides clean. Gentle PPO on top
of it without a read cost is unstable (0.720, 0.728, 0.635 over three seeds, the bad one
drifting to 17 % zoom and false positives). With a per-zoom cost of 1e-4 Dice in the reward,
the knee of the stopping curve, two seeds give 0.745 and 0.733 at 2-3 % zoom, 98 % of normal
slides clean. That is the best label-free result.

**Deployment semantics hurt.** A second environment mode keeps the low-res mask on
untouched tiles and adds an explicit normal action, which is how a deployed system would
work and the action space originally proposed. Every variant is 0.03-0.06 worse on tumour
slides in that mode, because the agent inherits the low-res mask's errors on boundary tiles
and spends actions undoing them; the plain level-2 agent in that mode learns to do nothing.
The supervised router in that mode is as good as any agent on tumour slides (0.733 at 1 %
zoom) but cannot keep normal slides clean.

### The table

| method | Dice all | Dice tumour | normal slides clean | zoom |
|---|---|---|---|---|
| plain PPO, level 3, 4 seeds | 0.825 ± 0.017 | 0.578 ± 0.034 | 0.95 | 0.04-0.08 |
| plain PPO, level 3, ground-truth buckets in state | 0.859 | 0.678 | 0.953 | 0.033 |
| clone + gentle PPO, level 3, ground-truth buckets | 0.900 | 0.751 | 0.976 | 0.028 |
| oracle routing, level 3 | 0.916 | 0.776 | 0.988 | 0.020 |
| level-2 low-res head alone | 0.852 | 0.588 | 0.988 | 0 |
| supervised router, level 2, deployment mode | 0.800 | 0.733 | - | 0.01 |
| clone only, level 2, no labels, 3 seeds | 0.887 ± 0.003 | 0.707 ± 0.013 | 0.98 | 0.01 |
| clone + gentle PPO + read cost, level 2, no labels, 2 seeds | 0.900 ± 0.002 | 0.739 ± 0.006 | 0.98 | 0.02-0.03 |
| oracle routing, level 2 | 0.921 | 0.792 | 0.988 | 0.012 |

## 7. The frontier

Four plots (`frontier_L3.png`, `frontier_L2.png`, and the deployment-mode twins) show Dice
against the fraction of tiles zoomed for random, greedy, uncertainty, supervised-router and
ground-truth visit orders on the same scorers, with every RL run as a point. On tumour slides
the plain agents sit on the router curve and the clone-based agents on or above the
ground-truth-order curve; the oracle routing point is above every curve, and the gap between
it and the ground-truth order is the value of marking core tiles without zooming them. On all
slides every agent is far above every curve, because an empty mask on a normal slide is
something a visit order cannot produce.

## 8. Where we stand

On a scale from "where we started" to "solved", for the binning question: about 70 %.

- Solved: we know the ceiling and have a recipe that reaches it when it can see the buckets
  (0.751 vs 0.776). We know the plain agent's two failure causes and which optimiser settings
  fix the second. We have a label-free policy at 0.74 of the oracle's 0.79 using 2 % of reads
  and leaving 98 % of normal slides clean, stable over seeds.
- Not solved: the gap between label-free and oracle (0.74 vs 0.79) is perception of boundary
  and small tiles from a single crop, and the small-vs-normal study says the lever for that is
  token-level scoring at finer resolution, which the agent's low-res head does not use yet.
  The read cost was chosen once from the stopping curve, not studied. Everything is on
  sub-patch Dice with the weak ViT-S high-res scorer. Stopping was characterised but no
  stopper was trained.
- The project's real question, fast segmentation, is further back: nothing has touched the
  high-res side (PFA or another segmenter on the zoomed tiles), and no timing was measured.

## 9. Caveats that change how to read the numbers

- Checkpoint selection on 9 val tumour slides; val-to-test gaps of 0.1 are common. The
  best-val-Dice criterion often picks a worse policy than the balanced one.
- Single seeds unless stated.
- Rows using ground-truth buckets or the oracle routing use test labels; they are ceilings.
- The CAMELYON16 annotations are polygons drawn at coarse zoom, 73 % of connected components
  are under 0.2 mm, and half of the small bucket is slivers; the floor on small-lesion recall
  is partly the labels.
- Dice here is at 512 px sub-patch resolution and all normal slides count as 1 or 0.

## 10. What to do next

1. Give the agent's low-res head the token-level scorer from the small-vs-normal study, at
   level 2, and rerun the clone-plus-costed-PPO recipe. That targets the one gap that is
   perception.
2. Sweep the read cost over three values with the same recipe; it is one run each.
3. Replace the ViT-S high-res head with a real segmenter on the zoomed tiles and measure
   pixel Dice and wall time per slide; that is the project's actual metric.
4. Train a stopper against the oracle stop curve only once the selector above is fixed.

## 11. Pointers

- Code, branch `worktree-rlbin`: `experiments/sep` (separability), `experiments/stop`
  (stopping ceiling), `experiments/rlbin` (agent, clone, baselines, frontier), `experiments/day`
  (this report, the 24-hour audit `LOG.md`, `RESULTS.md`). The small-vs-normal study is in
  the agent worktree under `experiments/small`. Neither worktree is merged into main.
- Server (`vivi`): `/data2/venkatavks/sep/report.md`, `/data2/venkatavks/small/report.md`,
  `/data2/venkatavks/stop/report.md`, `/data2/venkatavks/rlbin/` (per-run reports
  `report_<tag>_balanced.md`, `summary.md`, frontier plots, training logs and jsonl).
