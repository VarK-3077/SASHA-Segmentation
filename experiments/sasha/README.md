# SASHA rollout and segmentation eval

SASHA is the RL-driven patch-visiting policy this whole repo is built around: it classifies
a whole-slide image while only "visiting" (extracting high-res features for) a fraction of
its tiles, guided by a cosine-similarity TSU and a trained agent. This directory does not
retrain SASHA — the released `best_neurips_models/camelyon/` checkpoints are used as-is — it
turns SASHA's own attention into a tumor **segmentation** and checks how well that
segmentation localizes lesions on CAMELYON16, independent of the (very good) slide
classification numbers.

Two things live here:
- the "tests23" eval (`step13_common.py`, `step13_hafed_full_hr.py`,
  `step14_sasha_rollout_p1feat.py`, `step15_tests23_metrics.py`): dumps HAFED's phase-1/phase-2
  attention on the raw high-res sub-patch features, replays the SASHA-0.2 rollout on the
  phase-1 aggregate, and scores four heatmap variants (full-res phase-1, full-res phase-2, and
  the SASHA-0.2 rollout map under softmax/sigmoid normalisation) against the lesion XMLs.
- `dump_rollout_states.py` and `dump_trainval_attention.py`: dump the SASHA-0.2 rollout's final
  feature state and attention (test set, then train/val) into pickles every downstream
  patch-classifier experiment (model1, model2, weak, acq, ...) reads as its input features —
  these are the bridge between SASHA and everything else in `experiments/`.

## How to run

```
python experiments/sasha/step13_hafed_full_hr.py --config config/camelyon_seg_0.2.yml --seed 4 \
    --device cuda:1 --out ~/seg_outputs/tests23/partA_dump.pkl
python experiments/sasha/step14_sasha_rollout_p1feat.py --config config/camelyon_seg_0.2.yml --seed 4 \
    --device cuda:1 --partA ~/seg_outputs/tests23/partA_dump.pkl --out ~/seg_outputs/tests23/partB_dump.pkl
python experiments/sasha/step15_tests23_metrics.py --partA .../partA_dump.pkl --partB .../partB_dump.pkl \
    --out ~/seg_outputs/tests23/results.json

python experiments/sasha/dump_rollout_states.py --config config/camelyon_seg_0.2.yml --seed 4 \
    --device cuda:1 --out ~/seg_outputs/patch_cls/rollout_states.pkl
python experiments/sasha/dump_trainval_attention.py --config config/camelyon_seg_0.2.yml --seed 4 \
    --device cuda:0 --out ~/seg_outputs/patch_cls/trainval_attention.pkl
```

## Released checkpoints

`best_neurips_models/camelyon/` (untracked, present both locally and on the server) are the
paper authors' own models: classifier seed 4 (n_token_2=5, epoch 16), TSU seed 4, RL agent
seed 1. `cosine_tsu` threshold is 0.9 for the 0.1-visit-fraction config and 0.95 for 0.2.
Verified against the server's re-extracted level-1 features: slide-level test AUC 0.982 /
acc 0.977 — SASHA classifies slides very well, but its attention localizes tumor only weakly
(tile-level AUC ≈0.68 on the SASHA rollout map, ≈0.74 on the full high-res HAFED attention),
which is what the segmentation numbers below reflect.

## Key results (test set, 129 slides, split seed 4, `utils/segmentation_metrics` harness)

| heatmap | FROC @0.5 | FROC @1-p | Dice @0.5 | Dice @1-p |
|---|---|---|---|---|
| SASHA-0.2, softmax-normalised rollout map | 0.215 | 0.327 | 0.164 | 0.155 |
| SASHA-0.2, sigmoid-normalised | similar, both rules well under 0.35 FROC / 0.20 Dice | | | |
| HAFED full-res phase-1 / phase-2 | comparable, all rows well below the feature-classifier rows in `../model1` and `../model2` | | | |

The `sasha_0.2_p2_attention` row of `../attn_stability`'s reproduction gate confirms this
FROC (0.093 on the phase-2-only map) reproduces the published number to a 1.4e-5 drift, so
these low numbers are the real behaviour, not a scoring bug — SASHA is not built to segment,
it is built to decide fast where to look, and every other experiment in `experiments/` tries
to get a usable tumor map out of the same visited-tile information some other way.

## Where the data lives

- Local: `outputs/sasha/*.json` (tests23 + old sasha_0.1/0.2 result dumps), `outputs/sasha/time/`
  (feature-extraction / patch-extraction wall-clock CSVs from the core step1/step2 pipeline).
- Server (`ssh vivi`, repo `~/SASHA-Segmentation` — still at the pre-reorg layout, code paths
  there are `patch_cls/...` and root `step13_*.py` etc., not `experiments/...`):
  - features: `/data2/tej/SASHA/camelyon_lr_features/patch_feats_pretrain_medical_ssl.h5` (level-3/LR),
    `/data2/tej/SASHA/hafed13julyintmdtfeat/...h5` (level-1/HAFED-aggregated); raw HR (Nx16x384)
    per-slide `.pt` features are NOT on the server for all slides (only under `~/seg_outputs/hr_test_feats`
    and `hr_train_feats`, partial).
  - WSIs `/data1/namanm/dataset/camelyon16/images/all`, XMLs `.../annotations`, masks `.../masks`.
  - run outputs `~/seg_outputs/tests23/`, `~/seg_outputs/patch_cls/{rollout_states.pkl,trainval_attention.pkl,labels.pkl}`.
  - venv: `~/envs/sasha-seg/bin/python` (torch 2.11+cu128, needed for the RTX 5090 / sm_120 GPUs).
