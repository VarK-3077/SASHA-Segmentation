# The level-3 screening model: exact recipe

What it is: a crop classifier that reads a whole slide at pyramid level 3 and gives, for every
2048 px crop of the segmenter's grid, a probability for each of four kinds (normal / core /
boundary / small lesion). Its use: crops with P(tumour) = 1 − P(normal) above a threshold go to
the level-2 segmenter (ViT-L + conv layer, `experiments/tokhead`), every other crop stays empty.
Final model: run `L3_ctx_d128_s0`. Written 2026-11-04 from the code on branch `nov2-deck`
(commit acc65eb) and the run files; every value below is read from code or files, not memory.

## 1. Results it produces (for checking a recreation)

Test = all tissue crops of the 129 CAMELYON16 test slides (132,837 crops).

| measure | normal | core | boundary | small |
|---|---|---|---|---|
| recall, argmax | 0.942 | 0.836 | 0.777 | 0.683 |
| recall at 99 % specificity on normal crops (threshold from test normals) | – | 0.989 | 0.905 | 0.601 |

Confusion, rows = true, columns = predicted (normal, core, boundary, small):
normal 119,507 / 12 / 491 / 6,878; core 13 / 2,709 / 473 / 44; boundary 83 / 131 / 1,687 / 270;
small 114 / 0 / 57 / 368.

Validation: best macro-recall over the four classes 0.854 at step 1050; AUC tumour-vs-normal
crops on val 0.9963. Thresholds on P(tumour), set on validation to keep a share of val tissue
crops: 20 % -> 0.12554852664470673 (keeps 17.2 % of test crops), 10 % -> 0.6106325387954712
(9.0 % of test crops).

In the real cascade (screen + level-2 token head on kept crops, pipelined, one A100, cold page
cache, 129 test slides), vs segmenting every crop (8.44 s/slide, FROC 0.844, per-slide Dice 0.782
with the segmenter's validation-chosen cleaning t0.4 seed 0.9): 20 % budget 3.25 s/slide, FROC
0.814, per-slide Dice 0.782; 10 % budget 2.75 s/slide, FROC 0.794, Dice 0.773. Screen alone,
tissue mask included: 2.91 s/slide mean (2.76 median), 1,030 level-3 crops per slide.

## 2. Environment

- Machine used: isp-a100 (Brev, A100 80 GB, 12 CPUs, vivi-identical paths); vivi works the same.
- Python venv `~/envs/sasha-seg` (python 3.11.16 from `~/pybin`): torch 2.11.0+cu128, timm 1.0.28,
  scikit-learn 1.9.0, openslide-python 1.4.6 with openslide-bin 4.0.1.2,
  opencv-python-headless 5.0.0.93, numpy / Pillow / scikit-image as installed in that venv.
- `export PYTHONPATH=. TORCH_HOME=/data2/venkatavks/torch_cache CAMELYON16_MIRROR=/data2/venkatavks/c16_extra`
- Backbone weights: `torch.hub.load('kaiko-ai/towards_large_pathology_fms', 'vitl14', trust_repo=True)`,
  cached at `/data2/venkatavks/torch_cache/hub/kaiko-ai_towards_large_pathology_fms_main` (+ `checkpoints/`).
  Not pinned to a commit: the hub repo's `main` at the time of the first download. Keep that cache.

## 3. Data

- CAMELYON16, 399 slides: 384 in `/data1/namanm/dataset/camelyon16/images/all/`, the other 15
  normals in `/data2/venkatavks/c16_extra/` (`paths.slide_path` looks in both). Annotations
  `/data1/namanm/dataset/camelyon16/annotations/<slide>.xml`.
- Split: `dataset_csv/camelyon16/splits/split_4.json` (slide-grouped): 242 train, 27 val (9 of them
  with tumour), 129 test (the official test set).
- Every slide must have pyramid level > 6 (`labels.py` asserts it).

## 4. Units and labels (`labels.py`, run once: `python -m experiments.lowsep.labels 10`)

- Unit = 2048 × 2048 level-0 px on a grid anchored at (0, 0): unit (r, c) covers
  x in [2048c, 2048c+2048), y in [2048r, 2048r+2048). Grid size ceil(H/2048) × ceil(W/2048).
  This is exactly the token-head segmenter's crop grid; keep them identical.
- Tissue, as the segmenter does it: read level 5 (1/32) whole, RGB -> HSV, Otsu threshold on the
  saturation channel (`cv2.THRESH_BINARY + cv2.THRESH_OTSU`), AND value > 40 (drops black scanner
  margins). A unit has tissue if any of its 64 × 64 level-5 pixels does. Units holding any tumour
  are added to tissue (`tissue | frac > 0`).
- Tumour raster: XML polygons, `PartOfGroup` in {'_2', 'exclusion'} = exclusion, everything else
  tumour (`utils/segmentation_metrics.read_xml_polygons`); polygons scaled by 1/16, rounded,
  `cv2.fillPoly` tumour = 1 then exclusion = 0, at 16 level-0 px per cell (`MASK_DS = 16`; a unit
  is 128 × 128 cells).
- Per unit: frac = mean of the raster over its 128 × 128 cells. Lesions = 8-connected components
  of the raster (`skimage.measure.label(connectivity=2)`); lesion area in level-0 px =
  cells × 16². For each unit, the largest area among the lesions it touches.
- Buckets (`experiments/sep/buckets.assign`, footprint 2048): normal if frac = 0; core if
  frac ≥ 0.9; otherwise boundary if the largest touching lesion's area > 2048² level-0 px (the
  lesion is bigger than the unit), else small.
- Trust: the 20 training tumour slides with non-exhaustive annotations
  (`experiments/patch_common/ncrf_non_exhaustive.NON_EXHAUSTIVE_TUMOR_SLIDES`: tumor_010, 015,
  018, 020, 025, 029, 033, 034, 044, 046, 051, 054, 055, 056, 067, 079, 085, 092, 095, 110) get
  trust = False; their normal units are never used as targets.
- Gotcha: `labels.py` imports `lesion_lookup` from `experiments/sep/build_labels.py`, which
  multiplies by `experiments/sep/paths.MASK_DS`. That is also 16; if either changes, areas break.
- Output `/data2/venkatavks/lowsep/labels/<slide>.npz`: tissue, frac, bucket, trust, mpp, dims.

## 5. Screen tokens (`extract.py`: `python -m experiments.lowsep.extract --level 3 --device cuda:0 --workers 10`)

- Crops: at level 3 one 256 × 256 crop = one unit (256 × 2³ = 2048 level-0 px). A crop is
  extracted if its unit has tissue. Read with `openslide.read_region((2048c, 2048r), 3, (256, 256))`,
  RGBA composited on white (alpha blend, out-of-slide pixels become white), resized to 224 × 224
  with PIL bilinear.
- Backbone: Kaiko ViT-L/14, frozen, eval. Input = uint8 / 255, normalised with mean 0.5 and
  std 0.5 per channel. `model.forward_features(x)` under `torch.autocast('cuda', bfloat16)`; keep the
  last 256 tokens of the sequence (16 × 16 patch tokens; CLS and register tokens dropped), cast to
  float32, reshape to (B, 1024, 16, 16). Batch 128, 10 loader workers, pin_memory.
- PCA 1024 -> 256, fitted once per level on training slides only: 40 slides drawn with
  `np.random.default_rng(0).choice(sorted(train names), 40, replace=False)`; from every crop of
  those slides 32 tokens drawn uniformly with `np.random.default_rng(3)` (seed = level). Mean and
  covariance accumulated in float64 on the GPU, `torch.linalg.eigh`, top-256 eigenvectors. L3:
  1,304,608 tokens, variance kept 0.914. Saved to `/data2/venkatavks/lowsep/tokens_L3/pca.pt`
  ({mu, W, kept, n, slides}; load with `weights_only=False`).
- Projection: token' = (token − mu) @ W, stored float16. Per slide:
  `tokens_L3/<slide>.npy` shape (crops, 256, 16, 16) and `tokens_L3/<slide>_rc.npy` (crop row, col),
  rows in row-major order. All 398 slides: 52 GB.

## 6. Model (`model.py`, `ScreenHead(tok_per_unit=16, ctx=True, d=128, p_drop=0.3)`)

Input: a token map (B, 256, 16·H, 16·W) for H × W units (missing crops = zeros). Output: logits
(B, 4, H, W), classes in order normal, core, boundary, small.

1. `proj`: Conv2d 1×1, 256 -> 128, GELU.
2. `tok_ctx`: GlobalConv(128, k=15) added as a residual, then GELU. GlobalConv = two separable
   branches summed: Conv(15×1, pad 7) -> Conv(1×15, pad 7), plus Conv(1×15) -> Conv(15×1); all
   128 -> 128 with bias.
3. Per unit pooling over its 16 × 16 tokens: avg_pool2d(16) and max_pool2d(16), concatenated (256 ch).
4. `mix`: Conv2d 1×1, 256 -> 128, GELU.
5. `unit_ctx`: GlobalConv(128, k=7) over units (pad 3), residual, GELU.
6. `hidden`: Conv2d 1×1, 128 -> 128, GELU.
7. Dropout2d(0.3) (training only), `cls`: Conv2d 1×1, 128 -> 4.

1,525,636 parameters. Receptive field: 7 tokens (0.44 unit) at token level + 3 units at unit
level, under 4 units in every direction. There is no slide- or window-mean vector on purpose:
training sees 32 × 32 windows and inference whole slides, a mean would differ between the two.

## 7. Training (`train.py`)

Command for the final model:
`python -m experiments.lowsep.train --level 3 --ctx 1 --d 128 --drop 0.3 --wd 0.01 --name _d128 --seed 0 --steps 1500 --eval_every 150 --batch 4 --device cuda:0 --workers 2`

- Slides: the 242 train slides; per slide a `SlideMaps` (labels + memory-mapped tokens).
- Targets per unit: bucket where tissue, −1 (ignored) elsewhere; normal units of untrusted slides −1.
- Class weights: inverse of the training counts of each class over all valid units, normalised
  so they sum to 4 (`w = (1/count) / sum(1/count) × 4`).
- Samples: square windows of 32 × 32 crops (= 32 × 32 units at L3). Pick a centre crop: with
  probability 0.5 a crop whose unit holds tumour (any training slide), else any tissue crop; top-left
  = centre − 16 + U{−8 … 8} per axis (windows may extend past the slide: zeros / ignored).
  Augmentation: rotate by k × 90°, k uniform in {0, 1, 2, 3}, then horizontal flip with p = 0.5,
  applied to tokens and targets together. No colour augmentation (tokens are precomputed).
- Loader: `IterableDataset`, batch 4, 2 workers, persistent. Each worker seeds its RNG with
  `np.random.default_rng([seed, worker_id, int(time.time())])` — the clock makes window sampling
  non-reproducible. For a bit-exact rerun replace `int(time.time())` with a fixed number.
- Loss: `F.cross_entropy(logits.float(), target, weight=w, ignore_index=-1)`; forward under bf16
  autocast, loss in float32. No gradient clipping.
- Optimiser: AdamW, weight decay 0.01, `OneCycleLR(max_lr=1e-3, total_steps=1500, pct_start=0.05)`
  (PyTorch defaults otherwise: cosine annealing, start lr 4e-5, final lr 4e-9, momentum cycling of
  beta1 0.85 -> 0.95). `torch.manual_seed(0)` before building the model.
- Validation every 150 steps: each of the 27 val slides scored whole (one forward of the full
  token map, bf16, softmax), argmax per unit, recall of each class over the valid units, mean of
  the four = macro-recall. The checkpoint with the best macro-recall is saved
  (`runs/L3_ctx_d128_s0/best.pt`: {model, step, val_macro_recall}).
- Run log (step, last batch loss, val macro-recall, val recall normal/core/boundary/small):
  150 0.646 (small 0.00) · 300 0.718 · 450 0.802 · 600 0.824 · 750 0.826 · 900 0.819 ·
  **1050 0.854 (0.920 / 0.885 / 0.726 / 0.885) = kept** · 1200 0.842 · 1350 0.842 · 1500 0.838.
  16 min on the A100 while two other trainings shared it.

## 8. How the final model was chosen

1. First pass (`chain.sh`): width 256, dropout 0.1, weight decay 1e-4, 3000 steps, eval every
   300, with and without context, seeds 0 and 1. These overfit (training loss down to ~0.01 while
   val stalled).
2. Sweep (`sweep.sh`): context width 128 (dropout 0.3, wd 0.01), context width 64 (dropout 0.5,
   wd 0.05), no-context width 64 (dropout 0.5, wd 0.05); 1500 steps, eval every 150; seeds 0, 1.
3. Choice (`cascade_prep.py`): among the seed-0 run of each of the 5 configurations, the one with
   the highest validation AUC of P(tumour) for tumour units (bucket > 0) vs normal units ->
   `L3_ctx_d128_s0` (0.9963). The same pick results from ranking by validation ideal-segmenter
   Dice at a 20 % budget (`screen_ideal.py --select dice`, 0.9997).

All ten L3 runs on test (recall at 99 % spec core / boundary / small; ideal-segmenter Dice at 20 %):
context-256 s0 0.982 / 0.885 / 0.430, 0.841 · s1 0.985 / 0.892 / 0.482, 0.844 ·
context-128 s0 0.989 / 0.905 / 0.601, 0.964 · s1 0.990 / 0.905 / 0.512, 0.963 ·
context-64 s0 0.986 / 0.910 / 0.581, 0.942 · s1 0.990 / 0.906 / 0.508, 0.956 ·
no-context-256 s0 0.956 / 0.759 / 0.494, 0.969 · s1 0.962 / 0.737 / 0.490, 0.971 ·
no-context-64 s0 0.977 / 0.782 / 0.534, 0.971 · s1 0.974 / 0.773 / 0.473, 0.967.
Seed spread on small-lesion recall is about 0.09; on Dice about 0.01.

## 9. Using it (`cascade_pipe.py`)

`TOK_LEVEL=2 OMP_NUM_THREADS=1 python -m experiments.lowsep.cascade_pipe --level 3 --budget 0.2 --seg_ckpt /data2/venkatavks/pfa_ckpt/named/tokhead_kaiko_vitl14_stageA_input_l2.pt --out_dir DIR`
(reads `cascade_cfg.json` for the head tag and threshold). Per slide: Otsu tissue mask, crops in
row-major order through 8 persistent reader processes, ViT-L + PCA, tokens written into a GPU
canvas; after each batch of 128 crops the head reruns on the fully screened rows and a unit is
final once it is more than 5 unit-rows behind the screening front (5 > the receptive field, so
decisions equal the whole-slide ones: verified identical on 129/129 test slides). Final units with
P(tumour) ≥ threshold go to 8 other persistent readers (level-2 children) and the segmenter in
batches of 64 while the screen continues. Unkept crops are 0 in the output map.

## 10. Issues met and how they were resolved

- Overfitting of the first 256-wide heads: fixed by the narrower, more regularised sweep (§8).
- Argmax routing was unusable: across seeds of one configuration the share of crops predicted
  not-normal swung from 8 % to 75 % (calibration). Route by thresholding P(tumour) at a budget
  calibrated on validation instead.
- Validation is too small to choose heads reliably (9 tumour slides): AUC and Dice-based picks
  saturate near 0.99 for most heads. At L3 both picks agree; at L4 / L5 they picked heads that
  rank crops worse than others on test. Report seed and configuration spread, not one run.
- A window-mean context vector was removed from the head: training windows and whole-slide
  inference would see different means.
- The unit grid and tissue mask must be the segmenter's own (level-5 Otsu, grid at (0, 0)),
  otherwise kept crops do not line up with segmenter inputs.
- `torch.load` of `pca.pt` failed under PyTorch 2.11's `weights_only=True` default (numpy strings in
  the slide list): store `str` names and load with `weights_only=False`.
- Timing: the first (sequential) cascade spawned two loader pools per slide and waited for the full
  screen before segmenting (5.44 s/slide at 20 %); persistent pools + streaming gave 3.25 s with
  identical outputs. Time solo, cold cache (`posix_fadvise DONTNEED` per slide), `OMP_NUM_THREADS=1`.
- The `screen s/slide` column in `/data2/venkatavks/lowsep/report.md`'s first table comes from the
  extraction pass on a shared GPU; the clean screen timings are in `timing.json` (`time_screen.py`).
- Box operations (isp-a100 shared with other sessions): agree who stops the box; a killed vivi
  watcher's `sleep` child keeps its lock (find it with `fuser -v` on the lock file); never
  `pgrep -f` / `pkill -f` a pattern that also matches the ssh command line itself.

## 11. Files

- Code: `experiments/lowsep/` (`paths, labels, extract, maps, model, train, evaluate, routing,
  cascade_prep, cascade, cascade_pipe, time_screen, screen_ideal, chain.sh, sweep.sh`), branch
  `nov2-deck` / `worktree-rlbin` of the SASHA-Segmentation repo.
- isp-a100: `/data2/venkatavks/lowsep/` — `labels/`, `tokens_L3/` (with `pca.pt`), `runs/L3_ctx_d128_s0/`
  (`best.pt`, `args.json`, `train.jsonl`, `test.json`), `cascade_cfg.json`, `cascade/cascadepipe_L3_b0.{1,2}/`,
  `timing.json`, `deck_eq/` (equal-budget ideal-segmenter scores).
- Local copy of the run: `outputs/lowsep/runs/L3_ctx_d128_s0/` (includes `best.pt`).
