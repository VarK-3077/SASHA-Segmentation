# Locations and the two episode constants of the RL binning study.

import os

ROOT = '/data2/venkatavks/rlbin'
CACHE = f'{ROOT}/cache'          # level-3 tiles, ViT-S embeddings, 16 sub-patches
CACHE_L2 = f'{ROOT}/cache_l2'    # level-2 tiles, Kaiko ViT-L embeddings, 4 sub-patches
TILE_STEP = {3: 2048, 2: 1024}
CKPT = f'{ROOT}/ckpt'
PBUCKET = f'{ROOT}/pbucket'   # predicted bucket probabilities per slide, from bucket_head.py
HEADS = '/data2/venkatavks/stop/heads.pt'   # low-res / high-res scorers from the stopping study

ZOOM_CAP = 0.5        # at most this fraction of a slide's tiles may be zoomed
MAX_STEPS_FRAC = 0.6  # episode is cut after this many actions per tile

for d in (CACHE, CKPT):
    os.makedirs(d, exist_ok=True)
