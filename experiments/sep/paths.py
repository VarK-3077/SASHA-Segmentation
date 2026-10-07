# Where the separability study reads from and writes to on the server. Everything the
# study produces lives under /data2 because the home disk is full.

import os

ROOT = '/data2/venkatavks/sep'
LABEL_DIR = f'{ROOT}/labels'
PROBE_DIR = f'{ROOT}/probes'
SLIDE_DIR = '/data1/namanm/dataset/camelyon16/images/all'
XML_DIR = '/data1/namanm/dataset/camelyon16/annotations'
LR_H5 = '/data2/tej/SASHA/camelyon_lr_features/patch_feats_pretrain_medical_ssl.h5'
SPLIT_JSON = 'dataset_csv/camelyon16/splits/split_4.json'

LEVELS = (3, 2, 1, 0)
TILE_L0 = 2048          # footprint of one tile in the level-3 feature h5
CROP_PX = 256           # every crop is 256 px at its own level
MASK_DS = 16            # masks are rasterised at level 4 (15 um/px); level 3 made the pool thrash


def crops_npz(level):
    return f'{ROOT}/crops_L{level}.npz'


def feats_h5(level):
    return f'{ROOT}/feats_L{level}.h5'


def footprint_l0(level):
    return CROP_PX * (2 ** level)


os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(PROBE_DIR, exist_ok=True)
