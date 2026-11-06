# Where the low-level separability study reads and writes, and its fixed geometry. The routing
# unit is the token-head segmenter's own input: a 2048 px level-0 crop on a grid anchored at
# (0, 0), read by the segmenter as four level-2 children. The screen reads 256 px crops at a
# lower level L, each covering m x m units (m = 2^(L-3)), and turns them into 16 x 16 tokens.

import json
import os

ROOT = os.environ.get('LOWSEP_ROOT', '/data2/venkatavks/lowsep')   # vivi: /var/tmp/venkatavks/lowsep
DATA = '/data1/namanm/dataset/camelyon16'
EXTRA = '/data2/venkatavks/c16_extra'
SPLIT_JSON = 'dataset_csv/camelyon16/splits/split_4.json'
LEVELS = (3, 4, 5, 6)
UNIT = 2048          # level-0 px per routing unit (segmenter crop)
CROP_PX, TOK = 256, 16
D_PCA = 256
MASK_DS = 16         # annotation raster, level-0 px per cell (unit = 128 cells)
TISSUE_LEVEL = 5     # the segmenter's Otsu tissue mask level


def units_per_crop(level):
    return 2 ** (level - 3)


def tokens_per_unit(level):
    return TOK // units_per_crop(level)


def slide_path(name):
    p = os.path.join(DATA, 'images', 'all', name + '.tif')
    return p if os.path.exists(p) else os.path.join(EXTRA, name + '.tif')


def xml_path(name):
    return os.path.join(DATA, 'annotations', name + '.xml')


def split_of():
    d = json.load(open(SPLIT_JSON))
    return {s: k.replace('_names', '') for k in d for s in d[k]}


def label_npz(name):
    return f'{ROOT}/labels/{name}.npz'


BACKBONE = os.environ.get('LOWSEP_BACKBONE', 'kaiko_vitl14')  # screen backbone; other values get their own dirs and run tags


def tok_dir(level, backbone=None):
    backbone = backbone or BACKBONE
    return f'{ROOT}/tokens_L{level}' + ('' if backbone == 'kaiko_vitl14' else f'_{backbone}')


def backbone_tag():
    return '' if BACKBONE == 'kaiko_vitl14' else f'_{BACKBONE}'


def run_dir(tag):
    return f'{ROOT}/runs/{tag}'
