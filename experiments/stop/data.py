# Loads one slide's rollout substrate: the level-3 tile embedding, the 16 level-1 sub-patch
# embeddings per tile, and the sub-patch tumour fractions from the separability-study labels.
# step12 orders sub-patches x-major while the sep crops are y-major, hence the index map.

import json
import os

import h5py
import numpy as np
import torch

from experiments.patch_common.ncrf_non_exhaustive import NON_EXHAUSTIVE_TUMOR_SLIDES
from experiments.sep import paths as SP
from experiments.stop import paths as P

HR_K_TO_SEP = np.array([(k % 4) * 4 + k // 4 for k in range(16)])


def split_of():
    d = json.load(open(SP.SPLIT_JSON))
    return {s: k.replace('_names', '') for k in d for s in d[k]}


def hr_path(name):
    for d in P.HR_DIRS:
        p = os.path.join(d, name + '.pt')
        if os.path.exists(p):
            return p
    return None


def has_data(name):
    return hr_path(name) is not None and os.path.exists(os.path.join(SP.LABEL_DIR, name + '.npz'))


def load_slide(name):
    with h5py.File(SP.LR_H5, 'r') as f:
        lr = f[name]['feat'][:].astype(np.float32)
        coords = f[name]['coords'][:].astype(np.int64)
    d = torch.load(hr_path(name), map_location='cpu', weights_only=False)
    assert np.array_equal(np.asarray(d['coords']), coords), name
    z = np.load(os.path.join(SP.LABEL_DIR, name + '.npz'))
    sub_frac = z['L1_frac'].astype(np.float32).reshape(-1, 16)[:, HR_K_TO_SEP]
    return {'name': name, 'lr': lr, 'hr': d['feat'].numpy(), 'coords': coords,
            'sub_frac': sub_frac, 'tile_frac': z['L3_frac'].astype(np.float32),
            'trust_normals': name not in NON_EXHAUSTIVE_TUMOR_SLIDES}
