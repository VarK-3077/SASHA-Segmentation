"""
Shared plumbing for the tests23 experiments (step13/14/15): config + model loading that
mirrors step10, readers for the re-extracted raw high-res sub-patch features, the level-3
low-res features/labels, and the sub-patch coordinate formula used by step12.
"""

import json
import os
import sys

import h5py
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from step10_sasha_attention_dump import load_models  # noqa: E402,F401  (re-exported on purpose)
from utils.utils import Struct  # noqa: E402

HR_DIR = os.path.expanduser('~/seg_outputs/hr_test_feats')
LABEL_H5 = '/data2/tej/SASHA/hafed13julyintmdtfeat/patch_feats_pretrain_medical_ssl.h5'
XML_DIR = '/data1/namanm/dataset/camelyon16/annotations'
SLIDE_DIR = '/data1/namanm/dataset/camelyon16/images/all'
MASK_DIR = '/data1/namanm/dataset/camelyon16/masks'
SUB_TILE_L0 = 512
LR_TILE_L0 = 2048


def load_conf(config_path, device, seed):
    with open(config_path, 'r') as f:
        c = yaml.load(f, Loader=yaml.FullLoader)
    c.update({'device': torch.device(device), 'seed': seed})
    return Struct(**c)


def test_names(seed, dataset='camelyon16'):
    with open(f'./dataset_csv/{dataset}/splits/split_{seed}.json', 'r') as f:
        return json.load(f)['test_names']


def load_lr(conf, names):
    """{name: {'feat': (N,384) fp32, 'coords': (N,2) int}} from the level-3 h5."""
    path = os.path.join(conf.level3_path, f'patch_feats_pretrain_{conf.pretrain}.h5')
    with h5py.File(path, 'r') as f:
        return {n: {'feat': f[n]['feat'][:].astype(np.float32), 'coords': f[n]['coords'][:]} for n in names}


def load_labels(names, label_h5=LABEL_H5):
    with h5py.File(label_h5, 'r') as f:
        return {n: int(f[n].attrs['label']) for n in names}


def has_xml(name, xml_dir=XML_DIR):
    return os.path.exists(os.path.join(xml_dir, name + '.xml'))


def load_hr(name, hr_dir=HR_DIR):
    """{'feat': (N,16,384) fp16 tensor, 'coords': (N,2)} written by step12."""
    return torch.load(os.path.join(hr_dir, name + '.pt'), map_location='cpu', weights_only=False)


def available_hr(hr_dir=HR_DIR):
    return sorted(f[:-3] for f in os.listdir(hr_dir) if f.endswith('.pt'))


def sub_coords(coords):
    """(N,2) tile coords -> (N*16,2) level-0 sub-patch coords, k-major as step12 wrote them."""
    k = np.arange(16)
    dx, dy = (k // 4) * SUB_TILE_L0, (k % 4) * SUB_TILE_L0
    out = np.empty((len(coords), 16, 2), dtype=np.int64)
    out[:, :, 0] = coords[:, 0:1] + dx[None, :]
    out[:, :, 1] = coords[:, 1:2] + dy[None, :]
    return out.reshape(-1, 2)
