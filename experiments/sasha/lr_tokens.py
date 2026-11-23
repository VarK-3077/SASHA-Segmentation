"""Low-resolution (level 3) tile features for the supervised tile classifiers, from SASHA's own ViT-S (DINO, lunit
medical_ssl) and its own transforms: the CLS feature SASHA uses, and the mean and the max of the 196 patch tokens
(the per-channel mean+max pooling that kept small lesions in the small-lesion study).
  encode(model, x)          one forward -> (cls, mean, max)
  timed_meanmax(...)         the timed pipeline's pass (time_infer.py --mode tilemlp), SASHA's loader settings
  python -m experiments.sasha.lr_tokens --out DIR   caches every slide's tiles, in the order and at the coordinates
                             of the stored low-res feature h5, and checks the CLS against the stored features."""

import argparse
import os
from multiprocessing import get_context

import h5py
import numpy as np
import openslide
import torch
from torch.utils.data import DataLoader, Dataset

LR_H5 = '/data2/tej/SASHA/camelyon_lr_features/patch_feats_pretrain_medical_ssl.h5'
SOURCES = ('/data1/namanm/dataset/camelyon16/images/all', '/data2/venkatavks/c16_extra')
LEVEL, SIZE = 3, 256


@torch.no_grad()
def encode(model, x):
    t = model.forward_features(x)
    p = t[:, 1:]
    return t[:, 0], p.mean(1), p.amax(1)


def timed_meanmax(helper, slide, conf):
    """Mean+max token features of every tile with the loader SASHA's extract_lr_features uses (8 workers, batch 512)."""
    from datasets.dataset_h5 import Whole_Slide_Bag_FP
    wsi = openslide.open_slide(os.path.join(conf.source, f'{slide}.{conf.slide_ext}'))
    ds = Whole_Slide_Bag_FP(file_path=os.path.join(helper.step1_save_dir, f'patches/{slide}.h5'), wsi=wsi, img_transforms=helper.img_transforms,
                            extract_high_res_features=False, patch_level_high_res=conf.patch_level_high_res,
                            patch_level_low_res=conf.patch_level_low_res, dataset_name=conf.dataset)
    feats, coords = [], []
    for d in DataLoader(ds, batch_size=conf.batch_size_lr, num_workers=8, pin_memory=True):
        _, mean, mx = encode(helper.feature_extractor, d['img'].to(conf.device, non_blocking=True))
        feats.append(torch.cat([mean, mx], 1))
        coords.append(d['coord'].numpy())
    return torch.cat(feats), np.concatenate(coords)


def slide_path(name):
    return next(p for p in (os.path.join(s, name + '.tif') for s in SOURCES) if os.path.exists(p))


class Tiles(Dataset):
    def __init__(self, name, coords, tf):
        self.path, self.coords, self.tf, self.wsi = slide_path(name), coords, tf, None

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        if self.wsi is None:
            self.wsi = openslide.open_slide(self.path)
        x, y = (int(v) for v in self.coords[i])
        return self.tf(self.wsi.read_region((x, y), LEVEL, (SIZE, SIZE)).convert('RGB'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    from models_features_extraction.builder import get_encoder
    os.makedirs(a.out, exist_ok=True)
    model, tf = get_encoder('ViT-S/16', 'medical_ssl')
    model = model.eval().to(a.device)
    with h5py.File(LR_H5, 'r') as f:
        names = sorted(f)
    for i, name in enumerate(names):
        out = f'{a.out}/{name}.npz'
        if os.path.exists(out):
            continue
        with h5py.File(LR_H5, 'r') as f:
            coords, stored = f[name]['coords'][:], f[name]['feat'][:].astype(np.float32)
        cls, mm = [], []
        for x in DataLoader(Tiles(name, coords, tf), batch_size=256, num_workers=a.workers):
            c, mean, mx = encode(model, x.to(a.device))
            cls.append(c.cpu().numpy())
            mm.append(torch.cat([mean, mx], 1).cpu().numpy())
        cls = np.concatenate(cls)
        cos = (cls * stored).sum(1) / (np.linalg.norm(cls, axis=1) * np.linalg.norm(stored, axis=1) + 1e-8)
        np.savez(f'{out}.tmp.npz', meanmax=np.concatenate(mm).astype(np.float16), cls_cos=cos.astype(np.float32))
        os.replace(f'{out}.tmp.npz', out)
        if i % 25 == 0:
            print(i, name, len(coords), f'CLS vs stored feature: min cosine {cos.min():.4f}, mean {cos.mean():.4f}', flush=True)
    open(f'{a.out}/DONE', 'w').close()


if __name__ == '__main__':
    main()
