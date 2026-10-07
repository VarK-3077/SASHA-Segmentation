# Reads every sampled crop of one pyramid level from the slides once, runs all backbones
# on it, and stores the embeddings per backbone in one h5. Crops are visited slide by slide
# so each worker keeps a single open slide at a time. Finished backbones are skipped on
# rerun.

import argparse
import os
import time

import h5py
import numpy as np
import openslide
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from experiments.sep import paths as P
from experiments.sep.backbones import LOADERS, Encoder

INPUT_PX = 224


class Crops(Dataset):
    def __init__(self, slides, xs, ys, level):
        self.slides, self.xs, self.ys, self.level = slides, xs, ys, level
        self.handle = (None, None)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, i):
        name = self.slides[i]
        if self.handle[0] != name:
            self.handle = (name, openslide.OpenSlide(os.path.join(P.SLIDE_DIR, name + '.tif')))
        img = self.handle[1].read_region((int(self.xs[i]), int(self.ys[i])), self.level,
                                         (P.CROP_PX, P.CROP_PX)).convert('RGB')
        img = img.resize((INPUT_PX, INPUT_PX), Image.BILINEAR)
        return torch.from_numpy(np.asarray(img)).permute(2, 0, 1), i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--level', type=int, required=True)
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--backbones', nargs='+', default=list(LOADERS))
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()

    z = np.load(P.crops_npz(a.level))
    order = np.lexsort((z['y'], z['x'], z['slide']))
    slides, xs, ys = z['slide'][order], z['x'][order], z['y'][order]
    n = len(order)

    with h5py.File(P.feats_h5(a.level), 'a') as f:
        if 'order' not in f:
            f.create_dataset('order', data=order)
        todo = [b for b in a.backbones if not f.attrs.get(f'{b}_done', False)]
    if not todo:
        print('nothing to do')
        return
    encs = {b: Encoder(b, a.device) for b in todo}
    feats = {b: np.zeros((n, encs[b].dim), np.float16) for b in todo}

    dl = DataLoader(Crops(slides, xs, ys, a.level), batch_size=a.batch, num_workers=a.workers,
                    pin_memory=True)
    t0 = time.time()
    for bi, (img, idx) in enumerate(dl):
        x = img.to(a.device, non_blocking=True).float() / 255.0
        for b, e in encs.items():
            feats[b][idx.numpy()] = e(x).cpu().numpy().astype(np.float16)
        if bi % 50 == 0:
            done = (bi + 1) * a.batch
            print(f'L{a.level} {done}/{n} crops, {done / (time.time() - t0):.0f}/s', flush=True)

    with h5py.File(P.feats_h5(a.level), 'a') as f:
        for b in todo:
            out = np.empty_like(feats[b])
            out[order] = feats[b]            # back to crops_npz row order
            if b in f:
                del f[b]
            f.create_dataset(b, data=out, compression=None)
            f.attrs[f'{b}_done'] = True
    print(f'L{a.level} done in {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
