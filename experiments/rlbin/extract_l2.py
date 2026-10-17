# Embeds every 1024 px tile (256 px read at level 2) of every slide with Kaiko ViT-L/14, four
# children per level-3 tile in y-major order (child c = iy*2 + ix), one npz per slide so the
# run can resume. Feeds the level-2 variant of the binning agent and its low-res head.

import argparse
import os
import time

import numpy as np
import openslide
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from experiments.rlbin import paths as P
from experiments.sep import paths as SP
from experiments.sep.backbones import Encoder

L2_DIR = f'{P.ROOT}/l2feats'
CHILD_OFFS = np.array([[0, 0], [1024, 0], [0, 1024], [1024, 1024]])  # (dx, dy) for c = iy*2 + ix


class TileChildren(Dataset):
    def __init__(self, path, coords):
        self.path, self.coords = path, coords
        self.wsi = None

    def __len__(self):
        return len(self.coords) * 4

    def __getitem__(self, i):
        if self.wsi is None:
            self.wsi = openslide.OpenSlide(self.path)
        t, c = divmod(i, 4)
        x, y = int(self.coords[t, 0] + CHILD_OFFS[c, 0]), int(self.coords[t, 1] + CHILD_OFFS[c, 1])
        img = self.wsi.read_region((x, y), 2, (256, 256)).convert('RGB').resize((224, 224), Image.BILINEAR)
        return torch.from_numpy(np.array(img)).permute(2, 0, 1), i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--workers', type=int, default=24)
    a = ap.parse_args()
    os.makedirs(L2_DIR, exist_ok=True)
    enc = Encoder('kaiko_vitl14', a.device)
    names = sorted(f[:-4] for f in os.listdir(P.CACHE))
    t0, done = time.time(), 0
    for n in names:
        out = os.path.join(L2_DIR, n + '.npz')
        if os.path.exists(out):
            continue
        coords = np.load(os.path.join(P.CACHE, n + '.npz'))['coords']
        ds = TileChildren(os.path.join(SP.SLIDE_DIR, n + '.tif'), coords)
        feats = np.zeros((len(ds), enc.dim), np.float16)
        for img, idx in DataLoader(ds, batch_size=256, num_workers=a.workers, pin_memory=True):
            feats[idx.numpy()] = enc(img.to(a.device).float() / 255.0).cpu().numpy().astype(np.float16)
        np.savez(out, feat=feats, coords=coords)
        done += len(ds)
        print(f'{n}: {len(ds)} crops, {done / (time.time() - t0):.0f} crops/s overall', flush=True)
    print('l2 extraction done', flush=True)


if __name__ == '__main__':
    main()
