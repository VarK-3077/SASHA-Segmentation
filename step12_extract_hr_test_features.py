"""
Re-extracts the raw high-res sub-patch features (16 x 256px-at-level-1 crops per low-res
tile) for slides whose per-sub-patch features were not kept, at the exact coords stored in
the level-3 feature h5 so indices stay aligned with the existing SASHA dumps. Mirrors
step2/dataset_h5 crop logic: sub-patch k of tile (x,y) is read at level-0 offset
(x + (k//4)*512, y + (k%4)*512), 256px at level 1, x-major order. Saves one
<slide>.pt per slide: {'feat': (N,16,384) fp16, 'coords': (N,2)}.

python step12_extract_hr_test_features.py --level3_h5 H5 --slide_dir DIR --out_dir OUT --prefix test_
"""

import argparse
import os
import time

import h5py
import numpy as np
import openslide
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models_features_extraction.builder import vit_small

TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


class HRSubPatches(Dataset):
    """One item = one low-res tile -> stacked 16 transformed sub-patch crops."""

    def __init__(self, wsi_path, coords):
        self.wsi_path = wsi_path
        self.coords = coords
        self.wsi = None

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        if self.wsi is None:
            self.wsi = openslide.OpenSlide(self.wsi_path)
        x, y = (int(v) for v in self.coords[i])
        crops = []
        for xi in range(4):
            for yi in range(4):
                patch = self.wsi.read_region((x + xi * 512, y + yi * 512), 1, (256, 256)).convert('RGB')
                crops.append(TRANSFORM(patch))
        return i, torch.stack(crops)


def extract_slide(model, wsi_path, coords, device, batch_tiles, workers):
    ds = HRSubPatches(wsi_path, coords)
    loader = DataLoader(ds, batch_size=batch_tiles, num_workers=workers, pin_memory=True)
    out = torch.empty(len(coords), 16, 384, dtype=torch.float16)
    with torch.no_grad():
        for idx, imgs in loader:
            b = imgs.shape[0]
            feats = model(imgs.reshape(-1, 3, 224, 224).to(device, non_blocking=True))
            out[idx] = feats.reshape(b, 16, -1).to(torch.float16).cpu()
    return out


def main():
    parser = argparse.ArgumentParser('HR sub-patch feature extraction')
    parser.add_argument('--level3_h5', required=True, help='h5 with per-slide feat/coords groups (source of coords)')
    parser.add_argument('--slide_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--prefix', default='test_', help='only slides whose name starts with this')
    parser.add_argument('--batch_tiles', type=int, default=16, help='tiles per batch (x16 crops each)')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    model = vit_small(pretrained=True, progress=False, key='DINO_p16', patch_size=16).to(args.device)
    model.eval()
    os.makedirs(args.out_dir, exist_ok=True)

    with h5py.File(args.level3_h5, 'r') as f:
        names = sorted(n for n in f if n.startswith(args.prefix))
        coords_all = {n: f[n]['coords'][:] for n in names}

    for n, name in enumerate(names):
        out_path = os.path.join(args.out_dir, f'{name}.pt')
        if os.path.exists(out_path):
            continue
        t0 = time.time()
        coords = coords_all[name]
        feat = extract_slide(model, os.path.join(args.slide_dir, name + '.tif'), coords, args.device,
                             args.batch_tiles, args.workers)
        torch.save({'feat': feat, 'coords': coords}, out_path)
        print(f'[{n + 1}/{len(names)}] {name} N={len(coords)} {time.time() - t0:.1f}s', flush=True)
    with open(os.path.join(args.out_dir, 'DONE'), 'w') as f:
        f.write('\n'.join(names))
    print('all done')


if __name__ == '__main__':
    main()
