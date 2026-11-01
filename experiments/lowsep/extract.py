# Screen-pass embeddings at one pyramid level: every 256 px crop at level L that covers at least
# one tissue unit is embedded by frozen Kaiko ViT-L/14 into 16 x 16 patch tokens (CLS and
# registers dropped, as in the token head). Tokens are reduced from 1024 to 256 dims with a PCA
# fitted on training slides (the small-lesion study found PCA-256 Kaiko tokens lose nothing),
# then stored per slide as a .npy (crops, 256, 16, 16) float16 plus the crop grid positions.
# Per-slide wall time is kept so the screen's real cost can be quoted.

import argparse
import json
import os
import time

import numpy as np
import openslide
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from experiments.lowsep import paths as P
from experiments.sep.backbones import Encoder

PCA_SLIDES = 40
PCA_TOKENS_PER_CROP = 32


def crop_grid(name, level):
    """Crop positions (row, col) on the level-L crop grid that cover at least one tissue unit."""
    tissue = np.load(P.label_npz(name))['tissue']
    m = P.units_per_crop(level)
    hu, wu = tissue.shape
    hc, wc = -(-hu // m), -(-wu // m)
    t = np.pad(tissue, ((0, hc * m - hu), (0, wc * m - wu)))
    has = t.reshape(hc, m, wc, m).any(axis=(1, 3))
    return np.argwhere(has).astype(np.int32)


class Crops(Dataset):
    def __init__(self, name, level, rc):
        self.path, self.level, self.rc, self.slide = P.slide_path(name), level, rc, None
        self.fp = P.CROP_PX * 2 ** level

    def __len__(self):
        return len(self.rc)

    def __getitem__(self, i):
        if self.slide is None:
            self.slide = openslide.OpenSlide(self.path)
        r, c = self.rc[i]
        a = np.asarray(self.slide.read_region((int(c * self.fp), int(r * self.fp)), self.level, (P.CROP_PX, P.CROP_PX)), np.float32)
        alpha = a[..., 3:4] / 255.0
        rgb = (a[..., :3] * alpha + 255.0 * (1 - alpha)).astype(np.uint8)  # out-of-slide pixels white
        rgb = np.asarray(Image.fromarray(rgb).resize((224, 224), Image.BILINEAR))
        return torch.from_numpy(rgb.copy()).permute(2, 0, 1)


@torch.no_grad()
def tokens(enc, x):
    """uint8 (B, 3, 224, 224) -> (B, 1024, 16, 16) float32 patch tokens."""
    x = (x.float() / 255.0 - enc.mean) / enc.std
    with torch.autocast('cuda', dtype=torch.bfloat16):
        h = enc.model.forward_features(x)
    t = h[:, h.shape[1] - P.TOK * P.TOK:].float()
    return t.transpose(1, 2).reshape(len(x), -1, P.TOK, P.TOK)


def slide_loader(name, level, rc, workers):
    return DataLoader(Crops(name, level, rc), batch_size=128, num_workers=workers, pin_memory=True)


def fit_pca(enc, level, names, workers, device):
    rng = np.random.default_rng(level)
    xtx, s, n = torch.zeros(1024, 1024, device=device, dtype=torch.float64), torch.zeros(1024, device=device, dtype=torch.float64), 0
    for name in names:
        rc = crop_grid(name, level)
        for x in slide_loader(name, level, rc, workers):
            t = tokens(enc, x.to(device)).permute(0, 2, 3, 1).reshape(len(x), -1, 1024)
            idx = torch.tensor(rng.integers(0, P.TOK * P.TOK, (len(x), PCA_TOKENS_PER_CROP)), device=device)
            t = torch.gather(t, 1, idx[..., None].expand(-1, -1, 1024)).reshape(-1, 1024).double()
            xtx += t.T @ t; s += t.sum(0); n += len(t)
    mu = s / n
    cov = xtx / n - torch.outer(mu, mu)
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, descending=True)[:P.D_PCA]
    kept = float(evals[order].sum() / evals.sum())
    return mu.float(), evecs[:, order].float(), kept, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--level', type=int, required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    out = P.tok_dir(a.level)
    os.makedirs(out, exist_ok=True)
    enc = Encoder('kaiko_vitl14', a.device)
    split = P.split_of()
    names = sorted(split)
    pca_path = f'{out}/pca.pt'
    if not os.path.exists(pca_path):
        train = sorted(n for n in names if split[n] == 'train')
        pick = list(np.random.default_rng(0).choice(train, PCA_SLIDES, replace=False))
        mu, W, kept, n = fit_pca(enc, a.level, pick, a.workers, a.device)
        torch.save({'mu': mu.cpu(), 'W': W.cpu(), 'kept': kept, 'n': n, 'slides': [str(s) for s in pick]}, pca_path)
        print(f'L{a.level} pca: {n} tokens from {PCA_SLIDES} train slides, variance kept {kept:.3f}', flush=True)
    pca = torch.load(pca_path, weights_only=False)
    mu, W = pca['mu'].to(a.device), pca['W'].to(a.device)
    times = {}
    for i, name in enumerate(names):
        f = f'{out}/{name}.npy'
        if os.path.exists(f):
            continue
        t0 = time.time()
        rc = crop_grid(name, a.level)
        buf = np.zeros((len(rc), P.D_PCA, P.TOK, P.TOK), np.float16)
        k = 0
        for x in slide_loader(name, a.level, rc, a.workers):
            t = tokens(enc, x.to(a.device)).permute(0, 2, 3, 1)
            t = ((t - mu) @ W).permute(0, 3, 1, 2)
            buf[k:k + len(x)] = t.half().cpu().numpy(); k += len(x)
        np.save(f'{out}/{name}_rc.npy', rc)
        np.save(f, buf)
        times[name] = {'crops': len(rc), 'sec': time.time() - t0}
        if i % 25 == 0:
            print(f'L{a.level} {i}/{len(names)} {name}: {len(rc)} crops, {times[name]["sec"]:.1f}s', flush=True)
    old = json.load(open(f'{out}/times.json')) if os.path.exists(f'{out}/times.json') else {}
    json.dump({**old, **times}, open(f'{out}/times.json', 'w'), indent=1)
    print(f'L{a.level} extraction done', flush=True)


if __name__ == '__main__':
    main()
