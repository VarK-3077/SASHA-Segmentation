# Wall-clock cost of the screen pass on the test slides, run alone on the GPU box. Per slide and
# level: the page cache is dropped for the slide file, then the full screen is timed end to end
# (open slide, level-5 Otsu tissue mask, crop grid, reading 256 px crops at level L with a
# loader pool, Kaiko ViT-L/14 bf16, PCA projection, stitching, and the trained context head on
# the whole map), split into tissue, loader wait, GPU and head time. Level 2 is the reference:
# the same backbone over every 1024 px tile (four per unit), i.e. the token pass the segmenter
# already pays, with no head.

import argparse
import json
import os
import time

import numpy as np
import openslide
import torch

from experiments.lowsep import paths as P
from experiments.lowsep.extract import Crops, tokens
from experiments.lowsep.labels import block_reduce, pad_to, tissue_mask
from experiments.lowsep.model import ScreenHead
from experiments.lowsep.train import names_of
from experiments.sep.backbones import Encoder
from torch.utils.data import DataLoader


def crop_grid_from_tissue(slide, level):
    """Same crop grid as extract.crop_grid, but computed live from the Otsu mask (no label files)."""
    w, h = slide.dimensions
    t = tissue_mask(slide)
    fp = P.CROP_PX * 2 ** level
    k5 = fp // 2 ** P.TISSUE_LEVEL
    hc, wc = -(-h // fp), -(-w // fp)
    has = block_reduce(pad_to(t, hc * k5, wc * k5), k5, np.any)
    return np.argwhere(has).astype(np.int32), hc, wc


def run(name, level, enc, pca, head, workers, device):
    path = P.slide_path(name)
    fd = os.open(path, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
    torch.cuda.synchronize()
    t0 = time.time()
    s = openslide.OpenSlide(path)
    rc, hc, wc = crop_grid_from_tissue(s, level)
    t_tissue = time.time() - t0
    tm = dict(wait_io=0.0, gpu=0.0)
    canvas = torch.zeros((P.D_PCA, hc * P.TOK, wc * P.TOK), device=device, dtype=torch.bfloat16) if head else None
    k = 0
    it = iter(DataLoader(Crops(name, level, rc), batch_size=128, num_workers=workers, pin_memory=True))
    for _ in range(-(-len(rc) // 128)):
        s0 = time.time()
        x = next(it)
        s1 = time.time()
        t = tokens(enc, x.to(device, non_blocking=True)).permute(0, 2, 3, 1)
        t = ((t - pca['mu']) @ pca['W']).permute(0, 3, 1, 2)
        if head:
            for b, (r, c) in enumerate(rc[k:k + len(x)]):
                canvas[:, r * P.TOK:(r + 1) * P.TOK, c * P.TOK:(c + 1) * P.TOK] = t[b]
        k += len(x)
        torch.cuda.synchronize()
        tm['wait_io'] += s1 - s0; tm['gpu'] += time.time() - s1
    t_head = 0.0
    if head:
        h0 = time.time()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            head(canvas[None].float()).float().softmax(1)
        torch.cuda.synchronize()
        t_head = time.time() - h0
    return dict(slide=name, level=level, crops=int(len(rc)), wall=time.time() - t0, tissue=t_tissue, head=t_head, **tm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--levels', type=int, nargs='+', default=[2, 3, 4, 5, 6])
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    enc = Encoder('kaiko_vitl14', a.device)
    names = names_of('test', 6)
    out = {}
    for level in a.levels:
        pl = max(level, 3)
        pca = torch.load(f'{P.tok_dir(pl)}/pca.pt', weights_only=False)
        pca = {'mu': pca['mu'].to(a.device), 'W': pca['W'].to(a.device)}
        head = None
        if level >= 3:
            head = ScreenHead(P.tokens_per_unit(level), ctx=True).to(a.device).eval()
            head.load_state_dict(torch.load(f'{P.run_dir(f"L{level}_ctx_s0")}/best.pt', map_location=a.device)['model'])
        run(names[0], level, enc, pca, head, a.workers, a.device)  # warm-up: worker start, cudnn autotune
        rows = [run(n, level, enc, pca, head, a.workers, a.device) for n in names]
        mean = {k: float(np.mean([r[k] for r in rows])) for k in ('wall', 'tissue', 'wait_io', 'gpu', 'head', 'crops')}
        med = float(np.median([r['wall'] for r in rows]))
        out[f'L{level}'] = {'mean': mean, 'median_wall': med, 'slides': rows}
        print(f"L{level}: wall {mean['wall']:.2f} s/slide (median {med:.2f}), tissue {mean['tissue']:.2f}, io wait {mean['wait_io']:.2f}, "
              f"gpu {mean['gpu']:.2f}, head {mean['head']:.3f}, crops {mean['crops']:.0f}", flush=True)
        json.dump(out, open(f'{P.ROOT}/timing.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
