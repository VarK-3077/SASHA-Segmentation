# Precomputes everything an episode needs per slide so training never touches the heads:
# tile embedding, low-res and high-res scorer probabilities per 512 px sub-patch, the
# sub-patch ground truth, the tile's bucket, and a neighbourhood summary of the low-res score.

import os
import sys

import numpy as np
import torch

from experiments.sep import paths as SP
from experiments.stop import data as D
from experiments.stop.rollout import load_heads, predict
from experiments.rlbin import paths as P


def neighbour_mean(values, coords, step=2048):
    g = coords // step
    idx = {(int(x), int(y)): i for i, (x, y) in enumerate(g)}
    out = np.zeros(len(values), np.float32)
    for i, (x, y) in enumerate(g):
        nb = [idx[(x + dx, y + dy)] for dx in (-1, 0, 1) for dy in (-1, 0, 1)
              if (dx or dy) and (x + dx, y + dy) in idx]
        out[i] = values[nb].mean() if nb else values[i]
    return out


def main(device):
    heads = tuple(h.to(device) for h in load_heads())
    split = D.split_of()
    names = sorted(n for n in split if D.has_data(n))
    for i, n in enumerate(names):
        out = os.path.join(P.CACHE, n + '.npz')
        if os.path.exists(out):
            continue
        s = D.load_slide(n)
        p_lr, p_hr = predict(heads, s, device)
        z = np.load(os.path.join(SP.LABEL_DIR, n + '.npz'))
        np.savez(out, lr=s['lr'].astype(np.float16), p_lr=p_lr.astype(np.float16), p_hr=p_hr.astype(np.float16),
                 gt=s['sub_frac'] >= 0.5, bucket=z['L3_bucket'], coords=s['coords'],
                 nbr_p=neighbour_mean(p_lr.mean(1), s['coords']), split=split[n],
                 trust_normals=s['trust_normals'])
        if i % 50 == 0:
            print(f'{i}/{len(names)}', flush=True)
    print('cache done', flush=True)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'cuda:0')
