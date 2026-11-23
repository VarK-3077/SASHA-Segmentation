# Pooled level-3 embeddings per unit, for policies that decide from the embeddings instead of the heads' outputs:
# the mean and the maximum over a unit's 16 x 16 cached level-3 tokens (PCA-256 of the frozen ViT-L, the tokens
# the supervised L3 heads read), concatenated -> 512 numbers per unit. Nothing here is trained, so there is no
# fold to respect; stats.npz holds the train slides' per-feature mean and standard deviation for standardising.
# Usage: python -m experiments.rlpolicy.emb3 [--workers 4]

import argparse
import os
from multiprocessing import get_context

import numpy as np

from experiments.rlpolicy import where as W

OUT = f'{W.ROOT}/emb3'


def build(name):
    f = f'{OUT}/{name}.npz'
    if os.path.exists(f):
        return name
    tok = np.load(f'{W.TOK3}/{name}.npy', mmap_mode='r')
    rc = np.load(f'{W.TOK3}/{name}_rc.npy')
    out = np.zeros((len(tok), 512), np.float16)
    for i in range(0, len(tok), 256):
        t = np.asarray(tok[i:i + 256], np.float32).reshape(-1, 256, 256)
        out[i:i + 256] = np.concatenate([t.mean(2), t.max(2)], 1)
    np.savez(f'{f}.{os.getpid()}.npz', emb=out, rc=rc)
    os.replace(f'{f}.{os.getpid()}.npz', f)
    return name


def grid(s):
    """(hu, wu, 512) float32 standardised embeddings of slide s (a sim.Slide); zeros where no tokens are cached."""
    z, st = np.load(f'{OUT}/{s.name}.npz'), np.load(f'{OUT}/stats.npz')
    rc = z['rc']
    ok = (rc[:, 0] < s.hu) & (rc[:, 1] < s.wu)
    a = np.zeros((s.hu, s.wu, 512), np.float32)
    a[rc[ok, 0], rc[ok, 1]] = (z['emb'][ok].astype(np.float32) - st['mean']) / st['std']
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    from experiments.rlpolicy.simtab import names_of
    os.makedirs(OUT, exist_ok=True)
    train, test = names_of('train'), names_of('test')
    names = [n for n in train + test if os.path.exists(f'{W.TOK3}/{n}.npy')]
    print(len(train), 'train', len(test), 'test', len(names), 'with cached tokens', flush=True)
    with get_context('fork').Pool(a.workers) as pool:
        for i, n in enumerate(pool.imap_unordered(build, names), 1):
            if i % 40 == 0:
                print(i, n, flush=True)
    e = np.concatenate([np.load(f'{OUT}/{n}.npz')['emb'].astype(np.float32) for n in train if os.path.exists(f'{OUT}/{n}.npz')])
    np.savez(f'{OUT}/stats.npz', mean=e.mean(0), std=e.std(0) + 1e-6)
    print('done', e.shape, flush=True)


if __name__ == '__main__':
    main()
