# Training subset for the token-level L3 head, cut once from the cached level-3 tokens of the trusted train
# slides: every unit that holds tumour, the HARD normal units the out-of-fold unit head scores highest and
# RAND random others per slide. Each unit keeps its 16 x 16 tokens and a 16 x 16 label map, the tumour share
# of each token's footprint (4 x 4 cells of the scorer's level-5 ground truth).
# Usage: python -m experiments.rlpolicy.tok3_data [--workers 4]

import argparse
import os
import zlib
from multiprocessing import get_context

import numpy as np

from experiments.rlpolicy import where as W

HARD, RAND, U5, T = 150, 150, 64, 16


def train_names():
    out = []
    for f in sorted(os.listdir(W.UNITS)):
        if f.endswith('.npz'):
            z = np.load(f'{W.UNITS}/{f}')
            if str(z['split']) == 'train' and bool(z['trust']) and z['tissue'].any():
                out.append(f[:-4])
    return out


def token_labels(mask, rc):
    """(n, 16, 16) uint8: tumour cells (of 16) under each token of the units rc; mask is the level-5 ground truth or None."""
    out = np.zeros((len(rc), T, T), np.uint8)
    if mask is None:
        return out
    for i, (r, c) in enumerate(rc):
        src = mask[r * U5:(r + 1) * U5, c * U5:(c + 1) * U5] > 0
        if src.any():
            blk = np.zeros((U5, U5), np.uint8)
            blk[:src.shape[0], :src.shape[1]] = src
            out[i] = blk.reshape(T, U5 // T, T, U5 // T).sum((1, 3))
    return out


def gt_mask(name):
    z = np.load(f'{W.GT}/{name}.npz')
    return z['mask'] if 'mask' in z.files else None


def pick(name):
    z = np.load(f'{W.UNITS}/{name}.npz')
    oof = np.load(f'{W.UNITS}/oof/{name}.npz')
    rc = np.load(f'{W.TOK3}/{name}_rc.npy')
    hu, wu = z['tissue'].shape
    ok = (rc[:, 0] < hu) & (rc[:, 1] < wu)
    rows = np.flatnonzero(ok)
    r, c = rc[rows, 0], rc[rows, 1]
    frac = z['frac'][r, c]
    score = 1 - oof['p3_unit'][r, c, 0].astype(np.float32)
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    normal = np.flatnonzero(frac == 0)
    hard = normal[np.argsort(-score[normal], kind='stable')[:HARD]]
    rest = np.setdiff1d(normal, hard)
    rand = rng.choice(rest, min(RAND, len(rest)), replace=False)
    sel = np.sort(np.concatenate([np.flatnonzero(frac > 0), hard, rand]))
    return rows[sel], rc[rows[sel]], frac[sel]


def one(name):
    rows, rc, frac = pick(name)
    tok = np.load(f'{W.TOK3}/{name}.npy', mmap_mode='r')
    return name, np.ascontiguousarray(tok[rows]), rc, frac, token_labels(gt_mask(name), rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    os.makedirs(W.SUB, exist_ok=True)
    names = train_names()
    folds = W.folds()
    n = sum(len(pick(nm)[0]) for nm in names)
    print(len(names), 'slides', n, 'units', flush=True)
    out = np.lib.format.open_memmap(f'{W.SUB}/tok.npy', mode='w+', dtype=np.float16, shape=(n, 256, T, T))
    meta = dict(slide=[], rc=[], frac=[], lab=[])
    k = 0
    with get_context('fork').Pool(a.workers) as pool:
        for i, (name, tok, rc, frac, lab) in enumerate(pool.imap(one, names)):
            out[k:k + len(rc)] = tok
            k += len(rc)
            meta['slide'].append(np.full(len(rc), names.index(name)))
            meta['rc'].append(rc); meta['frac'].append(frac); meta['lab'].append(lab)
            if i % 20 == 0:
                print(i, name, k, flush=True)
    out.flush()
    np.savez(f'{W.SUB}/meta.npz', names=np.array(names), fold=np.array([folds[x] for x in names]),
             **{key: np.concatenate(v) for key, v in meta.items()})
    print('done', k, flush=True)


if __name__ == '__main__':
    main()
