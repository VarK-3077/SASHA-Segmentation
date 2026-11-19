# Per-slide tables for the fast cascade simulator, built once from the dense level-2 map of the real segmenter
# (out-of-fold maps for train slides): every connected component of the map (>= 0.4, tissue only, at the
# scorer's level-5 raster) split by unit, with its pixels, true-positive pixels, maximum and the lesion label
# under that maximum. A routing policy's map is the dense map kept on the units it segments, so its cleaned
# Dice and FROC candidates follow from these rows (sim.py). Also what a policy sees after a zoom (painted
# share, border strips, maximum per unit) and what painting a unit would add.
# Usage: python -m experiments.rlpolicy.simtab --split train|test [--workers 6]

import argparse
import os
from functools import partial
from multiprocessing import get_context

import numpy as np

from experiments.rlcascade import cascade as C
from experiments.rlcascade import fast_score as FS
from experiments.rlcascade import where as CW
from experiments.rlpolicy import where as W
from experiments.rlprobe import maps as M
from experiments.rlprobe import score as RS

U5, THR = FS.UNIT5, FS.THR
DENSE = {'test': CW.TEST_L2, 'train': f'{CW.ROOT}/oof_l2'}
OUT = f'{W.ROOT}/sim'


def names_of(split):
    out = []
    for f in sorted(os.listdir(W.UNITS)):
        if f.endswith('.npz'):
            z = np.load(f'{W.UNITS}/{f}')
            if str(z['split']) == split and (split != 'train' or bool(z['trust'])) and z['tissue'].any():
                out.append(f[:-4])
    return out


def unit_of(rr, cc, hu, wu):
    return np.minimum(rr // U5, hu - 1) * wu + np.minimum(cc // U5, wu - 1)


def build(name, split):
    f = f'{OUT}/{name}.npz'
    if os.path.exists(f):
        return name
    z = dict(np.load(f'{W.UNITS}/{name}.npz'))
    if split == 'train':
        z.update(np.load(f'{W.UNITS}/oof/{name}.npz'))
    seg = np.load(f'{W.UNITS}/seg/{name}.npz')
    hu, wu = z['tissue'].shape
    gt, mpp = FS.gt_of(name)
    tt = C.tissue_tok(name)
    m5 = RS.as_loaded(np.where(tt, M.dense(f'{DENSE[split]}/{name}.npz'), 0).astype(np.float32), gt['shape'])
    t5 = RS.as_loaded(tt.astype(np.float32), gt['shape']) > 0
    lab, k = FS.nd.label(m5 >= THR, structure=FS.CS.S8)
    idx = np.flatnonzero(lab)
    rr, cc = np.unravel_index(idx, m5.shape)
    unit = unit_of(rr, cc, hu, wu)
    l, v = lab.ravel()[idx], m5.ravel()[idx]
    tp = (gt['mask'].ravel()[idx] > 0) if gt['mask'] is not None else np.zeros(len(idx), bool)
    ev = gt['eval_mask'].ravel()[idx] if gt['eval_mask'] is not None else np.zeros(len(idx), np.int32)
    order = np.lexsort((idx, -v, unit, l))
    key = l[order].astype(np.int64) * (hu * wu) + unit[order]
    first = np.r_[0, np.flatnonzero(np.diff(key)) + 1] if len(key) else np.zeros(0, np.int64)
    cnt = np.diff(np.r_[first, len(key)])
    rows = dict(cu_comp=l[order][first] - 1, cu_unit=unit[order][first], cu_n=cnt, cu_tp=np.add.reduceat(tp[order].astype(np.int64), first) if len(key) else cnt,
                cu_max=v[order][first], cu_lab=ev[order][first])
    by = np.lexsort((rows['cu_unit'], -rows['cu_max'], rows['cu_comp']))      # per component: strongest unit first
    rows = {a: b[by] for a, b in rows.items()}
    pad = lambda a: np.pad(a, ((0, max(0, hu * U5 - a.shape[0])), (0, max(0, wu * U5 - a.shape[1]))))[:hu * U5, :wu * U5]
    blocks = lambda a, fn: fn(pad(a).reshape(hu, U5, wu, U5), axis=(1, 3))
    n_tis = blocks(t5.astype(np.int32), np.sum)
    gt_tis = blocks((t5 & (gt['mask'] > 0)).astype(np.int32), np.sum) if gt['mask'] is not None else np.zeros((hu, wu), np.int32)
    paint_lab = np.zeros((hu, wu), np.int32)
    if gt['eval_mask'] is not None:
        tr, tc = np.nonzero(t5)
        u = unit_of(tr, tc, hu, wu)
        uu, fi = np.unique(u, return_index=True)       # first tissue pixel of each unit in raster order
        paint_lab.reshape(-1)[uu] = gt['eval_mask'][tr[fi], tc[fi]]
    n_lab = int(gt['eval_mask'].max()) if gt['eval_mask'] is not None else 0
    p3 = z['p3_unit'].astype(np.float32)
    rec = dict(tissue=z['tissue'], frac=z['frac'], bucket=z['bucket'], pg=1 - z['p6'][..., 0].astype(np.float32), p3=1 - p3[..., 0],
               p_core=np.where(p3.argmax(-1) == 1, p3[..., 1], 0.0).astype(np.float32), seg_frac=seg['seg_frac'], seg_edges=seg['seg_edges'],
               unit_max=blocks(m5, np.max).astype(np.float32), n_tis=n_tis, gt_tis=gt_tis, paint_lab=paint_lab, n_lab=n_lab,
               itcs=np.array(gt['itcs'], np.int64), G=int((gt['mask'] > 0).sum()) if gt['mask'] is not None else 0, tumour=gt['mask'] is not None,
               split=split, **rows)
    os.makedirs(OUT, exist_ok=True)
    np.savez(f'{f}.{os.getpid()}.npz', **rec)
    os.replace(f'{f}.{os.getpid()}.npz', f)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True)
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    names = names_of(a.split)
    with get_context('fork').Pool(a.workers) as pool:
        for i, n in enumerate(pool.imap_unordered(partial(build, split=a.split), names), 1):
            if i % 40 == 0:
                print(i, n, flush=True)
    print(a.split, len(names), 'done', flush=True)


if __name__ == '__main__':
    main()
