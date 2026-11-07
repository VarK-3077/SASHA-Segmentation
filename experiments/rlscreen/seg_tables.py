# What the real level-2 segmenter does on each unit, from full maps: predicted tumour share and border
# strips (the outcome a policy sees), true positive and false positive cells (what scoring sees), and per
# unit and non-ITC lesion how many lesion cells the map recovers. Beside the slide tables as seg/<slide>.npz.
# Usage: python -m experiments.rlscreen.seg_tables --maps DIR [DIR ...]

import argparse
import glob
import os
from multiprocessing import get_context

import numpy as np

from experiments.lowsep import paths as LP
from experiments.rlscreen import paths as P
from utils.segmentation_metrics import evaluation_mask, itc_labels, rasterize_mask, read_xml_polygons

THR, G = 0.4, 32      # the head's threshold; tokens per unit side
K = (LP.UNIT // LP.MASK_DS) // G   # mask cells per token side
KM = LP.UNIT // LP.MASK_DS


def one(path):
    name = os.path.basename(path)[:-4]
    out = f'{P.units_dir("kaiko_vitl14")}/seg/{name}.npz'
    if os.path.exists(out):
        return
    z = np.load(LP.label_npz(name))
    hu, wu = z['tissue'].shape
    mpp, dims = float(z['mpp']), tuple(int(v) for v in z['dims'])
    p = np.load(path)['probs'].astype(np.float32)
    pred = np.pad(p >= THR, ((0, max(0, hu * G - p.shape[0])), (0, max(0, wu * G - p.shape[1]))))[:hu * G, :wu * G]
    u = pred.reshape(hu, G, wu, G).astype(np.float32)
    seg_frac = u.mean((1, 3))
    seg_edges = np.stack([u[:, 0].mean(-1), u[:, -1].mean(-1), u[:, :, :, 0].mean(1), u[:, :, :, -1].mean(1)], -1)
    pc = np.repeat(np.repeat(pred, K, 0), K, 1)
    rec = dict(seg_frac=seg_frac.astype(np.float32), seg_edges=seg_edges.astype(np.float32))
    xml = LP.xml_path(name)
    tum, exc = read_xml_polygons(xml) if os.path.exists(xml) else ([], [])
    empty = np.zeros(0, np.int64)
    if not tum:
        fp = pc.reshape(hu, KM, wu, KM).sum((1, 3)).astype(np.float32) / KM ** 2
        rec.update(tp=np.zeros((hu, wu), np.float32), fp=fp, gt=np.zeros((hu, wu), np.float32), lu_unit=empty, lu_lesion=empty, lu_tp=empty, lesion_px=empty)
    else:
        m = rasterize_mask(tum, exc, dims, LP.MASK_DS)
        m = np.pad(m, ((0, max(0, hu * KM - m.shape[0])), (0, max(0, wu * KM - m.shape[1]))))[:hu * KM, :wu * KM] > 0
        area = lambda x: x.reshape(hu, KM, wu, KM).sum((1, 3)).astype(np.float32) / KM ** 2
        rec.update(tp=area(pc & m), fp=area(pc & ~m), gt=area(m))
        ev = evaluation_mask(m.astype(np.uint8), mpp, LP.MASK_DS)
        n = int(ev.max())
        keep = np.ones(n + 1, bool); keep[0] = False
        if n:
            keep[np.array(itc_labels(ev, mpp, LP.MASK_DS), dtype=int)] = False
        lab = ev * m
        lab = lab * keep[lab]
        rr, cc = np.nonzero(lab & pc)
        key = ((rr // KM) * wu + cc // KM).astype(np.int64) * (n + 1) + lab[rr, cc]
        uk, cnt = np.unique(key, return_counts=True)
        rec.update(lu_unit=uk // (n + 1), lu_lesion=uk % (n + 1), lu_tp=cnt, lesion_px=np.bincount(lab[lab > 0], minlength=n + 1))
    np.savez(out, **rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--maps', nargs='+', required=True)
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    os.makedirs(f'{P.units_dir("kaiko_vitl14")}/seg', exist_ok=True)
    fs = [f for d in a.maps for f in glob.glob(f'{d}/*.npz')]
    with get_context('fork').Pool(a.workers) as pool:
        list(pool.imap_unordered(one, fs, chunksize=4))
    print(len(fs), 'slides', flush=True)


if __name__ == '__main__':
    main()
