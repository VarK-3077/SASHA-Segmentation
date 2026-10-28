# Turns the 4-bucket probe into a routing decision and scores it with an ideal segmenter.
# Every 1024 px tile of every test slide is routed: zoom (predicted boundary or small; the
# ideal segmenter returns the ground-truth mask there), select (predicted core; the whole
# tile becomes tumour) or leave (predicted normal; the tile stays empty). A second rule also
# zooms predicted-normal tiles whose normal probability is below a threshold. Dice is computed
# from the tiles' tumour fractions (level-4 raster), per slide and pooled.
#
# The probe is the separability study's seed-0 MLP on Kaiko ViT-L/14 level-2 embeddings,
# retrained on the same stratified train sample (its weights were not saved), then applied
# to the full-slide embeddings extracted for the RL study (same crop read and encoder).

import json
import os
import sys

import h5py
import numpy as np
import torch

from experiments.sep import paths as P
from experiments.sep.probe import fit, load

L2FEATS = '/data2/venkatavks/rlbin/l2feats'
ZOOM, SELECT, LEAVE = 0, 1, 2
ACT = ('zoom', 'select', 'leave')
THRESHOLDS = (0.5, 0.9, 0.99, 0.999)


def ideal_action(bucket):
    return np.where(bucket == 0, LEAVE, np.where(bucket == 1, SELECT, ZOOM))


def route(prob, tau=None):
    pred = prob.argmax(1)
    act = np.where(pred == 0, LEAVE, np.where(pred == 1, SELECT, ZOOM))
    if tau is not None:
        act[(pred == 0) & (prob[:, 0] < tau)] = ZOOM
    return act


def slide_dice(act, frac):
    inter = frac[act == ZOOM].sum() + frac[act == SELECT].sum()
    pred = frac[act == ZOOM].sum() + (act == SELECT).sum()
    gt = frac.sum()
    return 1.0 if pred + gt == 0 else 2 * inter / (pred + gt), inter, pred, gt


def score(rows, acts):
    per, I, Pd, G = [], 0, 0, 0
    for r, a in zip(rows, acts):
        d, i, p, g = slide_dice(a, r['frac'])
        per.append((d, g > 0))
        I, Pd, G = I + i, Pd + p, G + g
    tum = [d for d, t in per if t]
    nor = [d for d, t in per if not t]
    n = sum(len(a) for a in acts)
    counts = np.bincount(np.concatenate(acts), minlength=3)
    return {'dice_tumour': float(np.mean(tum)), 'dice_all': float(np.mean([d for d, _ in per])),
            'normal_clean': float(np.mean(nor)), 'dice_pooled': float(2 * I / (Pd + G)),
            'zoom': int(counts[0]), 'select': int(counts[1]), 'leave': int(counts[2]),
            'zoom_frac': float(counts[0] / n)}


def table(rows, acts):
    t = np.zeros((3, 3), np.int64)
    for r, a in zip(rows, acts):
        np.add.at(t, (ideal_action(r['bucket']), a), 1)
    return t


def main(device):
    X, z = load(2, 'kaiko_vitl14')
    y = z['bucket'].astype(np.int64)
    tr, va = z['split'] == 'train', z['split'] == 'val'
    net, val_mr, _ = fit('mlp', X, y, tr, va, device, seed=0)
    # load() L2-normalises then standardises with train-sample statistics; reproduce it for full slides
    with h5py.File(P.feats_h5(2), 'r') as f:
        raw = f['kaiko_vitl14'][:].astype(np.float32)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8
    mu, sd = raw[tr].mean(0), raw[tr].std(0) + 1e-6

    split = json.load(open(P.SPLIT_JSON))
    rows = []
    for name in sorted(split['test_names']):
        f = os.path.join(L2FEATS, name + '.npz')
        lab = np.load(os.path.join(P.LABEL_DIR, name + '.npz'))
        feat = np.load(f)['feat'].astype(np.float32)
        assert len(feat) == len(lab['L2_frac']), name
        feat /= np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8
        with torch.no_grad():
            prob = torch.softmax(net(torch.tensor((feat - mu) / sd, device=device)), 1).cpu().numpy()
        rows.append({'name': name, 'prob': prob, 'bucket': lab['L2_bucket'].astype(np.int64),
                     'frac': lab['L2_frac'].astype(np.float64)})

    out = {'val_macro_recall': val_mr, 'n_slides': len(rows), 'n_tiles': int(sum(len(r['frac']) for r in rows))}
    ideal = [ideal_action(r['bucket']) for r in rows]
    out['ideal'] = score(rows, ideal)
    runs = {'argmax': [route(r['prob']) for r in rows]}
    for tau in THRESHOLDS:
        runs[f'zoom if p_normal < {tau}'] = [route(r['prob'], tau) for r in rows]
    for k, acts in runs.items():
        out[k] = score(rows, acts)
        out[k]['table'] = table(rows, acts).tolist()
    json.dump(out, open(os.path.join(P.ROOT, 'route_dice.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items()}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'cuda:0')
