# How much is per-slide stopping worth? Full design, ideal segmenter. Units are zoomed in order of
# the supervised level-3 p(tumour); the only decision is how far down that ranking to go on each
# slide. Three frontiers of mean per-slide ideal Dice (tumour slides) vs mean zoom share (all slides):
#   fixed tau     go down to p >= tau on every slide (the current rule, one global threshold)
#   evidence h    zoom the ranking in batches, stop after a batch whose tumour hit rate is < h
#                 (uses only what the segmenter has shown so far: a deployable rule, no learning)
#   hindsight     per slide the prefix maximising Dice - lam * zoom share, knowing the outcome
#                 (an upper bound for any stopping rule on this ranking; normal slides stop at 0)
# Usage: python -m experiments.rlscreen.stop_headroom [--split test] [--batch 0.02]

import argparse
import os

import numpy as np

from experiments.lowsep import paths as LP
from experiments.rlscreen import paths as P


def slides(split, backbone):
    d = P.units_dir(backbone)
    out = []
    for f in sorted(os.listdir(d)):
        if not f.endswith('.npz'):
            continue
        z = np.load(f'{d}/{f}')
        if str(z['split']) != split or not z['tissue'].any():
            continue
        t = z['tissue']
        p = 1 - z['p3'][..., 0].astype(np.float32)[t]
        frac = z['frac'].astype(np.float64)[t]
        order = np.argsort(-p, kind='stable')
        p, frac = p[order], frac[order]
        tot = float(z['frac'].astype(np.float64)[t | (z['frac'] > 0)].sum())
        a = np.concatenate([[0.0], np.cumsum(frac) / tot]) if tot > 0 else np.zeros(len(p) + 1)
        dice = 2 * a / (1 + a) if tot > 0 else np.ones(len(p) + 1)
        out.append(dict(p=p, hit=frac > 0, dice=dice, tumour=tot > 0, n=len(p)))
    return out


def point(S, ks):
    tum = [s for s in S if s['tumour']]
    dice = np.mean([s['dice'][k] for s, k in zip(S, ks) if s['tumour']]) if tum else float('nan')
    zoom = np.mean([k / s['n'] for s, k in zip(S, ks)])
    zoom_n = np.mean([k / s['n'] for s, k in zip(S, ks) if not s['tumour']])
    return dice, zoom, zoom_n


def evidence_k(s, h, batch, floor):
    b = max(4, int(round(batch * s['n'])))
    k = 0
    while k < s['n'] and s['p'][k] >= floor:
        nb = min(b, s['n'] - k)
        rate = s['hit'][k:k + nb].mean()
        k += nb
        if rate < h:
            break
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test')
    ap.add_argument('--backbone', default='kaiko_vitl14')
    ap.add_argument('--batch', type=float, default=0.02, help='evidence rule batch, share of tissue units')
    ap.add_argument('--floor', type=float, default=0.02, help='evidence rule never zooms below this p')
    a = ap.parse_args()
    S = slides(a.split, a.backbone)
    print(f'{a.split}: {len(S)} slides, {sum(s["tumour"] for s in S)} tumour')
    print('| rule | setting | ideal Dice (tumour) | zoom (all) | zoom (normal slides) |\n|---|---|---|---|---|')
    for tau in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
        d, z, zn = point(S, [int((s['p'] >= tau).sum()) for s in S])
        print(f'| fixed tau | {tau:g} | {d:.3f} | {z:.3f} | {zn:.3f} |')
    for h in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
        d, z, zn = point(S, [evidence_k(s, h, a.batch, a.floor) for s in S])
        print(f'| evidence | h={h:g} | {d:.3f} | {z:.3f} | {zn:.3f} |')
    for lam in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        ks = [int(np.argmax(s['dice'] - lam * np.arange(s['n'] + 1) / s['n'])) if s['tumour'] else 0 for s in S]
        d, z, zn = point(S, ks)
        print(f'| hindsight | lam={lam:g} | {d:.3f} | {z:.3f} | {zn:.3f} |')


if __name__ == '__main__':
    main()
