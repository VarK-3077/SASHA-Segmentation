# How much is adaptive screening worth in the coarse design (no whole-slide level-3 pass)? Units are
# screened at level 3 in order of the level-6 p(tumour); screened units with level-3 p >= tz are
# zoomed (ideal segmenter, no border following, so Dice is below the teacher's). The decision is how
# far down the level-6 ranking to screen on each slide:
#   fixed screen tau   screen every unit with level-6 p >= ts (the current rule)
#   evidence h         screen in batches; stop after a batch in which < h of the units are zoom-worthy
#   hindsight          per slide the screening prefix maximising Dice - lam * seconds share (upper bound)
# Seconds use the per-unit costs measured in the real coarse runs: tissue 0.56, level-6 pass 0.12,
# level-3 screen 2.2 ms / unit, level-2 zoom 9.7 ms / unit.
# Usage: python -m experiments.rlscreen.coarse_headroom [--split test] [--tz 0.7]

import argparse
import os

import numpy as np

from experiments.rlscreen import paths as P

T0, SCREEN, ZOOM = 0.56 + 0.12, 0.0022, 0.0097


def slides(split, backbone, g, tz):
    d = P.units_dir(backbone)
    out = []
    for f in sorted(os.listdir(d)):
        if not f.endswith('.npz'):
            continue
        z = np.load(f'{d}/{f}')
        if str(z['split']) != split or not z['tissue'].any():
            continue
        t = z['tissue']
        pg = 1 - z[f'p{g}'][..., 0].astype(np.float32)[t]
        p3 = 1 - z['p3_unit'][..., 0].astype(np.float32)[t]
        frac = z['frac'].astype(np.float64)[t]
        o = np.argsort(-pg, kind='stable')
        pg, p3, frac = pg[o], p3[o], frac[o]
        tot = float(z['frac'].astype(np.float64)[t | (z['frac'] > 0)].sum())
        zoom = p3 >= tz
        a = np.concatenate([[0.0], np.cumsum(frac * zoom) / tot]) if tot > 0 else np.zeros(len(pg) + 1)
        nz = np.concatenate([[0], np.cumsum(zoom)])
        out.append(dict(pg=pg, zoom=zoom, dice=2 * a / (1 + a) if tot > 0 else np.ones(len(pg) + 1),
                        nz=nz, tumour=tot > 0, n=len(pg)))
    return out


def secs(s, k):
    return T0 + k * SCREEN + s['nz'][k] * ZOOM


def point(S, ks):
    dice = np.mean([s['dice'][k] for s, k in zip(S, ks) if s['tumour']])
    return (dice, np.mean([k / s['n'] for s, k in zip(S, ks)]), np.mean([s['nz'][k] / s['n'] for s, k in zip(S, ks)]),
            np.mean([secs(s, k) for s, k in zip(S, ks)]), np.mean([secs(s, k) for s, k in zip(S, ks) if not s['tumour']]))


def evidence_k(s, h, batch):
    b = max(4, int(round(batch * s['n'])))
    k = 0
    while k < s['n']:
        nb = min(b, s['n'] - k)
        rate = s['zoom'][k:k + nb].mean()
        k += nb
        if rate < h:
            break
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test')
    ap.add_argument('--backbone', default='kaiko_vitl14')
    ap.add_argument('--g_level', type=int, default=6)
    ap.add_argument('--tz', type=float, default=0.7)
    ap.add_argument('--batch', type=float, default=0.03)
    a = ap.parse_args()
    S = slides(a.split, a.backbone, a.g_level, a.tz)
    print(f'{a.split}: {len(S)} slides, {sum(s["tumour"] for s in S)} tumour; L{a.g_level} ranking, zoom tau {a.tz}')
    print('| rule | setting | ideal Dice (tumour) | screened | zoomed | est. s (all) | est. s (normal) |\n|---|---|---|---|---|---|---|')
    for ts in (0.1, 0.3, 0.5, 0.7, 0.9):
        print('| fixed screen tau | %g | %.3f | %.3f | %.3f | %.2f | %.2f |' % ((ts,) + point(S, [int((s['pg'] >= ts).sum()) for s in S])))
    for h in (0.02, 0.05, 0.1, 0.2, 0.3):
        print('| evidence | h=%g | %.3f | %.3f | %.3f | %.2f | %.2f |' % ((h,) + point(S, [evidence_k(s, h, a.batch) for s in S])))
    for lam in (0.003, 0.01, 0.03, 0.1, 0.3):
        ks = [int(np.argmax(s['dice'] - lam * np.array([secs(s, k) for k in range(s['n'] + 1)]))) if s['tumour'] else 0 for s in S]
        print('| hindsight | lam=%g | %.3f | %.3f | %.3f | %.2f | %.2f |' % ((lam,) + point(S, ks)))


if __name__ == '__main__':
    main()
