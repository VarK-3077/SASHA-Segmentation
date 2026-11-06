# What would selecting core units directly be worth? Coarse design, fixed rule (level-6 p >= ts
# screened at level 3, screened units with level-3 p(tumour) >= tz routed), no border following.
# A routed unit the per-unit level-3 head calls core with p(core) >= tc is painted all tumour (no
# level-2 zoom); the rest are zoomed (ideal segmenter: exact ground truth). Per-slide Dice counts a
# painted unit's non-tumour pixels as false positives. Seconds from the real coarse per-unit costs.
# Usage: python -m experiments.rlscreen.core_select [--split test]

import argparse
import os

import numpy as np

from experiments.rlscreen import paths as P

T0, SCREEN, ZOOM, PAINT = 0.68, 0.0022, 0.0097, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test')
    ap.add_argument('--backbone', default='kaiko_vitl14')
    ap.add_argument('--g_level', type=int, default=6)
    ap.add_argument('--ts', type=float, default=0.5)
    ap.add_argument('--tz', type=float, default=0.7)
    a = ap.parse_args()
    d = P.units_dir(a.backbone)
    S = []
    for f in sorted(os.listdir(d)):
        if not f.endswith('.npz'):
            continue
        z = np.load(f'{d}/{f}')
        if str(z['split']) != a.split or not z['tissue'].any():
            continue
        t = z['tissue']
        pg = 1 - z[f'p{a.g_level}'][..., 0].astype(np.float32)
        p3 = z['p3_unit'].astype(np.float32)
        frac = z['frac'].astype(np.float64)
        routed = t & (pg >= a.ts) & (1 - p3[..., 0] >= a.tz)
        S.append(dict(routed=routed, p3=p3, frac=frac, gt=float(frac[t | (frac > 0)].sum()), n=int(t.sum()),
                      bucket=z['bucket'], screened=int((t & (pg >= a.ts)).sum())))
    print(f'{a.split}: {len(S)} slides; screen L{a.g_level} p >= {a.ts}, route L3 p >= {a.tz}')
    print('| select core if p_core >= | ideal Dice (tumour) | zoomed | painted | painted that are true core | est. s |\n|---|---|---|---|---|---|')
    for tc in (None, 0.99, 0.95, 0.9, 0.8, 0.7, 0.5):
        dice, zm, pt, true_core, sec = [], [], [], [], []
        for s in S:
            r = s['routed']
            paint = r & (s['p3'].argmax(-1) == 1) & (s['p3'][..., 1] >= tc) if tc is not None else np.zeros_like(r)
            zoom = r & ~paint
            inter = s['frac'][zoom].sum() + s['frac'][paint].sum()
            pred = s['frac'][zoom].sum() + paint.sum()
            if s['gt'] > 0:
                dice.append(2 * inter / (pred + s['gt']))
            zm.append(zoom.sum() / s['n']); pt.append(paint.sum() / s['n'])
            if paint.any():
                true_core.append((s['bucket'][paint] == 1).mean())
            sec.append(T0 + s['screened'] * SCREEN + zoom.sum() * ZOOM)
        lab = 'never (current)' if tc is None else f'{tc:g}'
        print(f'| {lab} | {np.mean(dice):.3f} | {np.mean(zm):.3f} | {np.mean(pt):.3f} | '
              f'{np.mean(true_core) if true_core else float("nan"):.3f} | {np.mean(sec):.2f} |')


if __name__ == '__main__':
    main()
