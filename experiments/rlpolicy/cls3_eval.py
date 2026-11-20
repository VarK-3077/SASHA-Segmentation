# Scores the crude CLS-token cascade (cls3.py): L2-segment every tissue unit whose CLS-MLP probability is >= t,
# nothing else (no L6 screen, painting or tracing). Seconds are the cost model's with every tissue unit read
# at L3. --split train prints the cross-fitted frontier over t in the simulator, picks t (highest Dice + FROC,
# ties to the faster one), and scores that pick and the supervised rule with the exact scorer. --split test
# plays the all-train MLP at the saved t, once, with the exact scorer.
# Usage: python -m experiments.rlpolicy.cls3_eval --split train ; then --split test

import argparse
import json
import pickle
import time

import numpy as np

from experiments.rlcascade import report as RP
from experiments.rlpolicy import exact as X
from experiments.rlpolicy import sim as S
from experiments.rlpolicy import where as W
from experiments.rlpolicy.cls3 import OUT
from experiments.rlprobe.cost import seconds

GRID = (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999)


def play(s, t):
    zoom = s.tissue & (s.c3 >= t)
    none = np.zeros_like(zoom)
    out = S.outcome(s, zoom, none)
    out.update(n_l3=s.n, n_l2=int(zoom.sum()), n_paint=0, zoom=zoom, paint=none, screened=s.tissue, sec=seconds(s.n, int(zoom.sum())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=('train', 'test'), required=True)
    ap.add_argument('--workers', type=int, default=20)
    a = ap.parse_args()
    slides = S.load(a.split)
    for s in slides:
        s.c3 = np.load(f'{OUT}/pred/{s.name}.npy')
    rule = [S.rule(s) for s in slides]
    fmt = lambda m: f"{m['sec']:.3f} s  Dice {m['dice']:.4f}  FROC {m['froc']:.4f}  L3 {m['l3']:.0f}  L2 {m['l2']:.0f}  FP {m['fp']:.2f}"
    pick_file = f'{OUT}/pick.json'
    lines = []
    if a.split == 'train':
        lines.append('rule (simulator)   ' + fmt(S.summarise(rule)))
        best = None
        for t in GRID:
            m = S.summarise([play(s, t) for s in slides])
            lines.append(f'crude t {t:<6}    {fmt(m)}')
            key = (round(m['dice'] + m['froc'], 4), -m['sec'])
            if best is None or key > best[0]:
                best = (key, t)
        t = best[1]
        json.dump(dict(t=t), open(pick_file, 'w'))
        lines.append(f'picked t = {t} (highest cross-fitted Dice + FROC)')
    else:
        t = json.load(open(pick_file))['t']
        open(f'{W.ROOT}/TEST_READS.txt', 'a').write(f"{time.strftime('%F %T')} crude CLS-MLP cascade t={t}\n")
    recs = [play(s, t) for s in slides]
    ex = X.score_routes(a.split, {s.name: {'rule': r, 'crude': c} for s, r, c in zip(slides, rule, recs)}, a.workers)
    label = {'crude': f'crude L3 CLS-MLP screen + L2 (t = {t})'}
    head = f"## {a.split}, {len(slides)} slides, exact scorer ({'cross-fitted MLPs' if a.split == 'train' else 'all-train MLP, test read once'}); seconds = cost model, all tissue units read at L3\n"
    text = '\n'.join([head, RP.table(ex, ['rule', 'crude'], label), '', RP.BOOT_HEAD, RP.boot_line(ex, 'crude', label['crude']), ''] + lines) + '\n'
    open(f'{OUT}/{a.split}.md', 'w').write(text)
    pickle.dump(ex, open(f'{OUT}/{a.split}.pkl', 'wb'))
    print(text)


if __name__ == '__main__':
    main()
