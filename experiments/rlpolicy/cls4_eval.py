# Scores the crude four-class CLS-token cascade (cls3.py train4): every tissue unit is read at L3, its CLS token
# goes through a two-layer MLP with four classes (normal / core / boundary / small); units called core are painted,
# units called boundary or small are segmented at L2, the rest is left. No L6 screen, no border tracing.
# Two decision rules: "argmax" (the class with the highest probability, nothing chosen) and "t" (act when
# 1 - p_normal >= t, the kind by the larger of the tumour classes; t picked on cross-fitted train as for the
# two-class crude baseline: highest Dice + FROC, ties to the faster). --split train prints the simulator frontier,
# saves the pick and scores both rules and the supervised rule with the exact scorer; --split test plays the
# all-train MLP once per rule with the exact scorer.
# Usage: python -m experiments.rlpolicy.cls4_eval --split train ; then --split test

import argparse
import json
import pickle
import time

import numpy as np

from experiments.rlcascade import report as RP
from experiments.rlpolicy import exact as X
from experiments.rlpolicy import sim as S
from experiments.rlpolicy import where as W
from experiments.rlpolicy.cls3 import OUT4

GRID = (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)


def play(s, t):
    p = s.c4
    kind = p[..., 1:].argmax(-1) + 1
    act = s.tissue & ((p.argmax(-1) != 0) if t is None else (1 - p[..., 0] >= t))
    paint, zoom = act & (kind == 1), act & (kind != 1)
    out = S.outcome(s, zoom, paint)
    out.update(n_l3=s.n, n_l2=int(zoom.sum()), n_paint=int(paint.sum()), zoom=zoom, paint=paint, screened=s.tissue, sec=S.seconds(s.n, int(zoom.sum())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=('train', 'test'), required=True)
    ap.add_argument('--workers', type=int, default=20)
    a = ap.parse_args()
    slides = S.load(a.split)
    for s in slides:
        s.c4 = np.load(f'{OUT4}/pred/{s.name}.npy')
    rule = [S.rule(s) for s in slides]
    fmt = lambda m: f"{m['sec']:.3f} s  Dice {m['dice']:.4f}  FROC {m['froc']:.4f}  L3 {m['l3']:.0f}  L2 {m['l2']:.0f}  paint {m['paint']:.1f}  FP {m['fp']:.2f}"
    pick_file = f'{OUT4}/pick.json'
    lines = []
    if a.split == 'train':
        lines.append('rule (simulator)   ' + fmt(S.summarise(rule)))
        lines.append('crude4 argmax      ' + fmt(S.summarise([play(s, None) for s in slides])))
        best = None
        for t in GRID:
            m = S.summarise([play(s, t) for s in slides])
            lines.append(f'crude4 t {t:<6}   {fmt(m)}')
            key = (round(m['dice'] + m['froc'], 4), -m['sec'])
            if best is None or key > best[0]:
                best = (key, t)
        t = best[1]
        json.dump(dict(t=t), open(pick_file, 'w'))
        lines.append(f'picked t = {t} (highest cross-fitted Dice + FROC)')
    else:
        t = json.load(open(pick_file))['t']
        open(f'{W.ROOT}/TEST_READS.txt', 'a').write(f"{time.strftime('%F %T')} crude four-class CLS-MLP cascade: argmax, and t={t}\n")
    routes = {s.name: {'rule': r, 'crude4_argmax': play(s, None), 'crude4_t': play(s, t)} for s, r in zip(slides, rule)}
    ex = X.score_routes(a.split, routes, a.workers)
    label = {'crude4_argmax': 'crude L3 CLS-MLP, 4 classes, argmax: paint core, L2 on boundary + small', 'crude4_t': f'same, act when 1 - p_normal >= {t}'}
    head = f"## {a.split}, {len(slides)} slides, exact scorer ({'cross-fitted MLPs' if a.split == 'train' else 'all-train MLP, test read once'}); seconds = cost model, all tissue units read at L3\n"
    keys = ['rule', 'crude4_argmax', 'crude4_t']
    text = '\n'.join([head, RP.table(ex, keys, label), '', RP.BOOT_HEAD] + [RP.boot_line(ex, k, label[k]) for k in keys[1:]] + [''] + lines) + '\n'
    open(f'{OUT4}/{a.split}.md', 'w').write(text)
    pickle.dump(ex, open(f'{OUT4}/{a.split}.pkl', 'wb'))
    print(text)


if __name__ == '__main__':
    main()
