# Second number of the consensus policy: the L6 veto v6 (a unit is read at L3 only if the supervised L6 head
# passes it AND the second L6 head scores it >= v6), fitted on top of the L3 veto that vsel.py chose. Same rule:
# on the slides it is fitted on, Dice and FROC may not fall below the supervised rule's; take the largest v6
# whose every smaller grid value also keeps both, times a safety factor kappa. Cross-fitted with the folds' own
# L3 vetoes, so a slide is always played by numbers fitted without it.
# Usage: python -m experiments.rlpolicy.vsel6 --tok3 t1 --tok6 g0 --base vsel_t1 --run t_top16:0.5 --out NAME

import argparse
import os
import pickle

import numpy as np

from experiments.rlpolicy import policy as P
from experiments.rlpolicy import sim as S
from experiments.rlpolicy import where as W
from experiments.rlpolicy.vsel import KAPPA, fit

GRID = np.round(np.logspace(-3, np.log10(0.9), 31), 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tok3', required=True)
    ap.add_argument('--tok6', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    slides = S.load('train', a.tok3, tok6=a.tok6)
    base = pickle.load(open(f'{W.ROOT}/search/{a.base}/res.pkl', 'rb'))['runs'][a.run]
    key, kap3 = base['key'], float(a.run.split(':')[1])
    v3_all, v3_fold = float(base['theta'].veto), [kap3 * v for v in base['v_fold']]
    folds = W.folds()
    fold = np.array([folds[s.name] for s in slides])
    rule = [S.rule(s) for s in slides]
    tum = [i for i, s in enumerate(slides) if s.tumour]
    lost = lambda recs: sum(max(0, rule[i]['found'] - recs[i]['found']) for i in tum)
    below = lambda recs: sum(recs[i]['dice'] < rule[i]['dice'] - 1e-9 for i in tum)
    fmt = lambda m: f"{m['sec']:.3f} s  Dice {m['dice']:.4f}  FROC {m['froc']:.4f}  L3 {m['l3']:.0f}  L2 {m['l2']:.0f}  FP {m['fp']:.2f}"
    th = lambda v3, v6: P.RULE._replace(veto=float(v3), veto6=float(v6))
    print('rule            ', fmt(S.summarise(rule)))
    print('L3 veto only    ', fmt(S.summarise([P.play(s, th(v3_all, 0), key) for s in slides])))
    res = dict(names=[s.name for s in slides], rule=rule, runs={})
    for gk in ('g_max', 'g_mean'):
        for s in slides:
            s.g_key = gk
        print(f'--- {gk}: in-sample curve (L3 veto {v3_all:.5f})')
        plays = {v: [P.play(s, th(v3_all, v), key) for s in slides] for v in GRID}
        for v in GRID:
            print(f'  v6 {v:.4f}  {fmt(S.summarise(plays[v]))}  lost {lost(plays[v])}  Dice below rule on {below(plays[v])}')
        v_all = fit([(v, S.summarise(plays[v])) for v in GRID], S.summarise(rule))
        v_fold = []
        for k in range(W.N_FOLDS):
            tr = np.flatnonzero(fold != k)
            v_fold.append(fit([(v, S.summarise([P.play(slides[i], th(v3_fold[k], v), key) for i in tr])) for v in GRID], S.summarise([rule[i] for i in tr])))
        print(f'  largest safe v6: all slides {v_all}, folds {v_fold}')
        for kap in KAPPA:
            cv = [P.play(s, th(v3_fold[fold[i]], kap * v_fold[fold[i]]), key) for i, s in enumerate(slides)]
            print(f'  kappa {kap}: cross-fitted {fmt(S.summarise(cv))}  lost {lost(cv)}  Dice below rule on {below(cv)}  | all-train v6 = {kap * v_all:.5f}')
            res['runs'][f'{gk}:{kap}'] = dict(cv=cv, theta=th(v3_all, kap * v_all), v_fold=v_fold, key=key, g_key=gk)
    os.makedirs(f'{W.ROOT}/search/{a.out}', exist_ok=True)
    pickle.dump(res, open(f'{W.ROOT}/search/{a.out}/res.pkl', 'wb'))


if __name__ == '__main__':
    main()
