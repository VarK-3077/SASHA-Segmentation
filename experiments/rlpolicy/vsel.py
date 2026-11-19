# Picks the consensus policy's one number, the token-head veto v (a unit is segmented only if the supervised
# cascade wants it AND the token-level head scores it >= v), from the reward the simulator gives, with a hard
# floor: on the slides it is fitted on, cleaned Dice and FROC may not fall below the supervised rule's. The fit
# takes the largest v such that every smaller v on the grid also keeps both, then multiplies it by a safety
# factor kappa. Cross-fitted: each fold's slides are played with the v fitted on the other folds. Prints the
# in-sample curve, the cross-fitted row for every (token score, kappa), and saves the routes for evalx.
# Usage: python -m experiments.rlpolicy.vsel --tok3 t0 --out NAME

import argparse
import os
import pickle

import numpy as np

from experiments.rlpolicy import policy as P
from experiments.rlpolicy import sim as S
from experiments.rlpolicy import where as W

GRID = np.round(np.logspace(-3, np.log10(0.95), 31), 5)
KAPPA = (1.0, 0.5, 0.25)
KEYS = ('t_top4', 't_top16', 't_max')


QGRID = np.r_[-0.02, -0.01, -0.005, -0.002, 0.0, np.round(np.logspace(-4, np.log10(0.3), 26), 6)]


def theta(v, key='t'):
    """Consensus policy with veto v; for key 'q' the value policy: segment where Q >= v, whatever the supervised head says."""
    if key == 'q':
        return P.RULE._replace(p3a=1e-9, p3b=1e-9, veto=float(v))
    return P.RULE._replace(veto=float(v))


def fit(curve, base, start=0.0):
    """curve: [(v, summary)] ascending in v -> largest safe v (`start` if even the smallest breaks the floor)."""
    best = start
    for v, m in curve:
        if m['dice'] < base['dice'] - 1e-9 or m['froc'] < base['froc'] - 1e-9:
            break
        best = v
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tok3', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fixed', type=float, nargs='*', default=[], help='also save the policy at these values, with nothing fitted')
    ap.add_argument('--q', default=None, help='tag of a value policy (qpol.py): select its price c instead of a veto')
    a = ap.parse_args()
    slides = S.load('train', a.tok3)
    keys, grid = KEYS, GRID
    if a.q:
        for s in slides:
            s.q = np.load(f'{W.ROOT}/q/{a.q}/{s.name}.npy')
        keys, grid = ('q',), QGRID
    folds = W.folds()
    fold = np.array([folds[s.name] for s in slides])
    rule = [S.rule(s) for s in slides]
    tum = [i for i, s in enumerate(slides) if s.tumour]
    lost = lambda recs: sum(max(0, rule[i]['found'] - recs[i]['found']) for i in tum)
    below = lambda recs: sum(recs[i]['dice'] < rule[i]['dice'] - 1e-9 for i in tum)
    fmt = lambda m: f"{m['sec']:.3f} s  Dice {m['dice']:.4f}  FROC {m['froc']:.4f}  L2 {m['l2']:.0f}  FP {m['fp']:.2f}"
    print('rule            ', fmt(S.summarise(rule)))
    res = dict(names=[s.name for s in slides], rule=rule, runs={})
    for key in keys:
        plays = {v: [P.play(s, theta(v, key), key) for s in slides] for v in grid}
        print(f'--- {key}: in-sample curve')
        for v in grid:
            print(f'  v {v:.4f}  {fmt(S.summarise(plays[v]))}  lost {lost(plays[v])}  Dice below rule on {below(plays[v])}')
        sub = lambda v, idx: S.summarise([plays[v][i] for i in idx])
        v_all = fit([(v, S.summarise(plays[v])) for v in grid], S.summarise(rule), grid[0])
        v_fold = []
        for k in range(W.N_FOLDS):
            tr = np.flatnonzero(fold != k)
            v_fold.append(fit([(v, sub(v, tr)) for v in grid], S.summarise([rule[i] for i in tr]), grid[0]))
        print(f'  largest safe v: all slides {v_all}, folds {v_fold}')
        for kap in KAPPA:
            back = (lambda v: kap * v if v > 0 else v)
            cv = [P.play(s, theta(back(v_fold[fold[i]]), key), key) for i, s in enumerate(slides)]
            m = S.summarise(cv)
            print(f'  kappa {kap}: cross-fitted {fmt(m)}  lost {lost(cv)}  Dice below rule on {below(cv)}  | all-train v = {back(v_all):.5f}')
            res['runs'][f'{key}:{kap}'] = dict(cv=cv, theta=theta(back(v_all), key), v_fold=v_fold, key=key)
        for v in a.fixed:
            cv = [P.play(s, theta(v, key), key) for s in slides]
            print(f'  fixed {v}: {fmt(S.summarise(cv))}  lost {lost(cv)}  Dice below rule on {below(cv)}')
            res['runs'][f'{key}:fixed{v}'] = dict(cv=cv, theta=theta(v, key), key=key)
    os.makedirs(f'{W.ROOT}/search/{a.out}', exist_ok=True)
    pickle.dump(res, open(f'{W.ROOT}/search/{a.out}/res.pkl', 'wb'))


if __name__ == '__main__':
    main()
