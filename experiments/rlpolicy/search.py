# Direct policy search for the staged policy: cross-entropy method over its ten numbers, maximising the
# reward a set of slides gives in the simulator, R = Dice (tumour slides) + FROC - lam * seconds per slide.
# Thresholds are searched in logit space and the stage-1 budget in log space, starting from the supervised
# rule's values. cross_fit searches once per fold on the other folds and plays each slide with the policy
# that did not see it; fit_all searches on every slide (the policy the test slides get).
# Usage: python -m experiments.rlpolicy.search --tok3 t0 --heads t_top4 --lams 0.02 0.05 0.1 --out NAME

import argparse
import json
import os
import pickle
from functools import partial
from multiprocessing import get_context

import numpy as np

from experiments.rlpolicy import policy as P
from experiments.rlpolicy import sim as S
from experiments.rlpolicy import where as W

SLIDES = None
FIX = {}


def to_x(th):
    """Thresholds in logit space, the stage-1 budget and the veto in log space."""
    t = np.asarray(th, np.float64)
    p = np.clip(t[:8] / 1.02, 1e-4, 1 - 1e-4)
    return np.r_[np.log(p / (1 - p)), np.log(t[8]), np.log(max(t[9], 1e-4))]


def to_theta(x):
    return P.clip(np.r_[1.02 / (1 + np.exp(-x[:8])), np.exp(np.clip(x[8], 0, 25)), np.exp(np.clip(x[9], -9.3, 0))])


def reward(recs, lam):
    m = S.summarise(recs)
    return m['dice'] + m['froc'] - lam * m['sec'], m


def value(x, idx, lam, heads):
    recs = [P.play(SLIDES[i], to_theta(x), heads) for i in idx]
    return reward(recs, lam)[0]


def cem(idx, lam, heads, seed, iters=30, pop=128, elite=16, pool=None):
    rng = np.random.default_rng(seed)
    start = P.RULE._replace(**{**dict(t3a=0.5, t3b=0.5, ev=0.9, k1=200.0, veto=0.005), **FIX})
    mu, sd = to_x(start), np.r_[np.full(8, 1.5), 1.0, 1.5]
    sd[[P.Theta._fields.index(k) for k in FIX]] = 0.0
    floor = np.where(sd > 0, 0.02, 0.0)
    best = (value(mu, idx, lam, heads), mu)
    f = partial(value, idx=idx, lam=lam, heads=heads)
    for _ in range(iters):
        xs = mu + sd * rng.standard_normal((pop, len(mu)))
        vs = np.array(pool.map(f, xs) if pool else [f(x) for x in xs])
        top = np.argsort(-vs)[:elite]
        if vs[top[0]] > best[0]:
            best = (float(vs[top[0]]), xs[top[0]])
        mu, sd = xs[top].mean(0), xs[top].std(0) + floor
    return to_theta(mu), to_theta(best[1])


def main():
    global SLIDES, FIX
    ap = argparse.ArgumentParser()
    ap.add_argument('--tok3', default=None)
    ap.add_argument('--heads', default='t_top4', help="token-level score the policy reads; 'none' = supervised heads only")
    ap.add_argument('--lams', type=float, nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fix', nargs='*', default=[], help='name=value: numbers of theta the search leaves alone')
    a = ap.parse_args()
    a.heads = None if a.heads == 'none' else a.heads
    FIX = {k: float(v) for k, v in (f.split('=') for f in a.fix)}
    SLIDES = S.load('train', a.tok3)
    folds = W.folds()
    fold = np.array([folds[s.name] for s in SLIDES])
    out_dir = f'{W.ROOT}/search/{a.out}'
    os.makedirs(out_dir, exist_ok=True)
    rule = [S.rule(s) for s in SLIDES]
    print('rule', json.dumps({k: round(v, 4) for k, v in S.summarise(rule).items()}), flush=True)
    res = dict(rule=rule, names=[s.name for s in SLIDES], runs={})
    with get_context('fork').Pool(a.workers) as pool:
        for lam in a.lams:
            cv, thetas = [None] * len(SLIDES), []
            for k in range(W.N_FOLDS):
                th, _ = cem(np.flatnonzero(fold != k), lam, a.heads, a.seed + k, pool=pool)
                thetas.append(th)
                for i in np.flatnonzero(fold == k):
                    cv[i] = P.play(SLIDES[i], th, a.heads)
            th_all, _ = cem(np.arange(len(SLIDES)), lam, a.heads, a.seed + 99, pool=pool)
            fit = [P.play(s, th_all, a.heads) for s in SLIDES]
            res['runs'][lam] = dict(cv=cv, thetas=thetas, theta=th_all, fit=fit)
            tum = [i for i, s in enumerate(SLIDES) if s.tumour]
            print(f'lam {lam}: cross-fitted', json.dumps({k: round(v, 4) for k, v in S.summarise(cv).items()}),
                  '\n   lesions lost vs rule', sum(max(0, rule[i]['found'] - cv[i]['found']) for i in tum), 'gained', sum(max(0, cv[i]['found'] - rule[i]['found']) for i in tum),
                  '| tumour slides ended at stage 1:', sum(cv[i]['stage'] == 1 for i in tum), 'of', len(tum), '| Dice below rule on', sum(cv[i]['dice'] < rule[i]['dice'] - 1e-9 for i in tum),
                  'above on', sum(cv[i]['dice'] > rule[i]['dice'] + 1e-9 for i in tum),
                  '\n   stage-2 slides', sum(r['stage'] == 2 for r in cv), 'in-sample', json.dumps({k: round(v, 4) for k, v in S.summarise(fit).items()}),
                  '\n   theta', ' '.join(f'{f}={v:.3f}' for f, v in zip(P.Theta._fields, th_all)), flush=True)
            pickle.dump(res, open(f'{out_dir}/res.pkl', 'wb'))


if __name__ == '__main__':
    main()
