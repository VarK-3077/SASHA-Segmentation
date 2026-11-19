# A value-based zoom policy learned from reward, not from labels. For every unit the L6 screen passes, the
# reward of segmenting it is read off the real segmenter's outcome (simulator tables): its share of the slide's
# cleaned Dice, plus W_LES for each lesion whose best candidate lies in it, minus W_FP times the score of each
# false-positive candidate in it. Q(state) regresses that reward on the unit's state: both L3 heads' scores,
# the L6 score, and the unit's rank among the slide's screened units. The policy segments a unit when
# Q >= c (c plays the price of a zoom). Cross-fitted: a train slide's Q comes from a model fitted on the other
# folds; test slides get the all-train model. Saves Q grids as ROOT/q/TAG/<slide>.npy and the models.
# Usage: python -m experiments.rlpolicy.qpol --tok3 t1 --tag q0

import argparse
import os
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from experiments.rlpolicy import sim as S
from experiments.rlpolicy import where as W

TAU = 0.6
W_LES, W_FP = 0.248, 0.0497          # the RL cascade's reward weights (rlcascade ppo1)
FEATS = ('p3', 'pg', 't_top4', 't_top16', 't_max', 't_unit', 't_mean', 'p_core')


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def rank(v):
    return np.argsort(np.argsort(v)) / max(len(v) - 1, 1)


def state(s, m):
    """One row per screened unit: the heads' scores, and the unit's rank among the slide's screened units."""
    cols = [logit(getattr(s, k)[m]) for k in FEATS[:-1]] + [s.p_core[m]]
    return np.stack(cols + [rank(s.p3[m]), rank(s.t_top16[m]), np.full(m.sum(), np.log(m.sum() + 1.0))], 1)


def unit_reward(s, m):
    """Reward of each screened unit when all of them are segmented (no painting, no tracing)."""
    hw = s.tissue.size
    sel = np.flatnonzero(m.reshape(-1)[s.cu_unit])
    r = np.zeros(hw)
    if len(sel) == 0:
        return r.reshape(m.shape)[m]
    comp = s.cu_comp[sel]
    first = sel[np.r_[0, np.flatnonzero(np.diff(comp)) + 1]]
    mx, lab, unit = s.cu_max[first], s.cu_lab[first], s.cu_unit[first]
    fp = lab == 0
    np.add.at(r, unit[fp], -W_FP * mx[fp])
    if s.tumour:
        for l in s.lesions:
            k = np.flatnonzero(lab == l)
            if len(k):
                r[unit[k[np.argmax(mx[k])]]] += W_LES
        kept = np.zeros(int(s.cu_comp.max()) + 1, bool)
        kept[s.cu_comp[first[mx >= S.SEED]]] = True
        rows = sel[kept[comp]]
        iu, pu = np.bincount(s.cu_unit[rows], s.cu_tp[rows], hw), np.bincount(s.cu_unit[rows], s.cu_n[rows], hw)
        i, p = iu.sum(), pu.sum()
        d = 2 * i / (p + s.G) if p + s.G > 0 else 0.0
        r += d - 2 * (i - iu) / np.maximum(p - pu + s.G, 1e-9)
    return r.reshape(m.shape)[m]


def model(mono):
    """mono: Q may only rise with each head's score (ranks and the slide's size stay free)."""
    cst = [1] * len(FEATS) + [0, 0, 0] if mono else None
    return HistGradientBoostingRegressor(max_iter=200, max_depth=4, learning_rate=0.06, min_samples_leaf=40, l2_regularization=1.0, monotonic_cst=cst)


def grid(s, m, q):
    g = np.full(s.tissue.shape, -1e9, np.float32)
    g[m] = q
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tok3', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--mono', type=int, default=1)
    a = ap.parse_args()
    out = f'{W.ROOT}/q/{a.tag}'
    os.makedirs(out, exist_ok=True)
    tr, te = S.load('train', a.tok3), S.load('test', a.tok3)
    folds = W.folds()
    fold = np.array([folds[s.name] for s in tr])
    masks = [s.tissue & (s.pg >= TAU) for s in tr]
    X = [state(s, m) for s, m in zip(tr, masks)]
    Y = [unit_reward(s, m) for s, m in zip(tr, masks)]
    print('units', sum(len(y) for y in Y), 'mean reward', np.mean(np.concatenate(Y)), 'positive', np.mean(np.concatenate(Y) > 0), flush=True)
    for k in range(W.N_FOLDS):
        fit = np.flatnonzero(fold != k)
        reg = model(a.mono).fit(np.concatenate([X[i] for i in fit]), np.concatenate([Y[i] for i in fit]))
        for i in np.flatnonzero(fold == k):
            np.save(f'{out}/{tr[i].name}.npy', grid(tr[i], masks[i], reg.predict(X[i]) if masks[i].any() else np.zeros(0)))
        print('fold', k, flush=True)
    reg = model(a.mono).fit(np.concatenate(X), np.concatenate(Y))
    pickle.dump(reg, open(f'{out}/model.pkl', 'wb'))
    for s in te:
        m = s.tissue & (s.pg >= TAU)
        np.save(f'{out}/{s.name}.npy', grid(s, m, reg.predict(state(s, m)) if m.any() else np.zeros(0)))
    open(f'{out}/DONE', 'w').close()


if __name__ == '__main__':
    main()
