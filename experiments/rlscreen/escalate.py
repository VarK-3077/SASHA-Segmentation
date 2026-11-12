# The escalation rule, tuned on out-of-fold train with real segmenter outcomes (lesrl.py), no test.
# A slide is screened at level 3 down the level-6 ranking in stages (start -> [middle] -> final); it goes to
# the next stage only if the evidence so far shows tumour: at least minpos newly zoomed units whose segmented
# tumour share is >= minfrac, counted over the last stage ('last') or the whole episode ('any'). Units
# scoring p(tumour) >= tz at level 3 are zoomed (p(core) >= 0.99 painted); borders are followed.
# Selection: the cheapest setting within a tolerance of the best train score (lesion-aware J), checked by
# picking on four folds and scoring the fifth, and across the two hospitals. Test is printed only at the pick.
# Usage: RLSCREEN_OOF=1 python -m experiments.rlscreen.escalate

import argparse
import itertools
import json
from multiprocessing import get_context

import numpy as np

from experiments.lowsep.train import fold_of
from experiments.rlscreen import lesrl
from experiments.rlscreen import paths as P
from experiments.rlscreen import stoprl as R
from experiments.rlscreen.ops_select import hospital
from experiments.rlscreen.rollout import load_tables
from experiments.rlscreen.seqscreen import Slide

SLIDES = {}


def rule(stages, minpos, minfrac, mode, tc=0.99):
    def act(ep):
        if ep.follow.any():
            ep.step(0, 2.0)
            return True
        if ep.round == 0:
            k = 0
        else:
            ref = ep.stage_start if mode == 'last' else np.zeros_like(ep.zoomed)
            new = ep.zoomed & ~ref
            evid = int((new & ~ep.painted & (ep.s.seg_frac >= minfrac)).sum() + (new & ep.painted).sum())
            if evid < minpos or ep.stage + 1 >= len(stages):
                return False
            k = ep.stage + 1
        while k < len(stages):
            n = ep.count('thresh', stages[k])
            ep.stage = k
            ep.stage_start = ep.zoomed.copy()
            if n > 0:
                ep.step(n, tc)
                return True
            k += 1
        return False
    return act


def settings():
    out = []
    for start, final in itertools.product((0.95, 0.9, 0.8, 0.7), (0.7, 0.6, 0.5, 0.4, 0.3)):
        if final >= start:
            continue
        for mid in (None, round((start + final) / 2, 3)):
            st = (start, final) if mid is None else (start, mid, final)
            for minpos, minfrac, mode, tz in itertools.product((1, 2, 4), (0.0, 0.05), ('last', 'any'), (0.5, 0.6, 0.7)):
                out.append((st, minpos, minfrac, mode, tz))
    return out


def play(arg):
    split, (st, minpos, minfrac, mode, tz) = arg
    R.CFG.update(design='coarse', tz=tz, S=4.0, fp=1.0, tcs=(0.99,), ep=lesrl.SegEp)
    res = []
    for s in SLIDES[split]:
        e = R.run(s, 1.0, rule(st, minpos, minfrac, mode)).summary()
        res.append((e['j'], np.nan if e['dice'] is None else e['dice'], np.nan if e['found'] is None else e['found'], e['sec'], e['l2']))
    return np.array(res)


def pick(res, keys, mask, tol):
    j = {k: res[k][mask, 0].mean() for k in keys}
    best = max(j.values())
    return min([k for k in keys if j[k] >= best - tol], key=lambda k: res[k][mask, 3].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    for sp, trusted in (('train', P.NON_TRUSTED_SKIP), ('test', False)):
        S = lesrl.attach([Slide(t, 'mix') for t in load_tables(sp, 'coarse', 6, 'kaiko_vitl14', 'cpu', trusted_only=trusted)])
        for s in S:
            z = np.load(f'{P.units_dir("kaiko_vitl14")}/seg/{s.name}.npz')
            s.seg_frac, s.cluster = z['seg_frac'], R.cluster_stats(s)
        SLIDES[sp] = S
    R.CFG.update(design='coarse')
    R.build_actions()
    grid = settings()
    with get_context('fork').Pool(a.workers) as pool:
        tr = dict(zip(grid, pool.map(play, [('train', g) for g in grid], chunksize=4)))
        te = dict(zip(grid, pool.map(play, [('test', g) for g in grid], chunksize=4)))
    names = [s.name for s in SLIDES['train']]
    folds = np.array([fold_of(x, 5) for x in names])
    hosp = np.array([hospital(x) for x in names])
    checks = {'5-fold': [(folds != k, folds == k) for k in range(5)], 'A->B': [(hosp == 'A', hosp == 'B')], 'B->A': [(hosp == 'B', hosp == 'A')]}
    keys = list(tr)
    print(f'{len(grid)} settings, {len(names)} train slides\n')
    print('| tolerance | check | J lost vs best in hindsight | s | distinct picks |\n|---|---|---|---|---|')
    tols = (0.005, 0.01, 0.02, 0.03)
    for tol in tols:
        for name, parts in checks.items():
            lost, sec, picks = [], [], set()
            for inn, out in parts:
                k = pick(tr, keys, inn, tol)
                best = max(keys, key=lambda q: tr[q][out, 0].mean())
                lost.append(tr[best][out, 0].mean() - tr[k][out, 0].mean()); sec.append(tr[k][out, 3].mean()); picks.add(k)
            print(f'| {tol:g} | {name} | {np.mean(lost):.3f} | {np.mean(sec):.2f} | {len(picks)} |')
    print('\n| tolerance | picked on all train | train J | train s | test J | test raw Dice | test found | test L2 | test s |\n|---|---|---|---|---|---|---|---|---|')
    chosen = {}
    allm = np.ones(len(names), bool)
    for tol in tols:
        k = pick(tr, keys, allm, tol)
        t, e = tr[k], te[k]
        chosen[str(tol)] = k
        print(f'| {tol:g} | {k} | {t[:, 0].mean():.3f} | {t[:, 3].mean():.2f} | {e[:, 0].mean():.3f} | {np.nanmean(e[:, 1]):.3f} | {np.nanmean(e[:, 2]):.3f} | {e[:, 4].mean():.3f} | {e[:, 3].mean():.2f} |')
    ref = ((0.8, 0.6), 1, 0.0, 'last', 0.7)
    t, e = tr[ref], te[ref]
    print(f'| current rule | {ref} | {t[:, 0].mean():.3f} | {t[:, 3].mean():.2f} | {e[:, 0].mean():.3f} | {np.nanmean(e[:, 1]):.3f} | {np.nanmean(e[:, 2]):.3f} | {e[:, 4].mean():.3f} | {e[:, 3].mean():.2f} |')
    json.dump({k: list(v) if isinstance(v, tuple) else v for k, v in chosen.items()}, open(f'{P.ROOT}/escalate_chosen.json', 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
