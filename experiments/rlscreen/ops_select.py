# Chooses the L6 -> L3 operating point (screen ts, zoom tz, paint tc) from the out-of-fold train slides,
# without test and without a tuned budget. Every setting is played on every slide (ideal segmenter).
# Rules: 'one-SE' takes the cheapest setting whose Dice is within one paired standard error of the
# best setting (no price to choose); 'price p' takes the best Dice - p * seconds.
# The rules themselves are checked without test: pick on four folds and score on the fifth, and pick on
# one hospital's slides and score on the other's, the nearest thing here to a new dataset. Test is
# printed only at the settings chosen from all train slides.
# Usage: RLSCREEN_OOF=1 python -m experiments.rlscreen.ops_select

import argparse
import itertools
import json
from multiprocessing import get_context

import numpy as np

from experiments.lowsep.train import fold_of
from experiments.rlscreen import paths as P
from experiments.rlscreen import stoprl as R
from experiments.rlscreen.rollout import load_tables
from experiments.rlscreen.seqscreen import Slide

TS = (0.3, 0.5, 0.6, 0.7, 0.8, 0.9)
TZ = (0.3, 0.5, 0.6, 0.7, 0.8, 0.9)
TC = (2.0, 0.99, 0.95)
SLIDES = {}


def play(arg):
    split, ts, tz, tc = arg
    R.CFG.update(design='coarse', tz=tz, S=4.0, fp=1.0, tcs=(0.99,))
    out = []
    for s in SLIDES[split]:
        e = R.run(s, 1.0, R.fixed_rule(ts, tc)).summary()
        out.append((np.nan if e['dice'] is None else e['dice'], e['sec'], e['l2'], e['fp_units']))
    return (ts, tz, tc), np.array(out)


def hospital(name):
    """CAMELYON16 training numbering; not read from any metadata file here, so treat as approximate."""
    kind, num = name.split('_')
    return 'A' if int(num) <= (100 if kind == 'normal' else 70) else 'B'


def pick(res, keys, mask, rule):
    """res: setting -> per-slide array; mask selects the slides used for choosing."""
    d = {k: res[k][mask, 0] for k in keys}
    mean_d = {k: np.nanmean(v) for k, v in d.items()}
    sec = {k: res[k][mask, 1].mean() for k in keys}
    if rule == 'one-SE':
        best = max(keys, key=lambda k: mean_d[k])
        ok = [k for k in keys if mean_d[k] >= mean_d[best] - np.nanstd(d[k] - d[best]) / np.sqrt(np.sum(~np.isnan(d[k])))]
        return min(ok, key=lambda k: sec[k])
    if str(rule).startswith('tol'):
        best = max(mean_d.values())
        return min([k for k in keys if mean_d[k] >= best - float(rule[3:])], key=lambda k: sec[k])
    return max(keys, key=lambda k: mean_d[k] - float(rule) * sec[k])


def scores(slides):
    """Unit scores a deployment sees without labels: level-6 p over tissue, level-3 p over units level 6 would pass."""
    return (np.concatenate([s.pg[s.tissue] for s in slides]), np.concatenate([s.p3[s.tissue & (s.pg >= 0.5)] for s in slides]))


def align(k, src, dst):
    """Moves thresholds to the same score percentile on unlabelled target slides."""
    ts, tz, tc = k
    q6, q3 = (src[0] < ts).mean(), (src[1] < tz).mean()
    return float(np.quantile(dst[0], q6)), float(np.quantile(dst[1], q3)), tc


def play_on(slides, k):
    ts, tz, tc = k
    R.CFG.update(design='coarse', tz=tz, S=4.0, fp=1.0, tcs=(0.99,))
    e = [R.run(s, 1.0, R.fixed_rule(ts, tc)).summary() for s in slides]
    return np.array([(np.nan if x['dice'] is None else x['dice'], x['sec'], x['l2'], x['fp_units']) for x in e])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    R.CFG.update(design='coarse')
    SLIDES['train'] = [Slide(t, 'mix') for t in load_tables('train', 'coarse', 6, 'kaiko_vitl14', 'cpu', trusted_only=P.NON_TRUSTED_SKIP)]
    SLIDES['test'] = [Slide(t, 'mix') for t in load_tables('test', 'coarse', 6, 'kaiko_vitl14', 'cpu')]
    grid = list(itertools.product(TS, TZ, TC))
    with get_context('fork').Pool(a.workers) as pool:
        tr = dict(pool.map(play, [('train', *g) for g in grid]))
        te = dict(pool.map(play, [('test', *g) for g in grid]))
    names = [s.name for s in SLIDES['train']]
    keys = list(tr)
    n = len(names)
    rules = ('tol0.005', 'tol0.01', 'tol0.02', 'tol0.03', 'tol0.05', 0.02, 0.05, 0.1)
    print(f'{n} train slides, {len(grid)} settings')
    print('\n## Rule check, no test: pick on some train slides, score on the rest\n')
    print('| rule | check | mean Dice lost vs best in hindsight | seconds | settings picked |\n|---|---|---|---|---|')
    folds = np.array([fold_of(x, 5) for x in names])
    hosp = np.array([hospital(x) for x in names])
    splits = {'5-fold': [(folds != k, folds == k) for k in range(5)], 'hospital A->B': [(hosp == 'A', hosp == 'B')], 'hospital B->A': [(hosp == 'B', hosp == 'A')]}
    for rule in rules:
        for name, parts in splits.items():
            lost, secs, picked = [], [], []
            for inn, out in parts:
                k = pick(tr, keys, inn, rule)
                best = max(keys, key=lambda q: np.nanmean(tr[q][out, 0]))
                lost.append(np.nanmean(tr[best][out, 0]) - np.nanmean(tr[k][out, 0]))
                secs.append(tr[k][out, 1].mean())
                picked.append(k)
            print(f'| {rule} | {name} | {np.mean(lost):.3f} | {np.mean(secs):.2f} | {sorted(set(picked))} |')
    print('\n## Same rules with label-free alignment to the target slides\n')
    print('| rule | check | plain Dice, s | aligned Dice, s | aligned thresholds |\n|---|---|---|---|---|')
    S = SLIDES['train']
    for rule in rules:
        for name, (inn, out) in (('hospital A->B', splits['hospital A->B'][0]), ('hospital B->A', splits['hospital B->A'][0])):
            k = pick(tr, keys, inn, rule)
            A, B = [S[i] for i in np.flatnonzero(inn)], [S[i] for i in np.flatnonzero(out)]
            k2 = align(k, scores(A), scores(B))
            e = play_on(B, k2)
            print(f'| {rule} | {name} | {np.nanmean(tr[k][out, 0]):.3f}, {tr[k][out, 1].mean():.2f} | {np.nanmean(e[:, 0]):.3f}, {e[:, 1].mean():.2f} | {tuple(round(x, 3) for x in k2)} |')
    print('\n## Chosen from all train slides, then test\n')
    print('| rule | ts, tz, tc | train Dice | train s | test Dice | test L2 | test s | FP units, test |\n|---|---|---|---|---|---|---|---|')
    allm = np.ones(n, bool)
    chosen = {}
    for rule in rules:
        k = pick(tr, keys, allm, rule)
        t, e = tr[k], te[k]
        chosen[str(rule)] = k
        k2 = align(k, scores(SLIDES['train']), scores(SLIDES['test']))
        e2 = play_on(SLIDES['test'], k2)
        chosen[f'{rule} aligned'] = k2
        print(f'| {rule} | {k} | {np.nanmean(t[:, 0]):.3f} | {t[:, 1].mean():.2f} | {np.nanmean(e[:, 0]):.3f} | {e[:, 2].mean():.3f} | {e[:, 1].mean():.2f} | {e[:, 3].mean():.2f} |')
        print(f'| {rule} aligned | {tuple(round(x, 3) for x in k2)} | | | {np.nanmean(e2[:, 0]):.3f} | {e2[:, 2].mean():.3f} | {e2[:, 1].mean():.2f} | {e2[:, 3].mean():.2f} |')
    json.dump(chosen, open(f'{P.ROOT}/ops_chosen.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
