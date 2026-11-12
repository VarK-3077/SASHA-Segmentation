# Scores real ablation runs per slide with clean_sweep's own functions (cleaned map t0.4 seed 0.9 for
# Dice and lesion Dice, raw map t0.4 for FROC, as in real_results_vivi_2026-11-07.md), then prints the
# table and paired bootstraps over slides against two reference rows.
# Usage: python -m experiments.rlscreen.ablate_score --root DIR --runs r1 r2 ... --refs r3 r2

import argparse
import glob
import json
import os
from functools import partial
from multiprocessing import Pool

import numpy as np

from experiments.pfa_scannet import clean_sweep as CS

VAR = CS.variants([0.4], [0.9], [0])
CLEAN, RAW = 't0.4_seed0.9', 't0.4'


def per_slide(maps, workers):
    out = f'{maps}/per_slide.json'
    if os.path.exists(out):
        return json.load(open(out))
    names = sorted(os.path.basename(x)[:-4] for x in glob.glob(f'{maps}/test_*.npz'))
    res = {}
    with Pool(workers) as pool:
        for n, r in pool.imap_unordered(partial(CS.run_one, maps=maps, var=VAR), names):
            res[n] = r
    json.dump(res, open(out, 'w'))
    return res


def summarise(res, times, idx=None):
    names = sorted(res)
    if idx is not None:
        names = [names[i] for i in idx]
    tum = [n for n in names if 'dice' in res[n][CLEAN]]
    les = [l for n in tum for l in res[n][CLEAN]['lesions']]
    return dict(dice=float(np.mean([res[n][CLEAN]['dice'] for n in tum])),
                froc=float(CS.froc([res[n][RAW]['froc'] for n in names])[0]),
                lesion=[float(np.mean([d for u, d in les if lo <= u < hi])) for lo, hi in CS.BINS],
                sec=float(np.mean([times[n]['wall'] for n in names])), l2=float(np.mean([times[n]['zoomed'] for n in names])),
                painted=float(np.mean([times[n].get('painted', 0) for n in names])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--refs', nargs='+', required=True)
    ap.add_argument('--boot', type=int, default=1000)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    R = {}
    for run in a.runs:
        d = f'{a.root}/{run}'
        R[run] = (per_slide(d, a.workers), {os.path.basename(f)[:-10]: json.load(open(f)) for f in glob.glob(f'{d}/test_*_time.json')})
    print('| row | s/slide | L2 units | painted | Dice | lesion Dice <0.5 / 0.5-2 / >=2 mm | FROC |\n|---|---|---|---|---|---|---|')
    S = {run: summarise(*R[run]) for run in a.runs}
    for run, s in S.items():
        print(f"| {run} | {s['sec']:.2f} | {s['l2']:.0f} | {s['painted']:.1f} | {s['dice']:.3f} | {' / '.join(f'{x:.3f}' for x in s['lesion'])} | {s['froc']:.3f} |")
    n = len(R[a.runs[0]][0])
    rng = np.random.default_rng(0)
    idx = [rng.integers(n, size=n) for _ in range(a.boot)]
    print('\n| row | vs | dDice 95% | dFROC 95% | ds 95% |\n|---|---|---|---|---|')
    for ref in a.refs:
        B = [summarise(*R[ref], i) for i in idx]
        for run in a.runs:
            if run == ref:
                continue
            A = [summarise(*R[run], i) for i in idx]
            q = lambda k: np.percentile([x[k] - y[k] for x, y in zip(A, B)], [2.5, 97.5])
            d, f, t = q('dice'), q('froc'), q('sec')
            print(f"| {run} | {ref} | {S[run]['dice'] - S[ref]['dice']:+.3f} ({d[0]:+.3f} .. {d[1]:+.3f}) | {S[run]['froc'] - S[ref]['froc']:+.3f} ({f[0]:+.3f} .. {f[1]:+.3f}) | "
                  f"{S[run]['sec'] - S[ref]['sec']:+.2f} ({t[0]:+.2f} .. {t[1]:+.2f}) |", flush=True)


if __name__ == '__main__':
    main()
