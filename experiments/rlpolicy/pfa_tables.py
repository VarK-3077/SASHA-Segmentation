# Tables for the PFA-ScanNet L0 timing runs (run_pfa.py, cascade_pfa.run_cascade, cascade_pfa.run_dense): each
# run directory's maps are scored as the cascade + PFA window scored its PFA rows (raw map at 0.3 for Dice, lesion
# Dice and FROC, one candidate per component) and joined with the per-slide time records. The first run is the
# reference of the table and of the paired bootstrap. --subset limits every run to a json list of slides (the
# dense run covers only that subset) and adds how many times faster than the run named 'dense' each row is.
# Usage: python -m experiments.rlpolicy.pfa_tables --runs rule=DIR rl=DIR [dense=DIR --subset FILE] --out FILE.md

import argparse
import json
import os
from functools import partial
from multiprocessing import Pool

import numpy as np

from experiments.pfa_scannet import clean_sweep as CS
from experiments.rlcascade import report as RP

VAR, KEY = CS.variants([0.3], [], [0]), 't0.3'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--subset', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    dirs = dict(r.split('=', 1) for r in a.runs)
    first = next(iter(dirs))
    names = json.load(open(a.subset)) if a.subset else sorted(f[:-4] for f in os.listdir(dirs[first]) if f.endswith('.npz'))
    scores = {}
    with Pool(a.workers) as pool:
        for k, d in dirs.items():
            scores[k] = dict(pool.imap_unordered(partial(CS.run_one, maps=d, var=VAR), names))
    time_of = lambda k, n: json.load(open(f'{dirs[k]}/{n}_time.json'))
    res = {}
    for n in names:
        rows, sc = {}, {}
        for k in dirs:
            t, key = time_of(k, n), 'rule' if k == first else k
            rows[key] = dict(n_l3=t.get('screened', 0), n_l2=t['zoomed'], n_paint=t['painted'], stopped=False, map=key, sec=t['wall'])
            sc[key] = dict(clean=scores[k][n][KEY], raw=scores[k][n][KEY])
        res[n] = dict(rows=rows, scores=sc)
    keys = ['rule'] + [k for k in dirs if k != first]
    label = {'rule': first}
    lines = [f'## PFA-ScanNet L0 segmenter, {len(names)} test slides; s/slide = measured wall; PFA scoring (raw map at 0.3); reference row = {first}\n',
             RP.table(res, keys, label), '', RP.BOOT_HEAD] + [RP.boot_line(res, k) for k in keys[1:]]
    lines += ['', '| run | wall mean | median | p95 | PFA GPU s | L3 screen s | ROIs | mean load1 |' + (' x faster than dense |' if 'dense' in dirs else ''), '|---' * (8 + ('dense' in dirs)) + '|']
    dense = np.mean([time_of('dense', n)['wall'] for n in names]) if 'dense' in dirs else None
    for k in dirs:
        ts = [time_of(k, n) for n in names]
        w = np.array([t['wall'] for t in ts])
        lines.append(f"| {k} | {w.mean():.2f} | {np.median(w):.2f} | {np.percentile(w, 95):.2f} | {np.mean([t.get('pfa_gpu', 0) for t in ts]):.2f} | "
                     f"{np.mean([t.get('screen_l3', 0) for t in ts]):.2f} | {np.mean([t.get('rois', 0) for t in ts]):.0f} | {np.mean([t.get('load1', np.nan) for t in ts]):.2f} |"
                     + (f' {dense / w.mean():.1f} |' if dense else ''))
    open(a.out, 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
