# Dice and FROC of timed run directories under BOTH conventions, from the maps the runs wrote:
#   raw:     the map thresholded at --thr; Dice on it, FROC with one candidate per component (score = its maximum)
#   cleaned: the same map keeping only the components whose maximum reaches --seed; Dice and FROC on what is left
# (the usual level-2 rows report cleaned Dice with raw FROC; PFA rows report both raw at 0.3). Also the measured
# wall seconds, the units read, and the machine load recorded with each run.
# Usage: python -m experiments.rlpolicy.fourway --runs NAME=DIR ... --thr 0.4 --seed 0.9 --out FILE.md

import argparse
import json
import os
from functools import partial
from multiprocessing import Pool

import numpy as np

from experiments.pfa_scannet import clean_sweep as CS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--thr', type=float, default=0.4)
    ap.add_argument('--seed', type=float, default=0.9)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    dirs = dict(r.split('=', 1) for r in a.runs)
    VAR = CS.variants([a.thr], [a.seed], [0])
    RAW, CLEAN = CS.name(a.thr, None, 0), CS.name(a.thr, a.seed, 0)
    first = next(iter(dirs))
    names = sorted(f[:-4] for f in os.listdir(dirs[first]) if f.endswith('.npz'))
    L = [f'## {len(names)} test slides; map threshold {a.thr}; cleaned = components whose maximum reaches {a.seed}; s/slide = measured wall of that run', '',
         '| run | s/slide | L3 units | L2 units | painted | Dice raw | Dice cleaned | small-lesion Dice raw / cleaned | FROC raw | FROC cleaned | FP comps/slide raw / cleaned | lesions hit raw / cleaned | mean load1 |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    with Pool(a.workers) as pool:
        for k, d in dirs.items():
            sc = dict(pool.imap_unordered(partial(CS.run_one, maps=d, var=VAR), names))
            ts = [json.load(open(f'{d}/{n}_time.json')) if os.path.exists(f'{d}/{n}_time.json') else {} for n in names]
            cell = {}
            for tag, v in (('raw', RAW), ('clean', CLEAN)):
                recs = [sc[n][v] for n in names]
                tum = [r for r in recs if 'dice' in r]
                les = [x for r in tum for x in r['lesions']]
                small = [d2 for (size, d2) in les if size < 500] if les and len(les[0]) == 2 else []
                cell[tag] = dict(dice=float(np.mean([r['dice'] for r in tum])), froc=float(CS.froc([r['froc'] for r in recs])[0]), fp=float(np.mean([r['fp'] for r in recs])),
                                 hit=sum(1 for (_, d2) in les if d2 > 0) if small is not None and les and len(les[0]) == 2 else -1, n=len(les),
                                 small=float(np.mean(small)) if small else float('nan'))
            g = lambda key, alt=None: np.mean([t.get(key, t.get(alt, np.nan)) if alt else t.get(key, np.nan) for t in ts]) if ts and ts[0] else float('nan')
            L.append(f"| {k} | {g('wall'):.3f} | {g('screened'):.0f} | {g('zoomed', 'crops'):.0f} | {g('painted'):.1f} | {cell['raw']['dice']:.4f} | {cell['clean']['dice']:.4f} | "
                     f"{cell['raw']['small']:.3f} / {cell['clean']['small']:.3f} | {cell['raw']['froc']:.4f} | {cell['clean']['froc']:.4f} | {cell['raw']['fp']:.2f} / {cell['clean']['fp']:.2f} | "
                     f"{cell['raw']['hit']} / {cell['clean']['hit']} of {cell['raw']['n']} | {g('load1'):.1f} |")
            print(L[-1], flush=True)
    open(a.out, 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
