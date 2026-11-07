# Per compute price, each family in each run picks its best row on out-of-fold train (Dice - price *
# seconds); only that row is read on test.
# Usage: python -m experiments.rlscreen.qstop_pick qs_full_e0 qs_full_e02 ...

import json
import sys

from experiments.rlscreen import paths as P

print('| price | run | family | picked on train | test Dice | L2 | painted | s | s normal | FP units, normal |\n|---|---|---|---|---|---|---|---|---|---|')
for price in (0.02, 0.05, 0.1, 0.2):
    for run in sys.argv[1:]:
        o = json.load(open(f'{P.ROOT}/stop_{run}.json'))
        tr, te = o['train'], o['test']
        for fam in ('fixed', 'hit', 'RL'):
            keys = [k for k in tr if k.startswith(fam)]
            k = max(keys, key=lambda x: tr[x]['dice'] - price * tr[x]['sec'])
            r = te[k]
            print(f"| {price:g} | {run} | {fam} | {k} | {r['dice']:.3f} | {r['l2']:.3f} | {r['painted']:.3f} | {r['sec']:.2f} | {r['sec_normal']:.2f} | {r['fp_normal']:.2f} |")
