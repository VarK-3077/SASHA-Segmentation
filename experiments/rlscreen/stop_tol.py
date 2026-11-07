# Tolerance selection across rules and RL runs on L6 -> L3: for each Dice tolerance, the best train row
# over everything sets the bar, each family takes its cheapest train row within the tolerance, and only
# that row is read on test.
# Usage: python -m experiments.rlscreen.stop_tol --design coarse --runs seqs_s0 seqs_s1

import argparse
import json

from experiments.rlscreen import paths as P

ap = argparse.ArgumentParser()
ap.add_argument('--design', default='coarse')
ap.add_argument('--runs', nargs='*', default=[])
a = ap.parse_args()
rules = json.load(open(f"{P.ROOT}/stop_rules_{a.design}.json"))
tr, te = rules['train'], rules['test']
fam = {'fixed': (tr, te, [k for k in tr if k.startswith('fixed')]),
       'escalate': (tr, te, [k for k in tr if k.startswith('escalate')]),
       'hit-rate': (tr, te, [k for k in tr if k.startswith('hit')])}
for run in a.runs:
    o = json.load(open(f'{P.ROOT}/stop_{run}.json'))
    fam[run] = (o['train'], o['test'], [k for k in o['train'] if k.startswith('RL')])
    fam[run + ' clone'] = (o['train'], o['test'], [k for k in o['train'] if k.startswith('clone')])
print('| tolerance | family | picked on train | train Dice | train s | test Dice | L2 | screened | s | s normal | L2 normal |\n|---|---|---|---|---|---|---|---|---|---|---|')
best = max(t[k]['dice'] for t, _, ks in fam.values() for k in ks)
for tol in (0.01, 0.02, 0.03, 0.05, 0.1):
    for name, (t, e, ks) in fam.items():
        ok = [k for k in ks if t[k]['dice'] >= best - tol]
        if not ok:
            print(f'| {tol:g} | {name} | none within tolerance | | | | | | | | |')
            continue
        k = min(ok, key=lambda q: t[q]['sec'])
        r = e[k]
        print(f"| {tol:g} | {name} | {k} | {t[k]['dice']:.3f} | {t[k]['sec']:.2f} | {r['dice']:.3f} | {r['l2']:.3f} | {r['screened']:.3f} | {r['sec']:.2f} | {r['sec_normal']:.2f} | {r['l2_normal']:.3f} |")
