# Fair comparison of stopping and core rules: at each compute price, every family (fixed threshold,
# hit-rate, one RL run) picks its best row on the out-of-fold train slides (Dice - price * seconds, with
# a penalty for painting units on normal slides), and only that row is read on test.
# Usage: python -m experiments.rlscreen.stop_pick --design full --runs stop_full_s0 stop_full_s1

import argparse
import json

from experiments.rlscreen import paths as P

PRICES = (0.02, 0.05, 0.1, 0.2)


def value(r, price):
    return r['dice'] - price * r['sec']


def pick(train, test, keys, price):
    k = max(keys, key=lambda x: value(train[x], price))
    return k, test[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--design', required=True)
    ap.add_argument('--runs', nargs='*', default=[])
    a = ap.parse_args()
    rules = json.load(open(f'{P.ROOT}/stop_rules_{a.design}.json'))
    fam = {}
    tr, te = rules['train'], rules['test']
    fam['fixed, no paint'] = (tr, te, [k for k in tr if k.startswith('fixed') and 'no paint' in k])
    fam['fixed + paint'] = (tr, te, [k for k in tr if k.startswith('fixed')])
    fam['hit-rate, no paint'] = (tr, te, [k for k in tr if k.startswith('hit') and 'no paint' in k])
    fam['hit-rate + paint'] = (tr, te, [k for k in tr if k.startswith('hit')])
    for run in a.runs:
        o = json.load(open(f'{P.ROOT}/stop_{run}.json'))
        fam[run] = (o['train'], o['test'], [k for k in o['train'] if k.startswith('RL')])
    print('| price (Dice/s) | family | picked on OOF train | Dice | L2 | painted | s | s normal | L2 normal | FP units, normal |\n|---|---|---|---|---|---|---|---|---|---|')
    for p in PRICES:
        for name, (trn, tst, keys) in fam.items():
            k, r = pick(trn, tst, keys, p)
            print(f"| {p:g} | {name} | {k} | {r['dice']:.3f} | {r['l2']:.3f} | {r['painted']:.3f} | {r['sec']:.2f} | {r['sec_normal']:.2f} | {r['l2_normal']:.3f} | {r['fp_normal']:.2f} |")


if __name__ == '__main__':
    main()
