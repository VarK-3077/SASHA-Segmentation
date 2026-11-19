# Exact-scorer tables for searched policies. --split train scores the cross-fitted routes a search saved (each
# train slide played by the policy fitted without its fold); --split test plays the all-train policy on the
# test slides. Both print rlcascade.report's table and paired bootstrap against the supervised rule, with the
# simulator's own numbers beside them. Every test read is logged in ROOT/TEST_READS.txt: one read per policy.
# Usage: python -m experiments.rlpolicy.evalx --split train --tok3 t0 --runs SEARCH@RUN ... --out FILE.md

import argparse
import os
import pickle
import time

from experiments.rlcascade import report as RP
from experiments.rlpolicy import exact as X
from experiments.rlpolicy import policy as P
from experiments.rlpolicy import sim as S
from experiments.rlpolicy import where as W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=('train', 'test'), required=True)
    ap.add_argument('--tok3', required=True)
    ap.add_argument('--runs', nargs='+', required=True, help='SEARCH@RUN: a search directory and one of its runs (a lambda, or vsel\'s score:kappa)')
    ap.add_argument('--key', default='t_top4', help='token score, for searches that do not record it')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--tok6', default=None, help='tag of the second L6 head, for runs with an L6 veto')
    ap.add_argument('--q', default=None, help='tag of a value policy (qpol.py) whose Q grids the runs read')
    a = ap.parse_args()
    slides = S.load(a.split, a.tok3, tok6=a.tok6)
    if a.q:
        import numpy as np
        for s in slides:
            s.q = np.load(f'{W.ROOT}/q/{a.q}/{s.name}.npy')
    routes = {s.name: {'rule': S.rule(s)} for s in slides}
    lines, keys, sims = [], [], {'rule': [routes[s.name]['rule'] for s in slides]}
    for spec in a.runs:
        name, rk = spec.split('@')
        res = pickle.load(open(f'{W.ROOT}/search/{name}/res.pkl', 'rb'))
        run = res['runs'][rk] if rk in res['runs'] else res['runs'][float(rk)]
        key = run.get('key', a.key)
        k = f'{name} {rk}'
        keys.append(k)
        if a.split == 'train':
            assert res['names'] == [s.name for s in slides]
            recs = run['cv']
        else:
            for s in slides:
                s.g_key = run.get('g_key', 'g_max')
            if 'nets' in run:
                from experiments.rlpolicy.rl3 import player
                recs = [player(run)(s) for s in slides]
            else:
                recs = [P.play(s, run['theta'], None if key == 'none' else key) for s in slides]
            open(f'{W.ROOT}/TEST_READS.txt', 'a').write(f"{time.strftime('%F %T')} {spec} {'PPO policy' if 'nets' in run else tuple(round(float(v), 4) for v in run['theta'])}\n")
        sims[k] = recs
        for s, r in zip(slides, recs):
            routes[s.name][k] = r
        lines.append(f"- {k}: " + (f"PPO policy, state {run['state']}" if 'nets' in run else 'all-train theta ' + ' '.join(f'{f}={v:.4g}' for f, v in zip(P.Theta._fields, run['theta']))))
    ex = X.score_routes(a.split, routes, a.workers)
    head = f"## {a.split}, {len(slides)} slides, exact scorer ({'cross-fitted policies' if a.split == 'train' else 'all-train policy, test read once'}); seconds = cost model\n"
    sim_rows = ['', '| simulator | s | Dice | FROC | L3 | L2 | painted | FP comps |', '|---|---|---|---|---|---|---|---|']
    for k, recs in sims.items():
        m = S.summarise(recs)
        sim_rows.append(f"| {k} | {m['sec']:.3f} | {m['dice']:.4f} | {m['froc']:.4f} | {m['l3']:.0f} | {m['l2']:.0f} | {m['paint']:.1f} | {m['fp']:.2f} |")
    text = '\n'.join([head, RP.table(ex, ['rule'] + keys), '', RP.BOOT_HEAD] + [RP.boot_line(ex, k) for k in keys] + sim_rows + [''] + lines) + '\n'
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    open(a.out, 'w').write(text)
    pickle.dump(ex, open(a.out[:-3] + '.pkl', 'wb'))
    print(text)


if __name__ == '__main__':
    main()
