# Per-slide comparison of RL runs and rules on the L3 ranking: which tumour slides carry a difference in
# mean Dice, with a paired bootstrap over slides of the RL-minus-rule difference in Dice and seconds.
# Usage: RLSCREEN_OOF=1 python -m experiments.rlscreen.stop_slides --split test --runs stop_full_s0 stop_full_s1 --lam 0.416

import argparse
import json

import numpy as np
import torch

from experiments.rlscreen import paths as P
from experiments.rlscreen import stoprl as R
from experiments.rlscreen.rollout import load_tables
from experiments.rlscreen.seqscreen import Slide


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test')
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--lam', type=float, default=0.416)
    ap.add_argument('--t', type=float, nargs='+', default=[0.05, 0.1, 0.2, 0.3], help='fixed thresholds to compare with')
    a = ap.parse_args()
    R.CFG.update(design='full', tz=0.7, S=4.0, fp=1.0, tcs=(0.99,))
    R.build_actions()
    S = [Slide(t, 'mix') for t in load_tables(a.split, 'full', 6, 'kaiko_vitl14', 'cpu', trusted_only=a.split == 'train')]
    names = [s.name for s in S]
    res = {}
    for t in a.t:
        res[f'fixed {t:g}'] = [R.run(s, 1.0, R.fixed_rule(t, 0.99)).summary() for s in S]
    res['hit 0.02'] = [R.run(s, 1.0, R.hit_rule(0.02, 0.99)).summary() for s in S]
    for run in a.runs:
        pol = R.Net()
        pol.load_state_dict(torch.load(f'{P.run_dir(run)}/final.pt', map_location='cpu'))
        res[run] = [R.run(s, a.lam, R.policy_actor(pol)).summary() for s in S]
    tum = [i for i, s in enumerate(S) if s.tumour]
    d = {k: np.array([v[i]['dice'] for i in tum]) for k, v in res.items()}
    sec = {k: np.array([v[i]['sec'] for i in range(len(S))]) for k, v in res.items()}
    print(f'{a.split}: {len(S)} slides, {len(tum)} tumour')
    rng = np.random.default_rng(0)
    idx_t = rng.integers(len(tum), size=(5000, len(tum)))
    idx_a = rng.integers(len(S), size=(5000, len(S)))
    print('| RL run | vs | mean dDice | 95% CI | mean ds | 95% CI |\n|---|---|---|---|---|---|')
    for run in a.runs:
        for rule in [k for k in res if k not in a.runs]:
            dd, ds = d[run] - d[rule], sec[run] - sec[rule]
            bd, bs = dd[idx_t].mean(1), ds[idx_a].mean(1)
            print(f'| {run} | {rule} | {dd.mean():+.3f} | {np.percentile(bd, 2.5):+.3f} .. {np.percentile(bd, 97.5):+.3f} | '
                  f'{ds.mean():+.2f} | {np.percentile(bs, 2.5):+.2f} .. {np.percentile(bs, 97.5):+.2f} |')
    base = a.runs[0]
    for rule in ('fixed 0.3', 'hit 0.02'):
        print(f'slides where {base} and {rule} differ by > 0.05 Dice:')
        dd = d[base] - d[rule]
        for j in np.argsort(-np.abs(dd)):
            if abs(dd[j]) > 0.05:
                print(f'  {names[tum[j]]}: {base} {d[base][j]:.3f}, {rule} {d[rule][j]:.3f}')


if __name__ == '__main__':
    main()
