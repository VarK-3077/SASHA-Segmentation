# Paired per-slide comparison of L6 -> L3 RL stoppers with the fixed rule they should beat, on out-of-fold
# train and test: bootstrap over slides of the Dice and seconds differences, and slides lost.
# Usage: RLSCREEN_OOF=1 python -m experiments.rlscreen.seq_paired

import numpy as np
import torch

from experiments.rlscreen import paths as P
from experiments.rlscreen import stoprl as R
from experiments.rlscreen.rollout import load_tables
from experiments.rlscreen.seqscreen import Slide

PAIRS = (('seqs_long', 0.204, 0.6), ('seqs_s0', 0.849, 0.8), ('seqs_s2', 0.849, 0.8), ('seqs_s1', 0.849, 0.8))
R.CFG.update(design='coarse', tz=0.5, S=4.0, fp=1.0, tcs=(0.99,), cluster=True)
R.build_actions()
rng = np.random.default_rng(0)
print('| split | RL vs fixed | Dice RL / fixed | dDice 95% | s RL / fixed | ds 95% | normal s | missed tumour slides RL / fixed |\n|---|---|---|---|---|---|---|---|')
for split in ('train', 'test'):
    S = [Slide(t, 'mix') for t in load_tables(split, 'coarse', 6, 'kaiko_vitl14', 'cpu', trusted_only=split == 'train')]
    for s in S:
        s.cluster = R.cluster_stats(s)
    tum = np.array([s.tumour for s in S])
    for run, lam, ts in PAIRS:
        pol = R.Net()
        pol.load_state_dict(torch.load(f'{P.run_dir(run)}/final.pt', map_location='cpu'))
        a = [R.run(s, lam, R.policy_actor(pol)).summary() for s in S]
        b = [R.run(s, 1.0, R.fixed_rule(ts, 0.99)).summary() for s in S]
        da = np.array([x['dice'] for x, t in zip(a, tum) if t], float); db = np.array([x['dice'] for x, t in zip(b, tum) if t], float)
        sa, sb = np.array([x['sec'] for x in a]), np.array([x['sec'] for x in b])
        i = rng.integers(len(da), size=(5000, len(da))); j = rng.integers(len(sa), size=(5000, len(sa)))
        cd, cs = np.percentile((da - db)[i].mean(1), [2.5, 97.5]), np.percentile((sa - sb)[j].mean(1), [2.5, 97.5])
        print(f'| {split} | {run} lam {lam:g} vs t={ts:g} | {da.mean():.3f} / {db.mean():.3f} | {cd[0]:+.3f} .. {cd[1]:+.3f} | {sa.mean():.2f} / {sb.mean():.2f} | {cs[0]:+.2f} .. {cs[1]:+.2f} | '
              f'{sa[~tum].mean():.2f} / {sb[~tum].mean():.2f} | {(da < 0.05).sum()} / {(db < 0.05).sum()} |', flush=True)
