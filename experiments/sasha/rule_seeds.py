"""Seed spread of the tile-picking rules' slide classification, from the offline rollouts of patch_cls/acq/run_acq.py
(stored features, same env / TSU / HAFED, one rollout pickle per policy and seed): Accuracy, AUC and F1 at the 0.5
softmax argmax, per seed and as mean ± sd. The timed pipeline runs one seed per rule; this says how much that one
seed can move. Usage: python -m experiments.sasha.rule_seeds --dirs NAME=DIR ... --out FILE.md"""

import argparse
import glob
import pickle

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dirs', nargs='+', required=True, help='NAME=DIR; DIR holds rollout_<policy>.pkl files of one seed each in subdirectories s<seed>/')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    L = ['| rule | seeds | Accuracy mean ± sd | AUC mean ± sd | F1 mean ± sd | per seed (Acc / AUC / F1) |', '|---|---|---|---|---|---|']
    for spec in a.dirs:
        name, d = spec.split('=', 1)
        rows = []
        for f in sorted(glob.glob(f'{d}/s*/rollout_*.pkl')):
            r = pickle.load(open(f, 'rb'))['slides']
            y = np.array([v['label'] for v in r.values()])
            p = np.array([v['p_tumor'] for v in r.values()])
            rows.append((accuracy_score(y, p >= 0.5), roc_auc_score(y, p), f1_score(y, p >= 0.5)))
        v = np.array(rows)
        sd = v.std(0, ddof=1) if len(v) > 1 else np.zeros(3)
        L.append(f'| {name} | {len(v)} | {v[:, 0].mean():.3f} ± {sd[0]:.3f} | {v[:, 1].mean():.3f} ± {sd[1]:.3f} | {v[:, 2].mean():.3f} ± {sd[2]:.3f} | '
                 + '; '.join(f'{x[0]:.3f} / {x[1]:.3f} / {x[2]:.3f}' for x in v) + ' |')
    open(a.out, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
