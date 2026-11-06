# Out-of-fold head outputs for the train slides. The supervised heads saw every train slide, so their
# train probabilities are overconfident; an RL policy trained on them would learn to trust them too
# much. Each head is retrained n_folds times (lowsep train.py --fold k), and each train slide is
# scored by the copy that did not see it. Written beside the slide tables as oof/<slide>.npz with the
# same keys the tables use (p6, p5, p3_unit, p3); SlideTable reads them for train slides when
# RLSCREEN_OOF=1. Val and test keep the full-data heads.
# Usage: python -m experiments.rlscreen.oof --heads L6_full L3_unit L3_full [--n_folds 5]

import argparse
import json
import os

import numpy as np

from experiments.lowsep import paths as LP
from experiments.lowsep.cascade_prep import load_head
from experiments.lowsep.maps import SlideMaps
from experiments.lowsep.train import fold_of, predict
from experiments.rlscreen import paths as P

KEY = {'L6_full': ('p6', 6), 'L5_full': ('p5', 5), 'L3_unit': ('p3_unit', 3), 'L3_full': ('p3', 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--heads', nargs='+', default=['L6_full', 'L3_unit', 'L3_full'])
    ap.add_argument('--n_folds', type=int, default=5)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    d = P.units_dir(LP.BACKBONE)
    chosen = json.load(open(f'{d}/heads.json'))
    out = f'{d}/oof'
    os.makedirs(out, exist_ok=True)
    names = sorted(f[:-4] for f in os.listdir(d) if f.endswith('.npz') and str(np.load(f'{d}/{f}')['split']) == 'train')
    for role in a.heads:
        key, level = KEY[role]
        tag = chosen[role][1]
        heads = [load_head(f'{tag}_f{k}of{a.n_folds}', a.device) for k in range(a.n_folds)]
        for name in names:
            f = f'{out}/{name}.npz'
            rec = dict(np.load(f)) if os.path.exists(f) else {}
            hu, wu = np.load(f'{d}/{name}.npz')['tissue'].shape
            head = heads[fold_of(name, a.n_folds)]
            rec[key] = predict(head, SlideMaps(name, level), a.device)[:, :hu, :wu].transpose(1, 2, 0).astype(np.float16)
            np.savez(f, **rec)
        print(f'{role} ({tag}): {len(names)} train slides scored out of fold', flush=True)


if __name__ == '__main__':
    main()
