# Level-3 head outputs for the coarse design from a context head trained with unit dropout
# (lowsep train.py --unit_drop): per slide, the units the global view flags (p_tumour >= tau, the
# teacher's screen) are the only ones read at level 3, every other unit's tokens are zeroed, and the
# head scores the read units with whatever neighbours were read. Saved beside the slide tables as
# p3drop_L<g>_<head>/<slide>.npy; SlideTable uses them with RLSCREEN_P3=<head>. Also prints, per split,
# the AUC (tumour unit vs not, read tissue units only) of the per-unit head, the dropout head on the
# partial screen, and the context head on the full screen (the unreachable reference).
# Usage: LOWSEP_BACKBONE=kaiko_vitl14 python -m experiments.rlscreen.p3drop --head L3_ctx_d128_udrop_s0 --g_level 6

import argparse
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.lowsep import paths as LP
from experiments.lowsep.cascade_prep import load_head
from experiments.lowsep.maps import SlideMaps
from experiments.lowsep.train import predict
from experiments.rlscreen import paths as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--head', required=True)
    ap.add_argument('--g_level', type=int, default=6)
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    d = P.units_dir(LP.BACKBONE)
    out = f'{d}/p3drop_L{a.g_level}_{a.head}'
    os.makedirs(out, exist_ok=True)
    head = load_head(a.head, a.device)
    acc = {}
    for f in sorted(os.listdir(d)):
        if not f.endswith('.npz'):
            continue
        name = f[:-4]
        z = np.load(f'{d}/{f}')
        tissue = z['tissue']
        hu, wu = tissue.shape
        read = tissue & (1 - z[f'p{a.g_level}'][..., 0].astype(np.float32) >= a.tau)
        p = predict(head, SlideMaps(name, 3), a.device, keep=read)[:, :hu, :wu].transpose(1, 2, 0).astype(np.float16)
        np.save(f'{out}/{name}.npy', p)
        lab = z['bucket'] > 0
        r = acc.setdefault(str(z['split']), dict(y=[], unit=[], drop=[], full=[]))
        r['y'].append(lab[read]); r['drop'].append(1 - p[..., 0][read])
        r['unit'].append(1 - z['p3_unit'][..., 0][read]); r['full'].append(1 - z['p3'][..., 0][read])
    for split, r in sorted(acc.items()):
        y = np.concatenate(r['y'])
        if y.min() == y.max():
            continue
        aucs = {k: round(float(roc_auc_score(y, np.concatenate(r[k]).astype(np.float32))), 4) for k in ('unit', 'drop', 'full')}
        print(f'{split}: {len(y)} read units, {int(y.sum())} tumour; AUC per-unit {aucs["unit"]}, dropout head {aucs["drop"]}, '
              f'full-screen context head {aucs["full"]}', flush=True)


if __name__ == '__main__':
    main()
