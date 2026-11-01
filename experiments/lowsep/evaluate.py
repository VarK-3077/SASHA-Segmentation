# Scores a trained screen head on every tissue unit of the 129 test slides, three ways: the 4x4
# bucket confusion (argmax), recall of each tumour bucket at 99 % / 95 % specificity on normal
# units, and routing with an ideal segmenter (zoomed units get their ground truth, selected
# units become all tumour, left units stay empty; Dice from the units' tumour fractions). Cost
# is in backbone forwards relative to running the token-head segmenter on every unit (4 per
# unit): the screen costs 1/(4 m^2) of that, plus the zoomed fraction.

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from experiments.lowsep import paths as P
from experiments.lowsep.maps import SlideMaps
from experiments.lowsep.model import ScreenHead
from experiments.lowsep.routing import ROUTES, route_scores
from experiments.lowsep.train import names_of, predict

SPECS = (0.99, 0.95)


def recall_at_spec(p_tum, y):
    out = {}
    for spec in SPECS:
        thr = np.quantile(p_tum[y == 0], spec)
        out[str(spec)] = {name: float((p_tum[y == c] >= thr).mean()) for c, name in ((1, 'core'), (2, 'boundary'), (3, 'small')) if (y == c).any()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    args = json.load(open(f'{P.run_dir(a.tag)}/args.json'))
    level = args['level']
    head = ScreenHead(P.tokens_per_unit(level), ctx=bool(args['ctx']), d=args.get('d', 256)).to(a.device).eval()
    ck = torch.load(f'{P.run_dir(a.tag)}/best.pt', map_location=a.device)
    head.load_state_dict(ck['model'])
    slides = []
    for n in names_of('test', level):
        m = SlideMaps(n, level)
        prob = predict(head, m, a.device)
        keep = m.tissue
        slides.append({'name': n, 'prob': prob[:, keep].T, 'bucket': m.bucket[keep], 'frac': m.frac[keep].astype(np.float64)})
    y = np.concatenate([s['bucket'] for s in slides])
    p = np.concatenate([s['prob'] for s in slides])
    pred = p.argmax(1)
    conf = np.zeros((4, 4), np.int64)
    np.add.at(conf, (y, pred), 1)
    p_tum = 1 - p[:, 0]
    tum_slides = np.concatenate([np.full(len(s['bucket']), s['frac'].sum() > 0) for s in slides])
    auc = {}
    for c, name in ((2, 'boundary'), (3, 'small')):
        sel = tum_slides & ((y == 0) | (y == c))
        auc[name] = float(roc_auc_score(y[sel] == c, p_tum[sel]))
    m = P.units_per_crop(level)
    screen = 1.0 / (4 * m * m)
    res = {'tag': a.tag, 'level': level, 'ctx': args['ctx'], 'seed': args['seed'], 'config': args.get('name', ''), 'best_step': ck['step'],
           'val_macro_recall': ck['val_macro_recall'], 'n_units': int(len(y)), 'confusion': conf.tolist(),
           'recall': {['normal', 'core', 'boundary', 'small'][c]: float(conf[c, c] / conf[c].sum()) for c in range(4)},
           'recall_at_spec': recall_at_spec(p_tum, y), 'auc_vs_normal_tumour_slides': auc, 'screen_cost': screen,
           'routing': {}}
    for name in ROUTES:
        r = route_scores(slides, name)
        r['total_cost'] = screen + r['zoom_frac']
        res['routing'][name] = r
    json.dump(res, open(f'{P.run_dir(a.tag)}/test.json', 'w'), indent=1)
    rt = res['routing']['argmax']
    print(f"{a.tag}: recall {[round(v, 3) for v in res['recall'].values()]} | small@99 {res['recall_at_spec']['0.99'].get('small', 0):.3f} "
          f"| argmax routing Dice tumour {rt['dice_tumour']:.3f} zoom {rt['zoom_frac']:.3f} cost {rt['total_cost']:.3f}", flush=True)


if __name__ == '__main__':
    main()
