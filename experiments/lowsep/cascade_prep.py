# Validation-only choices for the screen + segment cascade: per screen level, the trained head
# (seed 0 of each config) whose p(tumour) best separates tumour units from normal units on the
# val slides (AUC), and the p(tumour) threshold that zooms each target fraction of val tissue
# units. Written to cascade_cfg.json; nothing here looks at test slides.

import glob
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from experiments.lowsep import paths as P
from experiments.lowsep.maps import SlideMaps
from experiments.lowsep.model import ScreenHead
from experiments.lowsep.train import names_of, predict

BUDGETS = (0.1, 0.2)


def load_head(tag, device):
    args = json.load(open(f'{P.run_dir(tag)}/args.json'))
    head = ScreenHead(P.tokens_per_unit(args['level']), ctx=bool(args['ctx']), d=args.get('d', 256)).to(device).eval()
    head.load_state_dict(torch.load(f'{P.run_dir(tag)}/best.pt', map_location=device)['model'])
    return head


def head_tags(level, ctx=None):
    """Seed-0 run tags of the current backbone at a level (ctx: None = any, True / False = only that head)."""
    pre = f'L{level}{P.backbone_tag()}_'
    out = []
    for d in sorted(glob.glob(f'{P.ROOT}/runs/{pre}*_s0')):
        rest = os.path.basename(d)[len(pre):]
        if rest.startswith('ctx') and ctx is not False or rest.startswith('noctx') and ctx is not True:
            out.append(os.path.basename(d))
    return out


def val_scores(head, level, device):
    p, y = [], []
    for n in names_of('val', level):
        m = SlideMaps(n, level)
        prob = predict(head, m, device)
        p.append(1 - prob[0][m.tissue]); y.append(m.bucket[m.tissue] > 0)
    return np.concatenate(p), np.concatenate(y)


def main(device='cuda:0'):
    cfg = {}
    for level in (3, 4):
        best = None
        for tag in head_tags(level):
            p, y = val_scores(load_head(tag, device), level, device)
            auc = float(roc_auc_score(y, p))
            print(f'L{level} {tag}: val AUC tumour-vs-normal units {auc:.4f}', flush=True)
            if best is None or auc > best[0]:
                best = (auc, tag, p)
        auc, tag, p = best
        cfg[f'L{level}'] = {'tag': tag, 'val_auc': auc,
                            'thresholds': {str(b): float(np.quantile(p, 1 - b)) for b in BUDGETS}}
        print(f'L{level}: chose {tag}, thresholds {cfg[f"L{level}"]["thresholds"]}', flush=True)
    json.dump(cfg, open(f'{P.ROOT}/cascade_cfg.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
