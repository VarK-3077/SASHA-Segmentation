# Collects every run's test.json into report.md: per level and head (context on / off), mean ±
# std over seeds of bucket recall, recall at 99 % specificity, AUC vs normal, and the routing
# rows (Dice with an ideal segmenter, zoom fraction, total cost relative to running the
# segmenter everywhere), plus the seed-0 confusion matrix and measured screen seconds per slide.

import glob
import json
import os
from collections import defaultdict

import numpy as np

from experiments.lowsep import paths as P

NAMES = ('normal', 'core', 'boundary', 'small')


def ms(v):
    v = [x for x in v if x is not None]
    return f'{np.mean(v):.3f} ± {np.std(v):.3f}' if len(v) > 1 else (f'{v[0]:.3f}' if v else '-')


def screen_seconds(level):
    f = f'{P.tok_dir(level)}/times.json'
    if not os.path.exists(f):
        return None
    t = json.load(open(f))
    test = [v['sec'] for n, v in t.items() if n.startswith('test_')]
    return float(np.mean(test)) if test else None


def main():
    runs = defaultdict(list)
    for f in sorted(glob.glob(f'{P.ROOT}/runs/*/test.json')):
        r = json.load(open(f))
        runs[(r['level'], r['ctx'])].append(r)
    L = ['# Low-level screen separability', '',
         'Unit = 2048 px level-0 crop, the token-head segmenter input. Screen = frozen Kaiko ViT-L/14 on 256 px crops at',
         'level L (m x m units per crop, m = 2^(L-3)), PCA-256 tokens, trained head. Test = every tissue unit of the 129',
         'test slides. Cost = backbone forwards relative to segmenting every unit (4 per unit). Mean ± std over seeds.', '',
         '## Bucket recall (argmax) and AUC vs normal on tumour slides', '',
         '| level | head | seeds | normal | core | boundary | small | small @99 % spec | boundary @99 % spec | AUC small | AUC boundary | screen cost | screen s/slide |',
         '|' + '---|' * 13]
    for (lv, ctx), rs in sorted(runs.items()):
        sec = screen_seconds(lv)
        L.append(f"| L{lv} | {'context' if ctx else 'no context'} | {len(rs)} | " + ' | '.join(ms([r['recall'][n] for r in rs]) for n in NAMES)
                 + f" | {ms([r['recall_at_spec']['0.99'].get('small') for r in rs])} | {ms([r['recall_at_spec']['0.99'].get('boundary') for r in rs])}"
                 + f" | {ms([r['auc_vs_normal_tumour_slides']['small'] for r in rs])} | {ms([r['auc_vs_normal_tumour_slides']['boundary'] for r in rs])}"
                 + f" | {rs[0]['screen_cost']:.4f} | {'-' if sec is None else f'{sec:.1f}'} |")
    routes = list(next(iter(runs.values()))[0]['routing'].keys()) if runs else []
    for route in routes:
        L += ['', f'## Routing with an ideal segmenter: {route}', '',
              '| level | head | Dice tumour | Dice all | normal slides clean | zoom fraction | total cost |', '|' + '---|' * 7]
        for (lv, ctx), rs in sorted(runs.items()):
            g = [r['routing'][route] for r in rs]
            L.append(f"| L{lv} | {'context' if ctx else 'no context'} | {ms([x['dice_tumour'] for x in g])} | {ms([x['dice_all'] for x in g])} | "
                     f"{ms([x['normal_clean'] for x in g])} | {ms([x['zoom_frac'] for x in g])} | {ms([x['total_cost'] for x in g])} |")
    L += ['', '## Confusion matrices (seed 0; rows true bucket, columns predicted)', '']
    for (lv, ctx), rs in sorted(runs.items()):
        r = sorted(rs, key=lambda x: x['seed'])[0]
        L += [f"### L{lv}, {'context' if ctx else 'no context'}", '', '| true \\ predicted | ' + ' | '.join(NAMES) + ' | units |', '|' + '---|' * 6]
        for i, n in enumerate(NAMES):
            row = r['confusion'][i]
            L.append(f'| {n} | ' + ' | '.join(f'{v:,}' for v in row) + f' | {sum(row):,} |')
        L.append('')
    open(f'{P.ROOT}/report.md', 'w').write('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
