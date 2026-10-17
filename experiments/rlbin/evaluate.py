# Scores the best checkpoint on the test slides next to the baselines that use the same
# scorers: low-res mask only, zoom everything, a supervised router at the agent's own zoom
# fraction and at fixed fractions, and the label-using oracle routing. Writes report.md.

import argparse
import json
import os

import numpy as np
import torch

from experiments.rlbin import paths as P
from experiments.rlbin.metrics import aggregate, baseline_status, dice_of, episode_metrics, lr_only_dice
from experiments.rlbin.policy import Policy
from experiments.rlbin.train import evaluate, load_slides
from experiments.sep import buckets as B


def baseline_rows(slides, kind, zoom_frac=0.0):
    return aggregate([episode_metrics(s, baseline_status(s, kind, zoom_frac), 0) for s in slides])


def fmt(agg):
    cells = [agg['dice_all'], agg['dice_tumour'], agg['normal_slides_clean'], agg['zoom_frac']]
    cells += [agg.get(f'{n}_zoom') for n in ('boundary', 'small', 'core', 'normal')]
    cells += [agg.get(f'{n}_core') for n in ('boundary', 'small', 'core', 'normal')]
    return ' | '.join('-' if c is None else f'{c:.3f}' for c in cells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--tag', default='l3')
    a = ap.parse_args()
    test = load_slides('test', a.device, trusted_only=False)
    policy = Policy().to(a.device)
    policy.load_state_dict(torch.load(os.path.join(P.CKPT, f'best_{a.tag}.pt'), map_location=a.device))
    policy.eval()
    agent, rows = evaluate(policy, test, a.device)
    results = {'agent': agent}
    lines = ['# RL binning, level 3, test slides (129)', '',
             'Dice over 512 px sub-patches; untouched tiles are negative. Columns after zoom frac: fraction of each',
             'bucket the method zoomed (boundary, small, core, normal), then the fraction it marked core.', '',
             '| method | Dice all | Dice tumour | normal slides clean | zoom frac | z-bnd | z-small | z-core | z-normal | c-bnd | c-small | c-core | c-normal |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|',
             f'| RL agent (best val) | {fmt(agent)} |']
    tum = [s for s in test if s.gt_pos > 0]
    lr_all = float(np.mean([lr_only_dice(s) for s in test]))
    lr_tum = float(np.mean([lr_only_dice(s) for s in tum]))
    lines.append(f'| low-res mask only, no zoom | {lr_all:.3f} | {lr_tum:.3f} | - | 0 | - | - | - | - | - | - | - | - |')
    for kind, zf in (('zoom_all', 0), ('sup_router', agent['zoom_frac']), ('sup_router', 0.1), ('sup_router', 0.2),
                     ('sup_router', 0.5), ('oracle_route', 0)):
        agg = baseline_rows(test, kind, zf)
        label = kind if kind != 'sup_router' else f'supervised router, zoom {zf:.2f}'
        results[label] = agg
        lines.append(f'| {label} | {fmt(agg)} |')
    lines += ['', '## per-slide agent rows', '', '| slide | n | tumour | Dice | zoom frac | steps frac |', '|---|---|---|---|---|---|']
    lines += [f"| {s.name} | {r['n']} | {int(r['tumour'])} | {r['dice']:.3f} | {r['zoom_frac']:.2f} | {r['steps_frac']:.2f} |"
              for s, r in zip(test, rows)]
    text = '\n'.join(lines)
    open(os.path.join(P.ROOT, f'report_{a.tag}.md'), 'w').write(text)
    json.dump(results, open(os.path.join(P.ROOT, f'results_{a.tag}.json'), 'w'), indent=1)
    print('\n'.join(lines[:16]))


if __name__ == '__main__':
    main()
