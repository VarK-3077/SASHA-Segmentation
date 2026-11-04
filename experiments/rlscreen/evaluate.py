# Offline test of screening policies with the ideal segmenter: the trained agent (greedy) at a
# range of lambda, the hand-written teacher with and without lesion following at the given
# thresholds, the supervised cascade at the lowsep validation thresholds (no following), and the
# oracle. Per row: Dice of the ideal mask on tumour slides (2a / (1 + a) for area recall a), area
# and lesion recall, share of non-ITC lesions touched, zoomed / screened units, rounds, and the
# cost-model A100 seconds (all slides, and normal slides alone). Writes offline_<tag>.json / .md.
# Usage: python -m experiments.rlscreen.evaluate --design coarse --g_level 6 [--tag RUN] [--taus 0.5 0.3]

import argparse
import json
import os

import numpy as np
import torch

from experiments.lowsep import paths as LP
from experiments.rlscreen import paths as P
from experiments.rlscreen.policy import ScreenPolicy
from experiments.rlscreen.rollout import load_tables, play

LAMS = (0.1, 0.3, 1.0, 3.0, 10.0)


def agg(res):
    tum = [r for r in res if r['tumour']]
    nor = [r for r in res if not r['tumour']]
    f = lambda k, rs: float(np.mean([r[k] for r in rs if r[k] is not None])) if rs else float('nan')
    return {'ideal_dice_tumour': f('ideal_dice', tum), 'area_recall': f('area_recall', tum), 'lesion_recall': f('lesion_recall', tum),
            'lesions_found': f('lesions_found', tum), 'zoom_frac': f('zoom_frac', res), 'screen_frac': f('screen_frac', res),
            'rounds': f('rounds', res), 'seconds': f('seconds', res), 'seconds_normal': f('seconds', nor), 'seconds_tumour': f('seconds', tum)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--design', choices=('coarse', 'full'), required=True)
    ap.add_argument('--g_level', type=int, default=6)
    ap.add_argument('--backbone', default='kaiko_vitl14')
    ap.add_argument('--tag', default=None, help='trained run to evaluate; omit for baselines only')
    ap.add_argument('--ckpt', default='final')
    ap.add_argument('--taus', type=float, nargs='*', default=[0.5])
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    test = load_tables('test', a.design, a.g_level, a.backbone, a.device)
    rows = {}
    if a.tag:
        args = json.load(open(f'{P.run_dir(a.tag)}/args.json'))
        pol = ScreenPolicy(args['state']).to(a.device).eval()
        pol.load_state_dict(torch.load(f'{P.run_dir(a.tag)}/{a.ckpt}.pt', map_location=a.device))
        for lam in LAMS:
            rows[f'agent lam={lam:g}'] = agg([play(t, lam, 'policy', args['metric'], args['S'], policy=pol, state=args['state'], greedy=True)[0] for t in test])
            print(a.tag, lam, rows[f'agent lam={lam:g}'], flush=True)
    for tau in a.taus:
        for follow in (False, True):
            rows[f'teacher tau={tau:g}{" + follow" if follow else ""}'] = agg([play(t, 1.0, 'teacher', tau=tau, follow=follow)[0] for t in test])
    if a.design == 'full':
        cfg_path = f'{LP.ROOT}/cascade_cfg.json'
        if os.path.exists(cfg_path):
            for b, thr in json.load(open(cfg_path))['L3']['thresholds'].items():
                rows[f'supervised cascade, val budget {b}'] = agg([play(t, 1.0, 'teacher', tau=thr, follow=False)[0] for t in test])
    rows['oracle'] = agg([play(t, 1.0, 'oracle')[0] for t in test])
    name = a.tag or f'baselines_{a.design}_L{a.g_level}_{a.backbone}'
    json.dump(rows, open(f'{P.ROOT}/offline_{name}.json', 'w'), indent=1)
    keys = ('ideal_dice_tumour', 'area_recall', 'lesions_found', 'zoom_frac', 'screen_frac', 'rounds', 'seconds', 'seconds_normal')
    lines = [f'# Offline (ideal segmenter, cost-model A100 s), {name}', '', '| policy | ' + ' | '.join(keys) + ' |', '|' + '---|' * (len(keys) + 1)]
    lines += [f'| {k} | ' + ' | '.join(f'{v[c]:.3f}' for c in keys) + ' |' for k, v in rows.items()]
    open(f'{P.ROOT}/offline_{name}.md', 'w').write('\n'.join(lines))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
