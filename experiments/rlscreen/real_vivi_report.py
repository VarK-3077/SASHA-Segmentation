# One table of the real runs under a directory: wall seconds per slide, L2 units, painted units, and the
# clean_sweep scores (Dice and lesion Dice of the cleaned map at t0.4 seed 0.9, FROC of the raw map at 0.4).
# Usage: python -m experiments.rlscreen.real_vivi_report DIR

import glob
import json
import os
import sys

import numpy as np

R = sys.argv[1]
print('| run | slides | wall s/slide | median | L2 units / slide | painted / slide | Dice cleaned | lesion Dice <0.5 / 0.5-2 / >=2 | FROC raw |\n|---|---|---|---|---|---|---|---|---|')
for d in sorted(glob.glob(f'{R}/*/')):
    t = [json.load(open(f)) for f in glob.glob(f'{d}test_*_time.json')]
    if not t:
        continue
    w = np.array([x['wall'] for x in t])
    get = lambda k: np.mean([x.get(k, x.get('crops', 0) if k == 'zoomed' else 0) for x in t])
    v = json.load(open(f'{d}clean.json')) if os.path.exists(f'{d}clean.json') else []
    c = next((r for r in v if r['variant'] == 't0.4_seed0.9'), None)
    raw = next((r for r in v if r['variant'] == 't0.4'), None)
    sc = f"{c['dice']:.3f} | {' / '.join(f'{x:.3f}' for x in c['lesion_dice'])}" if c else '- | -'
    fr = f"{(raw or c)['froc']:.3f}" if c else '-'
    print(f'| {os.path.basename(d[:-1])} | {len(t)} | {w.mean():.2f} | {np.median(w):.2f} | {get("zoomed"):.0f} | {get("painted"):.0f} | {sc} | {fr} |')
