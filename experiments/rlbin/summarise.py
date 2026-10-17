# Gathers every results_<tag>.json into one table so runs can be compared side by side: test
# Dice, normal-slide cleanliness, zoom fraction, and what the agent did with each bucket.
# Seed groups (tags ending in _s<k>) are also reported as mean ± std.

import glob
import json
import os
import re

import numpy as np

from experiments.rlbin import paths as P

COLS = ('dice_all', 'dice_tumour', 'normal_slides_clean', 'zoom_frac', 'boundary_zoom', 'small_zoom',
        'core_zoom', 'core_core', 'normal_zoom')


def row(name, agg):
    return '| ' + name + ' | ' + ' | '.join('-' if agg.get(c) is None else f'{agg[c]:.3f}' for c in COLS) + ' |'


def main():
    res = {}
    for f in sorted(glob.glob(os.path.join(P.ROOT, 'results_*.json'))):
        tag = os.path.basename(f)[8:-5]
        res[tag] = json.load(open(f))
    lines = ['# RL binning runs, test slides (129)', '',
             '| run | Dice all | Dice tumour | normal clean | zoom frac | bnd zoomed | small zoomed | core zoomed | core marked | normal zoomed |',
             '|' + '---|' * (len(COLS) + 1)]
    for tag, r in res.items():
        lines.append(row(f"{tag} (L{r.get('level', 3)}, state {r.get('state', 'none')})", r['agent']))
    groups = {}
    for tag, r in res.items():
        m = re.match(r'(.+)_s\d+(_balanced)?$', tag)
        if m:
            groups.setdefault(m.group(1) + (m.group(2) or ''), []).append(r['agent'])
    if groups:
        lines += ['', '## seed groups (mean ± std)', '']
        for g, aggs in groups.items():
            cells = []
            for c in COLS:
                v = [a[c] for a in aggs if a.get(c) is not None]
                cells.append(f'{np.mean(v):.3f} ± {np.std(v):.3f}' if v else '-')
            lines.append(f'| {g} (n={len(aggs)}) | ' + ' | '.join(cells) + ' |')
    any_r = next(iter(res.values()))
    lines += ['', '## baselines (same scorers, from the first results file)', '']
    for k, v in any_r.items():
        if k not in ('agent', 'checkpoint', 'state', 'level'):
            lines.append(row(k, v))
    text = '\n'.join(lines)
    open(os.path.join(P.ROOT, 'summary.md'), 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
