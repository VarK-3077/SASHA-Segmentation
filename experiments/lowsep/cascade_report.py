# Side-by-side table of the full level-2 token head and the screen + segment cascades: mean and
# median wall seconds per slide (with the tissue / screen / segment split for cascades, loader
# wait / GPU for the full run), fraction of tissue units segmented, and the score_maps results
# at the token head's validation-chosen operating point. Writes cascade_report.md.

import glob
import json
import os
import sys

import numpy as np

from experiments.lowsep import paths as P


def times(d):
    rows = [json.load(open(f)) for f in sorted(glob.glob(f'{d}/*_time.json'))]
    return rows


def main(root):
    lines = ['# Screen + segment vs full level-2 token head (129 test slides, A100, cold page cache per slide)', '',
             '| run | wall s/slide mean | median | tissue | screen | segment / GPU | units segmented | FROC | Dice (tumour slides) |',
             '|' + '---|' * 9]
    for name in ['full_l2'] + sorted(os.path.basename(d) for d in glob.glob(f'{root}/cascade_L*')):
        d = f'{root}/{name}'
        t = times(d)
        if not t:
            continue
        sc = json.load(open(f'{d}/scores.json'))[0] if os.path.exists(f'{d}/scores.json') else {}
        wall = [r['wall'] for r in t]
        if name == 'full_l2':
            split = f"- | - | {np.mean([r['gpu'] for r in t]):.2f} (gpu), {np.mean([r['wait_io'] for r in t]):.2f} io"
            seg = '100 %'
        else:
            split = f"{np.mean([r['tissue'] for r in t]):.2f} | {np.mean([r['screen'] for r in t]):.2f} | {np.mean([r['segment'] for r in t]):.2f}"
            seg = f"{100 * sum(r['zoomed'] for r in t) / max(1, sum(r['units'] for r in t)):.1f} %"
        lines.append(f"| {name} | {np.mean(wall):.2f} | {np.median(wall):.2f} | {split} | {seg} | "
                     f"{sc.get('froc', float('nan')):.3f} | {sc.get('dice', float('nan')):.3f} |")
    text = '\n'.join(lines)
    open(f'{root}/cascade_report.md', 'w').write(text)
    print(text)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else f'{P.ROOT}/cascade')
