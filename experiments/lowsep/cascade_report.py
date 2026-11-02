# Side-by-side table of the full level-2 token head (the pfa-analysis run, read from its own
# solo A100 timing jsons and its clean_sweep test scores, not rerun) and the screen + segment
# cascades: wall seconds per slide with the tissue / screen / segment split, fraction of units
# segmented, and the same clean_sweep variants (raw map at 0.4, cleaned t0.4 seed 0.9).
# Writes cascade_report.md.

import glob
import json
import os
import sys

import numpy as np

from experiments.lowsep import paths as P

BASE = '/data2/venkatavks/tokhead_l2'
VARIANTS = ('t0.4', 't0.4_seed0.9')


def score_row(path, variant):
    for r in json.load(open(path)):
        if r['variant'] == variant:
            return r
    return None


def fmt_scores(r):
    if r is None:
        return '- | - | - | - | -'
    return (f"{r['dice']:.3f} | {r['pooled_all']:.3f} | {' / '.join(f'{x:.3f}' for x in r['lesion_dice'])} | "
            f"{r['fp_per_slide']:.1f} | {r['froc']:.3f}")


def main(root):
    head = ('| run | mask | wall s/slide mean | median | tissue / screen / segment | units segmented | per-slide Dice | '
            'pooled Dice, all slides | lesion Dice <0.5 / 0.5-2 / >=2 mm | FP comps/slide | FROC |')
    lines = ['# Screen + segment vs full level-2 token head (129 test slides, A100, solo, cold page cache per slide)', '',
             'Full row: the existing level-2 run (`tokhead_l2`, timing `out_timed_a100`, scores `clean_test.json`), not rerun.',
             'Masks: raw map at the validation-chosen 0.4, and cleaned (components >= 0.4 kept if their peak >= 0.9).', '',
             head, '|' + '---|' * 11]
    t = [json.load(open(f)) for f in sorted(glob.glob(f'{BASE}/out_timed_a100/test_*_time.json'))]
    for v in VARIANTS:
        lines.append(f"| full L2 token head | {v} | {np.mean([r['wall'] for r in t]):.2f} | {np.median([r['wall'] for r in t]):.2f} | "
                     f"- / - / {np.mean([r['wall'] for r in t]):.2f} | 100 % | {fmt_scores(score_row(f'{BASE}/clean_test.json', v))} |")
    for d in sorted(glob.glob(f'{root}/cascade_L*')) + sorted(glob.glob(f'{root}/cascadepipe_L*')):
        t = [json.load(open(f)) for f in sorted(glob.glob(f'{d}/test_*_time.json'))]
        if not t:
            continue
        seg = 100 * sum(r['zoomed'] for r in t) / max(1, sum(r['units'] for r in t))
        if 'screen' in t[0]:
            split = f"{np.mean([r['tissue'] for r in t]):.2f} / {np.mean([r['screen'] for r in t]):.2f} / {np.mean([r['segment'] for r in t]):.2f}"
        else:  # pipelined: screen and segment overlap, the tail is segmentation left after the screen ends
            split = f"{np.mean([r['tissue'] for r in t]):.2f} / {np.mean([r['screen_phase'] for r in t]):.2f} (overlapped) / {np.mean([r['tail'] for r in t]):.2f} tail"
        for v in VARIANTS:
            sc = score_row(f'{d}/clean.json', v) if os.path.exists(f'{d}/clean.json') else None
            lines.append(f"| {os.path.basename(d)} | {v} | {np.mean([r['wall'] for r in t]):.2f} | {np.median([r['wall'] for r in t]):.2f} | "
                         f"{split} | {seg:.1f} % | {fmt_scores(sc)} |")
    text = '\n'.join(lines)
    open(f'{root}/cascade_report.md', 'w').write(text)
    print(text)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else f'{P.ROOT}/cascade')
