# One table for every real-segmenter run under /data2/venkatavks/rlscreen/real: wall seconds per
# slide with the tissue / global pass / rounds split, units screened and zoomed, rounds, and the
# clean_sweep scores (raw map at 0.4 and cleaned t0.4 seed 0.9) next to the full level-2 run and
# the pipelined supervised cascade. Writes real_report.md.

import glob
import json
import os

import numpy as np

ROOT = '/data2/venkatavks/rlscreen/real'
REFS = {'full level-2 token head': ('/data2/venkatavks/tokhead_l2/out_timed_a100', '/data2/venkatavks/rlscreen/ref_l2_clean_test.json'),
        'supervised L3 cascade, pipelined, 17 %': ('/data2/venkatavks/lowsep/cascade/cascadepipe_L3_b0.2', None)}


def score(path, variant):
    if not path or not os.path.exists(path):
        return None
    return next((r for r in json.load(open(path)) if r['variant'] == variant), None)


def cells(r):
    return '- | - | - | -' if r is None else f"{r['dice']:.3f} | {r['pooled_all']:.3f} | {' / '.join(f'{x:.3f}' for x in r['lesion_dice'])} | {r['froc']:.3f}"


def main():
    lines = ['# RL screening with the real level-2 segmenter (129 test slides, A100, solo, cold cache)', '',
             '| run | wall s/slide | median | tissue / global / rounds | screened | zoomed | rounds | Dice cleaned | pooled | lesion Dice <0.5/0.5-2/>=2 | FROC raw |',
             '|' + '---|' * 11]
    runs = [(k, v[0], v[1] or f'{v[0]}/clean.json') for k, v in REFS.items()]
    runs += [(os.path.basename(d), d, f'{d}/clean.json') for d in sorted(glob.glob(f'{ROOT}/*')) if os.path.isdir(d)]
    for name, d, sc in runs:
        t = [json.load(open(f)) for f in sorted(glob.glob(f'{d}/test_*_time.json'))]
        if not t:
            continue
        units = sum(r.get('units', 0) for r in t) or 1
        split = (f"{np.mean([r['tissue'] for r in t]):.2f} / {np.mean([r['global_pass'] for r in t]):.2f} / {np.mean([r['rounds_time'] for r in t]):.2f}"
                 if 'global_pass' in t[0] else '-')
        scr = f"{100 * sum(r.get('screened', 0) for r in t) / units:.1f} %" if 'screened' in t[0] else '-'
        zm = f"{100 * sum(r.get('zoomed', 0) for r in t) / units:.1f} %" if 'zoomed' in t[0] else '100 %'
        rounds = f"{np.mean([r['rounds'] for r in t]):.1f}" if 'rounds' in t[0] else '-'
        cl, raw = score(sc, 't0.4_seed0.9'), score(sc, 't0.4')
        froc = '-' if raw is None else f"{raw['froc']:.3f}"
        lines.append(f"| {name} | {np.mean([r['wall'] for r in t]):.2f} | {np.median([r['wall'] for r in t]):.2f} | {split} | {scr} | {zm} | {rounds} | "
                     f"{cells(cl).rsplit(' | ', 1)[0]} | {froc} |")
    text = '\n'.join(lines)
    open(f'{ROOT}/real_report.md', 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
