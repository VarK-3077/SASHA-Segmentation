# The one plot of the project: mask quality against the fraction of tiles zoomed, on the test
# slides, for every selector that uses the same low-res and high-res scorers: random order,
# greedy (highest low-res score), uncertainty (highest entropy), the supervised router (core by
# threshold, zoom the most uncertain), ground-truth order, plus the single points of the oracle
# routing and of every RL run that has a results json. Writes frontier_L<level>.{json,png}.

import argparse
import glob
import json
import os

import numpy as np

from experiments.rlbin import paths as P
from experiments.rlbin.env import CORE, ZOOMED, entropy
from experiments.rlbin.metrics import aggregate, baseline_status, episode_metrics
from experiments.rlbin.train import load_slides

FRACS = (0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
COLOURS = {'random': '#52514e', 'greedy': '#eda100', 'uncertainty': '#2a78d6', 'supervised router': '#1baf7a',
           'ground-truth order': '#6250d6', 'oracle routing': '#e34948', 'RL agent': '#eb6834'}


def order_status(slide, order, k):
    status = np.zeros(slide.n, np.int8)
    status[order[:k]] = ZOOMED
    return status


def curve(slides, kind, rng):
    rows = {f: [] for f in FRACS}
    for s in slides:
        pbar = s.p_lr.mean(1)
        if kind == 'random':
            order = rng.permutation(s.n)
        elif kind == 'greedy':
            order = np.argsort(-pbar)
        elif kind == 'uncertainty':
            order = np.argsort(-entropy(s.p_lr))
        elif kind == 'ground-truth order':
            order = np.lexsort((-entropy(s.p_lr), -(s.gt.mean(1) > 0)))
        for f in FRACS:
            if kind == 'supervised router':
                status = baseline_status(s, 'sup_router', f)
            else:
                status = order_status(s, order, int(round(f * s.n)))
            rows[f].append(episode_metrics(s, status, 0))
    return {f: aggregate(r) for f, r in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--level', type=int, default=3, choices=(3, 2))
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    test = load_slides('test', a.device, trusted_only=False, level=a.level)
    rng = np.random.default_rng(0)
    out = {'curves': {}, 'points': {}}
    for kind in ('random', 'greedy', 'uncertainty', 'supervised router', 'ground-truth order'):
        c = curve(test, kind, rng)
        out['curves'][kind] = {str(f): {'zoom_frac': f, 'dice_all': v['dice_all'], 'dice_tumour': v['dice_tumour']} for f, v in c.items()}
        print(kind, {f: round(v['dice_tumour'], 3) for f, v in c.items()}, flush=True)
    orc = aggregate([episode_metrics(s, baseline_status(s, 'oracle_route'), 0) for s in test])
    out['points']['oracle routing'] = {'zoom_frac': orc['zoom_frac'], 'dice_all': orc['dice_all'], 'dice_tumour': orc['dice_tumour']}
    for f in sorted(glob.glob(os.path.join(P.ROOT, 'results_*.json'))):
        r = json.load(open(f))
        if r.get('level', 3) != a.level:
            continue
        tag = os.path.basename(f)[8:-5]
        out['points'][f'RL {tag}'] = {'zoom_frac': r['agent']['zoom_frac'], 'dice_all': r['agent']['dice_all'],
                                      'dice_tumour': r['agent']['dice_tumour']}
    json.dump(out, open(os.path.join(P.ROOT, f'frontier_L{a.level}.json'), 'w'), indent=1)
    plot(out, a.level)


def plot(out, level):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    for ax, key, title in zip(axes, ('dice_tumour', 'dice_all'), ('tumour slides (49)', 'all test slides (129)')):
        for kind, c in out['curves'].items():
            xs = [v['zoom_frac'] for v in c.values()]
            ys = [v[key] for v in c.values()]
            ax.plot(xs, ys, color=COLOURS[kind], lw=2, marker='o', ms=4, label=kind)
        p = out['points']['oracle routing']
        ax.scatter([p['zoom_frac']], [p[key]], color=COLOURS['oracle routing'], s=60, zorder=5, label='oracle routing')
        first = True
        for name, p in out['points'].items():
            if name.startswith('RL '):
                ax.scatter([p['zoom_frac']], [p[key]], color=COLOURS['RL agent'], marker='^', s=55, zorder=5,
                           label='RL agent' if first else None)
                ax.annotate(name[3:], (p['zoom_frac'], p[key]), fontsize=7, xytext=(4, 3), textcoords='offset points')
                first = False
        ax.set_xscale('symlog', linthresh=0.01)
        ax.set_xlabel('fraction of tiles zoomed')
        ax.set_ylabel('Dice (512 px sub-patches)')
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f'Quality vs zoom fraction, level {level}', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(P.ROOT, f'frontier_L{level}.png'))


if __name__ == '__main__':
    main()
