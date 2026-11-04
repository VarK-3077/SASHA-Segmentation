# Offline sweep of the teacher's level-3 zoom threshold with the screen threshold held fixed (the
# dropout head's outputs were computed for one screened set), so level-3 heads can be compared at
# equal zoom rather than at one probability. Ideal segmenter, test slides; with border following.
# Usage: [RLSCREEN_P3=<head>] python -m experiments.rlscreen.zoom_sweep --design coarse --g_level 6

import argparse

from experiments.rlscreen import teacher as T
from experiments.rlscreen.evaluate import agg
from experiments.rlscreen.rollout import load_tables, play


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--design', required=True)
    ap.add_argument('--g_level', type=int, default=6)
    ap.add_argument('--backbone', default='kaiko_vitl14')
    ap.add_argument('--tau', type=float, default=0.5, help='screen threshold (coarse) / zoom threshold (full) when not swept')
    ap.add_argument('--zoom_taus', type=float, nargs='*', default=[0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7])
    ap.add_argument('--label', default='')
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    test = load_tables('test', a.design, a.g_level, a.backbone, a.device)
    for tz in a.zoom_taus:
        if a.design == 'coarse':
            actor = lambda ep, tz=tz: T.act(ep, tau=a.tau, follow=True, tau_zoom=tz)
        else:
            actor = lambda ep, tz=tz: T.act(ep, tau=tz, follow=True)
        s = agg([play(t, 1.0, actor)[0] for t in test])
        print(f'| {a.label} | {tz:g} | {s["ideal_dice_tumour"]:.3f} | {s["lesions_found"]:.3f} | {s["zoom_frac"]:.3f} | '
              f'{s["screen_frac"]:.3f} | {s["seconds"]:.2f} |', flush=True)


if __name__ == '__main__':
    main()
