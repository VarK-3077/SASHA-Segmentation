# Where the supervised rule spends its units, and how well a slide's best L3 score separates tumour slides
# from normal ones, on the out-of-fold train slides: per slide type the tissue units, L3 screens, L2 zooms
# (how many of them hold tumour) and seconds; then, for each L3 score, the AUC of "slide holds a non-ITC
# lesion" from the slide's maximum over screened units, and the rank of the first tumour unit among them.
# With --counts test it prints the rule's label-free unit counts on test (no scores are read).
# Usage: python -m experiments.rlpolicy.diag --tok3 t0 [--counts test]

import argparse

import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.rlpolicy import sim as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tok3', default=None)
    ap.add_argument('--counts', default=None)
    a = ap.parse_args()
    if a.counts:
        sl = S.load(a.counts)
        r = [S.rule(s) for s in sl]
        print(a.counts, 'tissue', np.mean([s.n for s in sl]), 'L3', np.mean([x['n_l3'] for x in r]), 'L2', np.mean([x['n_l2'] for x in r]), 'paint', np.mean([x['n_paint'] for x in r]))
        return
    sl = S.load('train', a.tok3)
    r = [S.rule(s) for s in sl]
    for tum in (False, True):
        ii = [i for i, s in enumerate(sl) if s.tumour == tum]
        hold = np.mean([(r[i]['zoom'] & (sl[i].frac > 0)).sum() for i in ii])
        print('tumour' if tum else 'normal', len(ii), 'slides: tissue %.0f, L3 %.0f, L2 %.0f (holding tumour %.1f), painted %.1f, %.2f s, FP comps %.2f' % (
            np.mean([sl[i].n for i in ii]), np.mean([r[i]['n_l3'] for i in ii]), np.mean([r[i]['n_l2'] for i in ii]), hold,
            np.mean([r[i]['n_paint'] for i in ii]), np.mean([r[i]['sec'] for i in ii]), np.mean([r[i]['n_fp'] for i in ii])))
    y = np.array([len(s.lesions) > 0 for s in sl])
    keys = ['p3'] + (['t_unit', 't_top4', 't_max', 't_mean'] if a.tok3 else [])
    for k in keys:
        mx, rank, n_above = [], [], []
        for s in sl:
            scr = s.tissue & (s.pg >= 0.6)
            v = getattr(s, k)
            mx.append(v[scr].max() if scr.any() else 0.0)
            pos = scr & (s.frac > 0)
            if len(s.lesions) and pos.any():
                rank.append(int((v[scr] > v[pos].max()).sum()) + 1)
            if not s.tumour:
                n_above.append([(v[scr] >= t).sum() for t in (0.5, 0.9, 0.99)])
        rank = np.array(rank)
        print(f'{k:7s} slide AUC from max screened score {roc_auc_score(y, mx):.4f}; first tumour unit rank: median {np.median(rank):.0f}, 90th pct {np.percentile(rank, 90):.0f}, '
              f'max {rank.max()}, >20: {(rank > 20).sum()}/{len(rank)}; normal slides units >= .5/.9/.99: ' + ' / '.join(f'{x:.1f}' for x in np.mean(n_above, 0)))


if __name__ == '__main__':
    main()
