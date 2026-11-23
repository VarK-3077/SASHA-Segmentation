# Room for a sequential L3-read decision: the units the L6 screen passes (pg >= 0.6), read at L3 in descending pg.
# Where do the units the oracle needs sit in that order, and can what the first N reads return tell tumour slides
# from normal ones?
import numpy as np
from sklearn.metrics import roc_auc_score
from experiments.rlpolicy import sim as S, where as W

sl = S.load('train', 't1')
for s in sl:
    s.add_tok3('s2', 's')
rows, feats = [], {N: [] for N in (25, 50, 100, 200)}
for s in sl:
    scr = np.flatnonzero((s.tissue & (s.pg >= 0.6)).ravel())
    order = scr[np.argsort(-s.pg.ravel()[scr])]
    need = (np.load(f'{W.ROOT}/oracle/cascade_reward/{s.name}.npy').ravel() != 0)[order]
    last = int(np.flatnonzero(need).max()) + 1 if need.any() else 0
    rows.append((s.tumour, len(order), int(need.sum()), last))
    for N in feats:
        top = order[:N]
        feats[N].append((s.tumour, float(s.t_top16.ravel()[top].max()) if len(top) else 0.0, float(s.p3.ravel()[top].max()) if len(top) else 0.0, float(s.s_top16.ravel()[top].max()) if len(top) else 0.0))
r = np.array(rows); tum = r[:, 0] == 1
n = len(r)
print(f'screened units/slide: tumour {r[tum, 1].mean():.0f}, normal {r[~tum, 1].mean():.0f}; needed units among them (tumour slides) {r[tum, 2].mean():.1f}')
frac = r[tum, 3] / np.maximum(r[tum, 1], 1)
print('tumour slides: position of the LAST needed unit in pg order, as a fraction of the screened list: quartiles', np.round(np.quantile(frac, [0.1, 0.25, 0.5, 0.75, 0.9]), 2), '; in absolute reads: median', int(np.median(r[tum, 3])), 'mean', round(float(r[tum, 3].mean()), 1))
print(f'reads/slide: all screened {r[:, 1].mean():.0f}; oracle stop after the last needed unit {r[:, 3].mean():.0f} (tumour slides {r[tum, 3].mean():.0f}, normal 0)')
for K in (50, 100, 150, 200, 300):
    print(f'  fixed budget of {K} reads: spends {np.minimum(r[:, 1], K).mean():.0f}/slide, cuts off needed units on {int((tum & (r[:, 3] > K)).sum())} of {int(tum.sum())} tumour slides')
print('can the first N reads tell tumour slides from normal ones? AUC of the slide maximum over the first N reads:')
for N, f in feats.items():
    f = np.array(f)
    print(f'  N {N}: t_top16 max {roc_auc_score(f[:, 0], f[:, 1]):.3f}, p3 max {roc_auc_score(f[:, 0], f[:, 2]):.3f}, s2 top16 max {roc_auc_score(f[:, 0], f[:, 3]):.3f}')
# a stop rule: read N, continue with everything only if the maximum t_top16 so far >= thr
for N in (50, 100):
    f = np.array(feats[N])
    for thr in (0.1, 0.3, 0.5):
        go = f[:, 1] >= thr
        reads = np.where(go, r[:, 1], np.minimum(r[:, 1], N)).mean()
        cut = int((tum & ~go & (r[:, 3] > N)).sum())
        print(f'  read {N}, go on only if max t_top16 >= {thr}: {reads:.0f} reads/slide, cuts off needed units on {cut} tumour slides; normal slides that go on {int((go & ~tum).sum())} of {int((~tum).sum())}')
