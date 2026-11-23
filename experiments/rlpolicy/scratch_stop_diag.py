# Is there room for a per-slide stopping decision? Clumps of the consensus policy's (or an agent's) zoom candidates,
# probed in descending order of their best t_top16. For each slide: how many clumps, where the clumps that matter
# (the probe returns something / the oracle needs a unit of the clump) rank, and what an oracle that stops right
# after the last clump that matters would spend, against probing every clump.
import numpy as np
from scipy import ndimage as nd
from experiments.rlpolicy import sim as S, where as W, probe as P

sl = S.load('train', 't1')
rows = []
for s in sl:
    screen, cand, paint = P.candidates(s, P.VETO)
    grp, k = nd.label(cand, structure=P.S8)
    need = np.load(f'{W.ROOT}/oracle/cascade_reward/{s.name}.npy').reshape(s.tissue.shape) != 0
    if not k:
        rows.append((s.tumour, 0, 0, 0, 0, 0)); continue
    idx = np.arange(1, k + 1)
    best = nd.maximum(s.t_top16, grp, idx)
    pos = nd.maximum(s.pos, grp, idx).astype(bool)
    nd_need = nd.maximum(need & cand, grp, idx).astype(bool)
    order = np.argsort(-best)
    rank_pos = np.flatnonzero(pos[order]) + 1
    rank_need = np.flatnonzero(nd_need[order]) + 1
    rows.append((s.tumour, k, int(pos.sum()), int(nd_need.sum()), int(rank_pos.max()) if len(rank_pos) else 0, int(rank_need.max()) if len(rank_need) else 0))
r = np.array(rows)
for name, m in (('tumour slides', r[:, 0] == 1), ('normal slides', r[:, 0] == 0)):
    x = r[m]
    print(f'{name} ({len(x)}): clumps/slide {x[:, 1].mean():.1f} (median {np.median(x[:, 1]):.0f}); clumps whose probe returns something {x[:, 2].mean():.1f}; clumps the oracle needs {x[:, 3].mean():.1f}')
    print(f'    probes if we stop right after the last clump that returns something: {x[:, 4].mean():.1f}/slide; after the last needed clump: {x[:, 5].mean():.1f}/slide; probing all: {x[:, 1].mean():.1f}')
    if name.startswith('tumour'):
        frac = x[:, 5] / np.maximum(x[:, 1], 1)
        print('    last needed clump, as a fraction of the way down the list: quartiles', np.round(np.quantile(frac, [0.25, 0.5, 0.75, 0.9, 1.0]), 2), ' | slides where it is in the top 3:', int((x[:, 5] <= 3).sum()), 'top 5:', int((x[:, 5] <= 5).sum()), 'top 10:', int((x[:, 5] <= 10).sum()))
n = len(r)
print(f'all slides: probing every clump {r[:, 1].sum() / n:.1f}/slide; oracle stop after last needed {r[:, 5].sum() / n:.1f}/slide; fixed budget of K probes per slide would lose needed clumps on:',
      {K: int(((r[:, 0] == 1) & (r[:, 5] > K)).sum()) for K in (3, 5, 8, 12, 20)}, 'tumour slides and spend', {K: round(float(np.minimum(r[:, 1], K).mean()), 1) for K in (3, 5, 8, 12, 20)})
