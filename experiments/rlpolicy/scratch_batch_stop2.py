# batch_stop.py with a rescue: in a clump whose probe returned nothing, units whose own token-head score is at
# least `rescue` are segmented anyway (same round as the next batch). Batches of 5, stop after an empty batch.
import sys
import numpy as np
from scipy import ndimage as nd
from experiments.rlpolicy import rl4, sim as S, where as W

sl = S.load('train', 't1')
for s in sl:
    s.add_tok3('s2', 's')
rule = [S.rule(s) for s in sl]
tum = [i for i, s in enumerate(sl) if s.tumour]
nor = [i for i, s in enumerate(sl) if not s.tumour]
fmt = lambda m: f"{m['sec']:.3f} s  Dice {m['dice']:.4f}  FROC {m['froc']:.4f}  L2 {m['l2']:.0f}  FP {m['fp']:.2f}"
lost = lambda recs: sum(max(0, rule[i]['found'] - recs[i]['found']) for i in tum)
print('rule            ', fmt(S.summarise(rule)), flush=True)

def play(s, screen, cand, paint, B, rescue, key='t_top16'):
    grp, k = nd.label(cand, structure=rl4.S8)
    zoom = np.zeros(cand.size, bool)
    rounds = 0
    if k:
        flat, sc, pos = grp.ravel(), getattr(s, key).ravel(), s.pos.ravel()
        idx = np.arange(1, k + 1)
        best = nd.maximum(s.t_top16, grp, idx)
        probe = np.array([p[0] * cand.shape[1] + p[1] for p in nd.maximum_position(s.t_top16, grp, idx)])
        order = np.argsort(-best, kind='stable')
        j = 0
        while j < k:
            b = order[j:j + B]; j += B; rounds += 1
            zoom[probe[b]] = True
            hit = pos[probe[b]]
            for c, h in zip(b, hit):
                m = flat == c + 1
                zoom[m & (h | (sc >= rescue))] = True
            if not hit.any():
                break
    return S.finish(s, screen, zoom.reshape(cand.shape), paint, rounds=max(rounds - 1, 0))

for base in ['consensus'] + sys.argv[1:]:
    acts = rl4.base_actions(sl, base)
    b0 = [S.finish(s, *a) for s, a in zip(sl, acts)]
    print(f'--- {base}: all candidates segmented   {fmt(S.summarise(b0))}  lost {lost(b0)}', flush=True)
    for B in (5, 8):
        for key, rescue in (('t_top16', 2.0), ('t_top16', 0.5), ('t_top16', 0.3), ('t_top16', 0.15), ('p3', 0.95), ('s_top16', 0.3)):
            recs = [play(s, *a, B, rescue, key) for s, a in zip(sl, acts)]
            print(f'  batches of {B}, rescue {key} >= {rescue if rescue <= 1 else "none"}: {fmt(S.summarise(recs))}  lost {lost(recs)}  Dice below rule on {sum(recs[i]["dice"] < rule[i]["dice"] - 1e-9 for i in tum)}'
                  f'  | zooms normal {np.mean([recs[i]["n_l2"] for i in nor]):.1f} tumour {np.mean([recs[i]["n_l2"] for i in tum]):.1f}', flush=True)
