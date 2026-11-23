# Two-stage reading with an early slide stop, simulator, train. Stage 1 reads only the N best units of the L6
# screen (by pg) at L3, takes the base policy's candidates among them and probes the best B1 clumps at L2; if no
# probe returns anything and nothing is painted, the slide is stopped there (the other screened units are never
# read at L3). Otherwise stage 2 reads the rest and runs probe-in-batches-of-5-with-stop on all candidates.
import sys, pickle
import numpy as np, torch
from scipy import ndimage as nd
from experiments.rlpolicy import rl3, rl4, sim as S, where as W
from experiments.rlprobe.cost import seconds

sl = S.load('train', 't1')
for s in sl:
    s.add_tok3('s2', 's')
rule = [S.rule(s) for s in sl]
tum = [i for i, s in enumerate(sl) if s.tumour]
nor = [i for i, s in enumerate(sl) if not s.tumour]
fmt = lambda m: f"{m['sec']:.3f} s  Dice {m['dice']:.4f}  FROC {m['froc']:.4f}  L3 {m['l3']:.0f}  L2 {m['l2']:.0f}  FP {m['fp']:.2f}"
lost = lambda recs: sum(max(0, rule[i]['found'] - recs[i]['found']) for i in tum)
print('rule            ', fmt(S.summarise(rule)), flush=True)

def clumps(s, cand):
    grp, k = nd.label(cand, structure=rl4.S8)
    if not k:
        return k, None, None, None
    idx = np.arange(1, k + 1)
    best = nd.maximum(s.t_top16, grp, idx)
    pos = nd.maximum_position(s.t_top16, grp, idx)
    probe = np.array([p[0] * cand.shape[1] + p[1] for p in pos])
    order = np.argsort(-best, kind='stable')
    return k, grp.ravel(), probe, order

def probe_stop(s, cand, B, zoom):
    k, flat, probe, order = clumps(s, cand)
    rounds, j, pos = 0, 0, s.pos.ravel()
    while k and j < k:
        b = order[j:j + B]; j += B; rounds += 1
        zoom[probe[b]] = True
        hit = pos[probe[b]]
        for c in b[hit]:
            zoom[flat == c + 1] = True
        if not hit.any():
            break
    return rounds

def play(s, screen, cand, paint, N, B1):
    scr = np.flatnonzero(screen.ravel())
    top = scr[np.argsort(-s.pg.ravel()[scr], kind='stable')[:N]]
    s1 = np.zeros(screen.size, bool); s1[top] = True; s1 = s1.reshape(screen.shape)
    zoom = np.zeros(screen.size, bool)
    c1, p1 = cand & s1, paint & s1
    k, flat, probe, order = clumps(s, c1)
    hit = False
    if k:
        b = order[:B1]
        zoom[probe[b]] = True
        h = s.pos.ravel()[probe[b]]
        for c in b[h]:
            zoom[flat == c + 1] = True
        hit = bool(h.any())
    if not (hit or p1.any()):
        return S.finish(s, s1, zoom.reshape(screen.shape), p1, rounds=0)
    z2 = np.zeros(screen.size, bool)
    r = probe_stop(s, cand, 5, z2)
    return S.finish(s, screen, (zoom | z2).reshape(screen.shape), paint, rounds=r)

for base in ['consensus'] + sys.argv[1:]:
    acts = rl4.base_actions(sl, base)
    ref = []
    for s, (screen, cand, paint) in zip(sl, acts):
        z = np.zeros(screen.size, bool); r = probe_stop(s, cand, 5, z)
        ref.append(S.finish(s, screen, z.reshape(screen.shape), paint, rounds=max(r - 1, 0)))
    print(f'--- {base}: probe in batches of 5 with stop, every screened unit read   {fmt(S.summarise(ref))}  lost {lost(ref)}', flush=True)
    for N in (50, 100, 150, 200):
        for B1 in (3, 5, 10):
            recs = [play(s, *a, N, B1) for s, a in zip(sl, acts)]
            stopped_t = sum(recs[i]['n_l3'] <= N for i in tum); stopped_n = sum(recs[i]['n_l3'] <= N for i in nor)
            print(f'  stage 1 = best {N} reads, {B1} probes: {fmt(S.summarise(recs))}  lost {lost(recs)}  Dice below rule on {sum(recs[i]["dice"] < rule[i]["dice"] - 1e-9 for i in tum)}'
                  f'  | stopped early: {stopped_n}/{len(nor)} normal, {stopped_t}/{len(tum)} tumour slides', flush=True)
