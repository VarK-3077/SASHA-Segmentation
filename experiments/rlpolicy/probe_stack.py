# Clump probing stacked on PPO agents' own candidates (cross-fitted: each train slide is played by the networks
# trained without its fold), simulator. Prints one row per (agent, probes, rescue).
import pickle, sys
import numpy as np, torch
from scipy import ndimage as nd
from experiments.rlpolicy import rl3, sim as S, where as W, probe as P

sl = S.load('train', 't1')
for s in sl:
    s.add_tok3('s2', 's')
fold = W.folds()
rule = [S.rule(s) for s in sl]
tum = [i for i, s in enumerate(sl) if s.tumour]
fmt = lambda m: f"{m['sec']:.3f} s  Dice {m['dice']:.4f}  FROC {m['froc']:.4f}  L3 {m['l3']:.0f}  L2 {m['l2']:.0f}  FP {m['fp']:.2f}"
lost = lambda recs: sum(max(0, rule[i]['found'] - recs[i]['found']) for i in tum)
print('rule            ', fmt(S.summarise(rule)), flush=True)

def clump(s, screen, cand, paint, key, m, rescue):
    score = getattr(s, key)
    grp, k = nd.label(cand, structure=P.S8)
    zoom = np.zeros_like(cand)
    if k:
        flat, sc, pos = grp.ravel(), score.ravel(), s.pos.ravel()
        order = np.lexsort((-sc, flat)); order = order[flat[order] > 0]
        g = flat[order]; first = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
        rank = np.arange(len(order)) - np.repeat(first, np.diff(np.r_[first, len(order)]))
        probe = order[rank < m]
        hit = np.zeros(k + 1, bool); np.logical_or.at(hit, flat[probe], pos[probe])
        take = np.zeros(cand.size, bool); take[probe] = True
        rest = order[rank >= m]; take[rest] = hit[flat[rest]] | (sc[rest] >= rescue)
        zoom = take.reshape(cand.shape)
    return S.finish(s, screen, zoom, paint, rounds=1 if k else 0)

for spec in sys.argv[1:]:
    name, rk = spec.split('@')
    run = pickle.load(open(f'{W.ROOT}/search/{name}/res.pkl', 'rb'))['runs'][rk]
    rl3.SEG3 = run.get('seg3')
    nets = [rl3.nets_of(sd) for sd in run['fold_nets']]
    acts = []
    for s in sl:
        x6, x3 = rl3.states(s, run['state'], run['tok6'])
        n = nets[fold[s.name]]
        with torch.no_grad():
            scr = (n.pi6(x6)[:, 0] >= 0).numpy(); a3 = n.pi3(x3).argmax(1).numpy()
        idx = np.flatnonzero(s.tissue.ravel())
        g = lambda v: (lambda z: (z.__setitem__(idx, v), z.reshape(s.tissue.shape))[1])(np.zeros(s.tissue.size, bool))
        acts.append((g(scr), g(scr & (a3 == rl3.ZOOM)), g(scr & (a3 == rl3.PAINT))))
    base = [S.finish(s, *a) for s, a in zip(sl, acts)]
    print(f'--- {spec}: as trained   {fmt(S.summarise(base))}  lost {lost(base)}', flush=True)
    for key in ('t_top16', 's_top16'):
        for m in (1, 2):
            for rescue in (2.0, 0.5, 0.3):
                recs = [clump(s, *a, key, m, rescue) for s, a in zip(sl, acts)]
                print(f'  probe {m} by {key} rescue {"none" if rescue > 1 else rescue}: {fmt(S.summarise(recs))}  lost {lost(recs)}  Dice below rule on {sum(recs[i]["dice"] < rule[i]["dice"] - 1e-9 for i in tum)}', flush=True)
