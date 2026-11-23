# Early slide stop, learned from the reward. Stage 1 reads the N best units of the L6 screen at L3 (by pg), takes
# the base policy's candidates among them and probes the B1 best clumps at L2. One decision per slide follows:
# stop (the other screened units are never read) or go on (read the rest, then probe in batches of 5 with the
# stop-after-an-empty-batch rule). The state is what stage 1 returned. Both outcomes of every train slide are
# known in the simulator, so the policy is trained by the exact policy gradient of the reward (rl3's reward with a
# price of time lam): maximise sum_slides p(go on | state) x (J_go - J_stop); no sampling is needed. Cross-fitted.
# Usage: rl5.py BASE [N] [B1]      (BASE = consensus | rule | SEARCH@RUN)
import sys
import numpy as np, torch, torch.nn as nn
from scipy import ndimage as nd
from experiments.rlpolicy import rl3, rl4, sim as S, where as W

base = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100
B1 = int(sys.argv[3]) if len(sys.argv) > 3 else 5
sl = S.load('train', 't1')
for s in sl:
    s.add_tok3('s2', 's')
fold = W.folds()
rule = [S.rule(s) for s in sl]
tum = [i for i, s in enumerate(sl) if s.tumour]
nor = [i for i, s in enumerate(sl) if not s.tumour]
fmt = lambda m: f"{m['sec']:.3f} s  Dice {m['dice']:.4f}  FROC {m['froc']:.4f}  L3 {m['l3']:.0f}  L2 {m['l2']:.0f}  FP {m['fp']:.2f}"
lost = lambda recs: sum(max(0, rule[i]['found'] - recs[i]['found']) for i in tum)
lg = rl3.lg

def clumps(s, cand):
    grp, k = nd.label(cand, structure=rl4.S8)
    if not k:
        return 0, None, None, None, np.zeros(0)
    idx = np.arange(1, k + 1)
    best = nd.maximum(s.t_top16, grp, idx)
    pos = nd.maximum_position(s.t_top16, grp, idx)
    probe = np.array([p[0] * cand.shape[1] + p[1] for p in pos])
    order = np.argsort(-best, kind='stable')
    return k, grp.ravel(), probe, order, best[order]

def probe_stop(s, cand, zoom):
    k, flat, probe, order, _ = clumps(s, cand)
    rounds, j, pos = 0, 0, s.pos.ravel()
    while k and j < k:
        b = order[j:j + 5]; j += 5; rounds += 1
        zoom[probe[b]] = True
        hit = pos[probe[b]]
        for c in b[hit]:
            zoom[flat == c + 1] = True
        if not hit.any():
            break
    return rounds

def both(s, screen, cand, paint):
    scr = np.flatnonzero(screen.ravel())
    pgs = s.pg.ravel()[scr]
    o = np.argsort(-pgs, kind='stable')
    top = scr[o[:N]]
    s1 = np.zeros(screen.size, bool); s1[top] = True; s1 = s1.reshape(screen.shape)
    zoom = np.zeros(screen.size, bool)
    c1, p1 = cand & s1, paint & s1
    k, flat, probe, order, best = clumps(s, c1)
    hits, strength = 0, 0.0
    if k:
        b = order[:B1]
        zoom[probe[b]] = True
        h = s.pos.ravel()[probe[b]]
        for c in b[h]:
            zoom[flat == c + 1] = True
        hits = int(h.sum()); strength = float(s.unit_max.ravel()[probe[b]][h].max()) if hits else 0.0
    stop = S.finish(s, s1, zoom.reshape(screen.shape), p1, rounds=0)
    z2 = np.zeros(screen.size, bool)
    r = probe_stop(s, cand, z2)
    go = S.finish(s, screen, (zoom | z2).reshape(screen.shape), paint, rounds=r)
    t16, p3, s16 = s.t_top16.ravel()[top], s.p3.ravel()[top], s.s_top16.ravel()[top]
    top8 = np.zeros(8, np.float32); top8[:min(8, k)] = lg(best[:8]) if k else 0
    x = np.r_[float(hits > 0), hits / B1, strength, float(p1.any()), top8, np.log(k + 1) / 3, lg(t16.max()) if len(top) else -2, lg(np.sort(t16)[-5]) if len(top) >= 5 else -2,
              lg(p3.max()) if len(top) else -2, lg(s16.max()) if len(top) else -2, (t16 >= 0.3).mean() if len(top) else 0, (p3 >= 0.5).mean() if len(top) else 0,
              pgs[o[:N]].mean() if len(top) else 0, lg(pgs[o[N]]) if len(scr) > N else -2, np.log(len(scr) + 1) / 7, np.log(s.n + 1) / 8, min(1.0, N / max(len(scr), 1))].astype(np.float32)
    return stop, go, x

acts = rl4.base_actions(sl, base)
recs = [both(s, *a) for s, a in zip(sl, acts)]
X = torch.from_numpy(np.stack([r[2] for r in recs]))
mu, sd = X.mean(0), X.std(0) + 1e-6
X = (X - mu) / sd
print('rule            ', fmt(S.summarise(rule)), flush=True)
print(f'base {base}, stage 1 = best {N} reads + {B1} probes', flush=True)
print('  never stop   ', fmt(S.summarise([r[1] for r in recs])), ' lost', lost([r[1] for r in recs]), flush=True)
naive = [r[0] if (r[2][0] == 0 and r[2][3] == 0) else r[1] for r in recs]
print('  stop if no hit', fmt(S.summarise(naive)), ' lost', lost(naive), f' stopped {sum(naive[i] is recs[i][0] for i in nor)}/{len(nor)} normal, {sum(naive[i] is recs[i][0] for i in tum)}/{len(tum)} tumour', flush=True)

def J(o, lam, w_les):
    fp, tp, _ = o['froc']
    return o.get('dice', 0.0) + w_les * o['found'] - rl3.W_FP * float(np.sum(fp)) - lam * o['sec']

f = np.array([fold[s.name] for s in sl])
for w_les in (8.0, 32.0):
    for lam in (0.03, 0.1, 0.3, 1.0):
        adv = torch.tensor([J(r[1], lam, w_les) - J(r[0], lam, w_les) for r in recs], dtype=torch.float32)
        go = np.zeros(len(sl), bool)
        for k in range(W.N_FOLDS):
            tr = torch.from_numpy(f != k)
            torch.manual_seed(k)
            net = nn.Sequential(nn.Linear(X.shape[1], 32), nn.Tanh(), nn.Linear(32, 1))
            opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=0.05)
            for _ in range(600):
                loss = -(torch.sigmoid(net(X[tr])[:, 0]) * adv[tr]).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                go[f == k] = (net(X[torch.from_numpy(f == k)])[:, 0] >= 0).numpy()
        cv = [r[1] if g else r[0] for r, g in zip(recs, go)]
        print(f'  learned stop, w_les {w_les:.0f} lam {lam:>4}: cross-fitted {fmt(S.summarise(cv))}  lost {lost(cv)}  Dice below rule on {sum(cv[i]["dice"] < rule[i]["dice"] - 1e-9 for i in tum)}'
              f'  | stopped {int((~go[nor]).sum())}/{len(nor)} normal, {int((~go[tum]).sum())}/{len(tum)} tumour', flush=True)
