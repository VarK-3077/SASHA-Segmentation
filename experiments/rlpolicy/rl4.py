# Sequential probing agent: the first RL policy here whose state holds what the segmenter returned so far.
# The zoom candidates of a base policy (the supervised rule, the consensus policy, or a PPO agent of rl3.py played
# with the networks that did not see the slide) are grouped into connected clumps and visited in descending order
# of their best token-head score, in batches of B clumps per round. In a round the best unit of each clump of the
# batch is segmented (the probe). Two small PPO policies then act on what came back:
#   clump policy: a clump whose probe returned nothing -> drop its other units, or segment them anyway
#                 (a clump whose probe returned something is always segmented in full);
#   stop policy:  after the round -> go on with the next batch, or stop the slide.
# Painted core units and border tracing are the cascade's. Reward, credited per decision: the per-unit reward of
# rl3 (Dice share + w_les per lesion - w_fp x score per false positive) of the units a decision caused to be
# segmented, minus lam x their seconds, minus lam x ROUND_S per extra round. Cross-fitted like rl3; --init rule
# starts both policies at the hand-written reference (drop on an empty probe; stop after an empty round) by
# behaviour cloning, --init none from random weights.
# Usage: python -m experiments.rlpolicy.rl4 --out NAME --base consensus|rule|SEARCH@RUN --lams 3 10 [--batch 5]

import argparse
import os
import pickle
from multiprocessing import get_context

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage as nd

from experiments.rlpolicy import rl3
from experiments.rlpolicy import sim as S
from experiments.rlpolicy import where as W
from experiments.rlprobe.cost import L2

S8 = np.ones((3, 3), bool)
DC, DS = 9, 9
DATA = None


class Nets(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.pc, self.vc, self.ps, self.vs = rl3.mlp(DC, 1, h), rl3.mlp(DC, 1, h), rl3.mlp(DS, 1, h), rl3.mlp(DS, 1, h)


def prepare(s, screen, cand, paint):
    """Static description of a slide's clumps, best first."""
    grp, k = nd.label(cand, structure=S8)
    d = dict(s=s, screen=screen, cand=cand, paint=paint, k=k)
    if not k:
        return d
    idx = np.arange(1, k + 1)
    best = nd.maximum(s.t_top16, grp, idx)
    order = np.argsort(-best, kind='stable')
    pos = nd.maximum_position(s.t_top16, grp, idx)
    probe = np.array([p[0] * cand.shape[1] + p[1] for p in pos])[order]
    flat = grp.ravel()
    members = [np.flatnonzero(flat == c + 1) for c in order]
    mx = lambda a: nd.maximum(a, grp, idx)[order]
    size = np.array([len(m) for m in members])
    second = np.array([np.sort(s.t_top16.ravel()[m])[-2] if len(m) > 1 else 0.0 for m in members])
    lg = rl3.lg
    d.update(probe=probe, members=members, size=size, hit=s.pos.ravel()[probe], strength=s.unit_max.ravel()[probe],
             fc=np.stack([lg(best[order]), lg(mx(s.p3)), lg(mx(s.s_top16)), lg(mx(s.pg)), np.log(size) / 3, lg(second), np.arange(k) / max(k - 1, 1)], 1).astype(np.float32),
             best=lg(best[order]).astype(np.float32))
    return d


def play(nets, d, B, rng=None, lam=0.0, W=(8.0, rl3.W_FP)):
    """One episode. rng None = greedy. Returns the simulator record and, when sampling, the PPO transitions."""
    s, k = d['s'], d['k']
    zoom = np.zeros(d['cand'].size, bool)
    tc, ts = [], []          # (x, action, logp, units caused) per clump decision; (x, action, logp, round) per stop decision
    rounds, hits, strength, j = 0, 0, 0.0, 0
    unit_round = {}
    painted = float(d['paint'].any())
    while j < k:
        b = np.arange(j, min(j + B, k))
        j, rounds = b[-1] + 1, rounds + 1
        zoom[d['probe'][b]] = True
        for u in d['probe'][b]:
            unit_round[int(u)] = rounds
        h = d['hit'][b]
        for c in b[h]:
            zoom[d['members'][c]] = True
            for u in d['members'][c]:
                unit_round.setdefault(int(u), rounds)
        ask = b[~h & (d['size'][b] > 1)]
        if len(ask):
            x = torch.from_numpy(np.concatenate([d['fc'][ask], np.full((len(ask), 1), min(hits, 5) / 5, np.float32), np.full((len(ask), 1), strength, np.float32)], 1))
            with torch.no_grad():
                p = torch.sigmoid(nets.pc(x)[:, 0]).numpy()
            a = (rng.random(len(ask)) < p) if rng is not None else (p >= 0.5)
            for i, c in enumerate(ask):
                rest = d['members'][c][d['members'][c] != d['probe'][c]]
                if a[i]:
                    zoom[rest] = True
                    for u in rest:
                        unit_round.setdefault(int(u), rounds)
                tc.append((x[i], float(a[i]), float(np.log(max(p[i] if a[i] else 1 - p[i], 1e-8))), rest if a[i] else rest[:0]))
        hits += int(h.sum())
        if h.any():
            strength = max(strength, float(d['strength'][b][h].max()))
        if j >= k:
            break
        y = torch.tensor([[rounds / 5, float(h.sum()) / B, min(hits, 5) / 5, strength, d['best'][j], d['best'][b[-1]], np.log(k - j + 1) / 3, np.log(k + 1) / 4, painted]], dtype=torch.float32)
        with torch.no_grad():
            p = float(torch.sigmoid(nets.ps(y)[0, 0]))
        go = (rng.random() < p) if rng is not None else (p >= 0.5)
        ts.append((y[0], float(go), float(np.log(max(p if go else 1 - p, 1e-8))), rounds))
        if not go:
            break
    out = S.finish(s, d['screen'], zoom.reshape(d['cand'].shape), d['paint'], rounds=max(rounds - 1, 0))      # the first round is the cascade's own
    if rng is None:
        return out, None
    # credit: each segmented unit's reward minus its seconds goes to the round that caused it; traced units go to
    # the earliest round among the clump units of their connected group
    q = S.unit_rewards(s, out, W[0], W[1]).ravel()
    net = np.where(out['zoom'].ravel(), q - lam * L2, 0.0)
    grp, n = nd.label(out['zoom'], structure=S8)
    flat = grp.ravel()
    first = np.full(n + 1, 10 ** 6)
    for u, r in unit_round.items():
        if flat[u]:
            first[flat[u]] = min(first[flat[u]], r)
    rnd = np.where(flat > 0, first[flat], 0)
    for u, r in unit_round.items():
        rnd[u] = r
    per_round = np.array([net[rnd == r].sum() - (lam * S.ROUND_S if r > 1 else 0.0) for r in range(1, rounds + 1)])
    rc = [float(net[rest].sum()) for *_, rest in tc]
    rs = [float(per_round[r:].sum()) if go else 0.0 for _, go, _, r in ts]
    return out, dict(xc=[t[0] for t in tc], ac=[t[1] for t in tc], lc=[t[2] for t in tc], rc=rc, xs=[t[0] for t in ts], as_=[t[1] for t in ts], ls=[t[2] for t in ts], rs=rs)


def reference(data, B):
    """The hand-written reference's decisions as cloning targets: never segment the rest of a clump whose probe
    returned nothing; go on iff the round returned something."""
    xc, xs, ts = [], [], []

    class Ref:
        pc = staticmethod(lambda x: torch.full((len(x), 1), -20.0))
        ps = staticmethod(lambda y: (y[:, 1:2] > 0).float() * 40 - 20)
    for d in data:
        _, t = play(Ref, d, B, rng=np.random.default_rng(0))
        xc += t['xc']; xs += t['xs']; ts += t['as_']
    return xc, xs, ts


def clone(nets, data, B, steps=1500, seed=0):
    xc, xs, ts = reference(data, B)
    opt = torch.optim.Adam(list(nets.pc.parameters()) + list(nets.ps.parameters()), lr=3e-3)
    xs_t, ts_t = torch.stack(xs), torch.tensor(ts)
    xc_t = torch.stack(xc) if xc else torch.zeros(1, DC)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(len(xs_t), (2048,), generator=g)
        j = torch.randint(len(xc_t), (2048,), generator=g)
        loss = nn.functional.binary_cross_entropy_with_logits(nets.ps(xs_t[i])[:, 0], ts_t[i]) + nn.functional.binary_cross_entropy_with_logits(nets.pc(xc_t[j])[:, 0], torch.zeros(len(j)))
        opt.zero_grad(); loss.backward(); opt.step()


def ppo(data, lam, a, seed, log=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    nets = Nets(a.hidden)
    if a.init == 'rule':
        clone(nets, data, a.batch, seed=seed)
    opt = torch.optim.Adam([dict(params=list(nets.pc.parameters()) + list(nets.ps.parameters()), lr=a.lr), dict(params=list(nets.vc.parameters()) + list(nets.vs.parameters()), lr=a.critic_lr)])
    tum = [i for i, d in enumerate(data) if d['s'].tumour]
    nor = [i for i, d in enumerate(data) if not d['s'].tumour]
    norm = lambda x: (x - x.mean()) / (x.std() + 1e-8)
    summ = lambda: S.summarise([play(nets, d, a.batch)[0] for d in data])
    if log:
        m = summ()
        log(f"    start ({a.init}): greedy on its train slides {m['sec']:.3f} s  Dice {m['dice']:.4f}  FROC {m['froc']:.4f}  L2 {m['l2']:.0f}")
    for up in range(1, a.updates + 1):
        pick = np.r_[rng.choice(tum, a.episodes // 2), rng.choice(nor, a.episodes - a.episodes // 2)]
        T = [play(nets, data[i], a.batch, rng, lam, (a.w_les, a.w_fp))[1] for i in pick]
        cat = lambda key: [v for t in T for v in t[key]]
        parts = []
        for px, pa, pl, pr, pi, vf in (('xc', 'ac', 'lc', 'rc', nets.pc, nets.vc), ('xs', 'as_', 'ls', 'rs', nets.ps, nets.vs)):
            if len(cat(px)) < 8:
                continue
            x, act, old, ret = torch.stack(cat(px)), torch.tensor(cat(pa)), torch.tensor(cat(pl)), torch.tensor(cat(pr), dtype=torch.float32)
            with torch.no_grad():
                adv = norm(ret - vf(x)[:, 0] * a.vscale)
            parts.append((x, act, old, ret, adv, pi, vf))
        for _ in range(a.epochs):
            loss = 0.0
            for x, act, old, ret, adv, pi, vf in parts:
                logit = pi(x)[:, 0]
                lp = -nn.functional.binary_cross_entropy_with_logits(logit, act, reduction='none')
                ratio = (lp - old).exp()
                p = torch.sigmoid(logit).clamp(1e-6, 1 - 1e-6)
                ent = -(p * p.log() + (1 - p) * (1 - p).log()).mean()
                loss = loss - torch.min(ratio * adv, ratio.clamp(1 - a.clip, 1 + a.clip) * adv).mean() + ((vf(x)[:, 0] - ret / a.vscale) ** 2).mean() - a.ent * ent
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(nets.parameters(), 1.0)
            opt.step()
        if log and up % a.log_every == 0:
            m = summ()
            log(f"    update {up}: greedy on its train slides {m['sec']:.3f} s  Dice {m['dice']:.4f}  FROC {m['froc']:.4f}  L2 {m['l2']:.0f}")
    return nets


def base_actions(slides, base):
    """(screen, zoom candidates, paint) per slide from the base policy; an rl3 agent plays each train slide with the
    networks trained without its fold."""
    fold = W.folds()
    if base in ('rule', 'consensus'):
        veto = 0.0
        if base == 'consensus':       # fold vetoes of the consensus policy (vsel), as rl3 --init consensus uses them
            t = pickle.load(open(f'{W.ROOT}/search/vsel_t1/res.pkl', 'rb'))['runs']['t_top16:0.5']
            vf, va = [0.5 * v for v in t['v_fold']], float(t['theta'].veto)
        out = []
        for s in slides:
            v = (vf[fold[s.name]] if s.name in fold else va) if base == 'consensus' else veto
            screen = s.tissue & (s.pg >= 0.6)
            want = screen & (s.p3 >= 0.5) & (s.t_top16 >= v)
            paint = want & (s.p_core >= 0.99)
            out.append((screen, want & ~paint, paint))
        return out
    name, rk = base.split('@')
    run = pickle.load(open(f'{W.ROOT}/search/{name}/res.pkl', 'rb'))['runs'][rk]
    rl3.SEG3 = run.get('seg3')
    fn, an = [rl3.nets_of(sd) for sd in run['fold_nets']], rl3.nets_of(run['nets'])
    out = []
    for s in slides:
        x6, x3 = rl3.states(s, run['state'], run['tok6'])
        n = fn[fold[s.name]] if s.name in fold else an
        with torch.no_grad():
            scr = (n.pi6(x6)[:, 0] >= 0).numpy()
            a3 = n.pi3(x3).argmax(1).numpy()
        idx = np.flatnonzero(s.tissue.ravel())
        g = lambda v: (lambda z: (z.__setitem__(idx, v), z.reshape(s.tissue.shape))[1])(np.zeros(s.tissue.size, bool))
        out.append((g(scr), g(scr & (a3 == rl3.ZOOM)), g(scr & (a3 == rl3.PAINT))))
    return out


def load(split, base, seg3='s2'):
    slides = S.load(split, 't1')
    for s in slides:
        s.add_tok3(seg3, 's')
    return [prepare(s, *act) for s, act in zip(slides, base_actions(slides, base))]


def job(task):
    lam, k, a = task
    fold = W.folds()
    torch.set_num_threads(1)
    lines = []
    say = lambda t: (lines.append(t), print(f'  [lam {lam}]' + t, flush=True))
    nets = ppo([d for d in DATA if fold[d['s'].name] != k], lam, a, seed=a.seed + 10 * (k + 1), log=say if k < 0 else None)
    held = [(d['s'].name, play(nets, d, a.batch)[0]) for d in DATA if fold[d['s'].name] == k]
    return lam, k, {n: v.detach().clone() for n, v in nets.state_dict().items()}, held


def main():
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--base', default='consensus')
    ap.add_argument('--seg3', default='s2')
    ap.add_argument('--lams', type=float, nargs='+', required=True)
    ap.add_argument('--batch', type=int, default=5)
    ap.add_argument('--init', choices=('rule', 'none'), default='rule')
    ap.add_argument('--updates', type=int, default=400)
    ap.add_argument('--episodes', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--critic_lr', type=float, default=1e-3)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--ent', type=float, default=0.003)
    ap.add_argument('--vscale', type=float, default=0.05)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--w_les', type=float, default=8.0)
    ap.add_argument('--w_fp', type=float, default=rl3.W_FP)
    ap.add_argument('--log_every', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    DATA = load('train', a.base, a.seg3)
    slides = [d['s'] for d in DATA]
    rule = [S.rule(s) for s in slides]
    tum = [i for i, s in enumerate(slides) if s.tumour]
    fmt = lambda m: f"{m['sec']:.3f} s  Dice {m['dice']:.4f}  FROC {m['froc']:.4f}  L3 {m['l3']:.0f}  L2 {m['l2']:.0f}  paint {m['paint']:.1f}  FP {m['fp']:.2f}"
    lost = lambda recs: sum(max(0, rule[i]['found'] - recs[i]['found']) for i in tum)
    print('rule            ', fmt(S.summarise(rule)), flush=True)
    base = [S.finish(d['s'], d['screen'], d['cand'], d['paint']) for d in DATA]
    print(f'base ({a.base}), all candidates segmented: {fmt(S.summarise(base))}  lost {lost(base)}', flush=True)
    out_dir = f'{W.ROOT}/search/{a.out}'
    os.makedirs(out_dir, exist_ok=True)
    res = dict(names=[s.name for s in slides], runs={}, args=vars(a))
    tasks = [(lam, k, a) for lam in a.lams for k in list(range(W.N_FOLDS)) + [-1]]
    got = {}
    with get_context('fork').Pool(min(a.workers, len(tasks))) as pool:
        for lam, k, sd, held in pool.imap_unordered(job, tasks):
            got[(lam, k)] = (sd, dict(held))
            if all((lam, j) in got for j in list(range(W.N_FOLDS)) + [-1]):
                by = {}
                for j in range(W.N_FOLDS):
                    by.update(got[(lam, j)][1])
                cv = [by[s.name] for s in slides]
                nor = np.mean([cv[i]['n_l2'] for i in range(len(slides)) if not slides[i].tumour])
                print(f'lam {lam}: cross-fitted {fmt(S.summarise(cv))}  lost {lost(cv)}  Dice below rule on {sum(cv[i]["dice"] < rule[i]["dice"] - 1e-9 for i in tum)}'
                      f'  zooms normal {nor:.1f} tumour {np.mean([cv[i]["n_l2"] for i in tum]):.1f}', flush=True)
                res['runs'][f'lam{lam}'] = dict(cv=cv, nets=got[(lam, -1)][0], fold_nets=[got[(lam, j)][0] for j in range(W.N_FOLDS)], base=a.base, batch=a.batch, seg3=a.seg3, hidden=a.hidden)
                pickle.dump(res, open(f'{out_dir}/res.pkl', 'wb'))
    print('done', flush=True)


if __name__ == '__main__':
    main()
