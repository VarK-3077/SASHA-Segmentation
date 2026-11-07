# Two low-variance learners on the sequential env (stoprl.py), offline.
# Core critic: a small per-unit net predicts a unit's non-tumour share from its head probabilities and
# what the episode has revealed: how much tumour already-segmented neighbours show on the shared border,
# painted neighbours, and how solid this slide's likely-core units turned out when segmented. Any unit
# about to be zoomed, routed or reached by border following, is painted if the predicted non-tumour
# share is <= eps. Interior units of a lesion can therefore be painted, which needs episode state.
# Optimal stop: units are taken in a fixed order, so a slide's trajectory does not depend on when it
# stops. For every round we know the end state of stopping there (finish following, then end), for any
# lambda, without sampling. A net g(state, lam) is fitted by backward induction (fitted Q for optimal
# stopping, as Longstaff-Schwartz): g_t = W_t+1 - W_t + max(0, g_t+1), W_t = phi(J) - lam * cost of
# stopping at t. Take another batch while g > 0. State is slide level and cheap, plus clusters of the
# score map, since tumour clusters and false positives scatter.
# Usage: RLSCREEN_OOF=1 python -m experiments.rlscreen.qstop --design full --eps 0.02 --tag NAME

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage

from experiments.rlscreen import paths as P
from experiments.rlscreen import stoprl as R
from experiments.rlscreen.rollout import load_tables
from experiments.rlscreen.seqscreen import Slide, phi

N_CORE = 13
N_EXTRA = 12


class CoreNet(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(N_CORE, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.f(x)[..., 0]


def cluster_stats(s):
    out = []
    for t in (0.5, 0.9):
        m = (s.pg >= t) & s.tissue
        lab, k = ndimage.label(m)
        sizes = np.bincount(lab.ravel())[1:] if k else np.zeros(1)
        out += [sizes.max() / max(s.n, 1), np.log1p(sizes.max()) / 6, np.log1p(k) / 6, (sizes >= 4).sum() / max(k, 1)]
    return np.asarray(out, np.float32)


class Ep2(R.Ep):
    critic, eps = None, 0.0

    def core_feats(self, cand):
        s = self.s
        seg = self.zoomed & ~self.painted
        # (neighbour offset, that neighbour's strip facing this unit: N S W E = 0 1 2 3)
        sides = (((-1, 0), 1), ((1, 0), 0), ((0, -1), 3), ((0, 1), 2))
        ev, cnt, pa, mn = np.zeros_like(s.pg, np.float32), np.zeros_like(s.pg, np.float32), np.zeros_like(s.pg, np.float32), np.ones_like(s.pg, np.float32)
        for (dr, dc), k in sides:
            sg = self._shift(seg, dr, dc)
            ef = np.zeros_like(s.pg, np.float32)
            r0, r1, c0, c1 = max(dr, 0), s.hu + min(dr, 0), max(dc, 0), s.wu + min(dc, 0)
            ef[r0:r1, c0:c1] = s.edge_frac[r0 - dr:r1 - dr, c0 - dc:c1 - dc, k]
            ev += ef * sg; cnt += sg; mn = np.where(sg, np.minimum(mn, ef), mn)
            pa += self._shift(self.painted, dr, dc)
        probe = seg & (s.p_core >= 0.5)
        solid = float((s.frac[probe] >= 0.95).mean()) if probe.any() else 0.5
        known = self.screened
        idx = np.flatnonzero(cand)
        r = idx // s.wu, idx % s.wu
        pr = s.probs[r] * known[r][:, None]
        return np.column_stack([pr, known[r], cnt[r] / 4, ev[r] / np.maximum(cnt[r], 1), mn[r] * (cnt[r] > 0), pa[r] / 4, np.full(len(idx), solid),
                                np.full(len(idx), float(probe.any())), np.full(len(idx), self.zoomed.sum() / max(s.n, 1)),
                                np.full(len(idx), np.log1p(seg.sum()) / 6)]).astype(np.float32), idx

    def paint_mask(self, routed, cand, tc):
        out = super().paint_mask(routed, cand, tc)
        if self.critic is None or not cand.any():
            return out
        x, idx = self.core_feats(cand)
        with torch.no_grad():
            q = torch.sigmoid(self.critic(torch.from_numpy(x))).numpy()
        out.reshape(-1)[idx[q <= self.eps]] = True
        return out

    def features2(self):
        s = self.s
        q = self.queue()
        b = max(2, int(round(0.02 * s.n)))
        nxt = s.pg.reshape(-1)[q[:b]]
        rates = [h[5] for h in self.hist if h[0] > 0]
        zero = 0
        for r in reversed(rates):
            if r > 0:
                break
            zero += 1
        probe = (self.zoomed & ~self.painted) & s.pos
        return np.concatenate([self.features(), s.cluster, np.asarray([nxt.sum() / b if len(nxt) else 0.0, np.mean(rates[-3:]) if rates else 0.0,
                                                                      zero / 10, float(probe.sum() == 0)], np.float32)])


def make_ep(s, lam):
    return Ep2(s, lam)


def finish(ep):
    """Stop taking units: finish following and return the end state."""
    e = copy.copy(ep)
    for k in ('screened', 'zoomed', 'painted', 'front', 'follow'):
        setattr(e, k, getattr(ep, k).copy())
    e.hist = list(ep.hist)
    while not e.done and e.follow.any():
        e.step(0, 2.0)
    return e


def trajectory(s, batch):
    """Full-depth rollout. Per decision point: state features, end J and seconds of stopping there."""
    R.R_MAX = 200
    ep = make_ep(s, 1.0)
    F, J, C = [], [], []
    while True:
        e = finish(ep)
        F.append(ep.features2()); J.append(e.J); C.append(e.sec)
        n = ep.count('batch', batch)
        if n == 0 or ep.done:
            break
        ep.step(n, 0.99)
    return np.stack(F), np.array(J), np.array(C)


def fit_stop(slides, a, log):
    data = [trajectory(s, a.batch) + (R.FULL_PER_UNIT * s.n,) for s in slides]
    F = torch.from_numpy(np.concatenate([d[0] for d in data]))
    J = torch.from_numpy(np.concatenate([d[1] for d in data])).float()
    C = torch.from_numpy(np.concatenate([d[2] / d[3] for d in data])).float()
    nxt = np.concatenate([np.r_[np.arange(1, len(d[1])), -1] + o for d, o in zip(data, np.cumsum([0] + [len(d[1]) for d in data[:-1]]))])
    last = np.concatenate([np.r_[np.zeros(len(d[1]) - 1, bool), True] for d in data])
    keep = torch.from_numpy(~last)
    ni = torch.from_numpy(nxt)
    log({'stop_states': len(F), 'nonterminal': int(keep.sum())})
    Phi = lambda j: (a.S ** j - 1) / np.log(a.S)
    net = nn.Sequential(nn.Linear(F.shape[1], 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lo, hi = np.log(a.lam_min), np.log(a.lam_max)
    idx = torch.nonzero(keep)[:, 0]
    tgt = copy.deepcopy(net)
    for it in range(a.fq_iters):
        for _ in range(a.fq_steps):
            i = idx[torch.randint(len(idx), (512,))]
            ll = torch.empty(len(i)).uniform_(lo, hi)
            x, xn = F[i].clone(), F[ni[i]].clone()
            x[:, 0] = ll / 5; xn[:, 0] = ll / 5
            lam = ll.exp()
            dw = Phi(J[ni[i]]) - Phi(J[i]) - lam * (C[ni[i]] - C[i])
            with torch.no_grad():
                y = dw + tgt(xn)[:, 0].clamp(min=0)
                y = torch.where(last_t(ni[i], last), dw, y)
            loss = ((net(x)[:, 0] - y) ** 2).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        tgt = copy.deepcopy(net)
        if it % 10 == 0:
            log({'fq_iter': it, 'loss': float(loss)})
    return net


def last_t(i, last):
    return torch.from_numpy(last[i.numpy()])


def stop_actor(net, batch):
    def act(ep):
        x = ep.features2()
        n = ep.count('batch', batch)
        if not ep.stopping and n > 0:
            with torch.no_grad():
                go = float(net(torch.from_numpy(x)[None])[0, 0]) > 0
            if go:
                ep.step(n, 0.99)
                return True
            ep.stopping = True
        if ep.follow.any():
            ep.step(0, 2.0)
            return True
        return False
    return act


def collect_core(slides, rng, repeats=1):
    X, Y = [], []
    for s in slides:
        ep = make_ep(s, 1.0)
        while not ep.done:
            cand = (ep.follow | ep_top(ep, 0.02)) & ~ep.zoomed
            if cand.any():
                x, idx = ep.core_feats(cand)
                X.append(x); Y.append(1 - s.frac.reshape(-1)[idx])
            n = ep.count('batch', 0.02)
            if n == 0:
                break
            ep.step(n, 0.99)
    return np.concatenate(X), np.concatenate(Y).astype(np.float32)


def ep_top(ep, batch):
    m = np.zeros_like(ep.s.tissue)
    m.reshape(-1)[ep.queue()[:ep.count('batch', batch)]] = True
    return m


def fit_core(slides, a, log, rng):
    net = CoreNet()
    Ep2.critic, Ep2.eps = None, 0.0
    X, Y = collect_core(slides, rng)
    for rnd in range(2):                     # second round: states the painting agent itself visits
        Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for step in range(3000):
            i = torch.randint(len(Xt), (1024,))
            loss = nn.functional.binary_cross_entropy_with_logits(net(Xt[i]), Yt[i])
            opt.zero_grad(); loss.backward(); opt.step()
        log({'core_round': rnd, 'units': len(X), 'loss': float(loss)})
        Ep2.critic, Ep2.eps = net, a.eps
        x2, y2 = collect_core(slides, rng)
        X, Y = np.concatenate([X, x2]), np.concatenate([Y, y2])
    return net


def rule_rows(S):
    rows = {}
    ts = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9) if R.CFG['design'] == 'full' else (0.5, 0.6, 0.7, 0.8, 0.9)
    for t in ts:
        rows[f'fixed t={t:g}'] = R.agg([R.run_with(make_ep, s, R.fixed_rule(t, 0.99)).summary() for s in S])
    for h in (0.02, 0.05, 0.1, 0.2, 0.3):
        rows[f'hit h={h:g}'] = R.agg([R.run_with(make_ep, s, R.hit_rule(h, 0.99)).summary() for s in S])
    return rows


def hindsight(s, lam, batch):
    ep = make_ep(s, lam)
    obj, best = None, None
    while True:
        e = finish(ep)
        v = phi(e.J, R.CFG['S']) - lam * e.sec / e.seconds_full()
        if obj is None or v > obj:
            obj, best = v, e.summary()
        n = ep.count('batch', batch)
        if n == 0 or ep.done:
            return best
        ep.step(n, 0.99)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--design', choices=('full', 'coarse'), required=True)
    ap.add_argument('--g_level', type=int, default=6)
    ap.add_argument('--eps', type=float, default=0.02, help='paint if the predicted non-tumour share is <= eps; 0 = critic off, rule at 0.99')
    ap.add_argument('--batch', type=float, default=0.02)
    ap.add_argument('--tz', type=float, default=0.7)
    ap.add_argument('--S', type=float, default=4.0)
    ap.add_argument('--lam_min', type=float, default=0.1)
    ap.add_argument('--lam_max', type=float, default=30.0)
    ap.add_argument('--fq_iters', type=int, default=60)
    ap.add_argument('--fq_steps', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    R.CFG.update(design=a.design, tz=a.tz, S=a.S, fp=1.0, tcs=(0.99,))
    R.build_actions()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    out = P.run_dir(a.tag)
    os.makedirs(out, exist_ok=True)
    json.dump(vars(a), open(f'{out}/args.json', 'w'), indent=1)

    def load(sp, trusted=False):
        S = [Slide(t, 'mix') for t in load_tables(sp, a.design, a.g_level, 'kaiko_vitl14', 'cpu', trusted_only=trusted)]
        for s in S:
            s.cluster = cluster_stats(s)
        return S

    tr, te = load('train', trusted=P.NON_TRUSTED_SKIP), load('test')
    log = lambda r: print(r, flush=True)
    t0 = time.time()
    if a.eps > 0:
        core = fit_core(tr, a, log, rng)
        torch.save(core.state_dict(), f'{out}/core.pt')
        Ep2.critic, Ep2.eps = core, a.eps
    else:
        Ep2.critic = None
    stop = fit_stop(tr, a, log)
    torch.save(stop.state_dict(), f'{out}/stop.pt')
    print(f'fitted in {time.time() - t0:.0f} s', flush=True)
    lams = tuple(float(x) for x in np.geomspace(a.lam_min, a.lam_max, 13).round(3))
    res = {'train': {}, 'test': {}}

    def actor(lam):
        return stop_actor(stop, a.batch)

    for sp, S in (('train', tr), ('test', te)):
        res[sp].update(rule_rows(S))
        for lam in lams:
            res[sp][f'RL lam={lam:g}'] = R.agg([R.run_with(make_ep, s, stop_actor(stop, a.batch), lam).summary() for s in S])
        if sp == 'test':
            for lam in (0.3, 1.0, 3.0):
                res[sp][f'hindsight lam={lam:g}'] = R.agg([hindsight(s, lam, a.batch) for s in S])
    json.dump(res, open(f'{P.ROOT}/stop_{a.tag}.json', 'w'), indent=1)
    print(R.table(res['test'], f'{a.tag}, test (ideal segmenter, est. s)'), flush=True)


if __name__ == '__main__':
    main()
