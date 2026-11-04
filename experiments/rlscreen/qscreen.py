# Value-based screening with an analytic lambda. Under the ideal segmenter the reward splits
# exactly over units: zooming unit u earns S^J * (share_u - lam * price_zoom), screening it costs
# S^J * lam * price_screen and earns only through the zoom it may enable. So the agent learns two
# lambda-free or lambda-conditioned per-unit values and derives every decision from them:
#   g(u)       expected n * share_u of zooming u now (regressed on its exact value: supervised),
#   q_s(u,lam) expected max(0, g'(u) - lam * kz) of screening u first, where g' is the zoom value
#              once u's level-3 view is revealed (bootstrapped from a frozen copy: the RL part).
# Act: per unit argmax of {none: 0, zoom: g - lam*kz, screen: q_s - lam*ks}; stop when the round's
# summed positive net value no longer pays the round's fixed price. lam is the only knob, used
# analytically at deploy time; no thresholds are tuned on any split. States come from teacher
# rollouts at several thresholds, then from the agent's own rollouts with a little exploration
# (fitted Q-iteration with dataset aggregation).
# Usage: python -m experiments.rlscreen.qscreen --design coarse --g_level 6 --state light --tag NAME

import argparse
import copy
import json
import os
import time

import numpy as np
import torch

from experiments.rlscreen import costs as C
from experiments.rlscreen import paths as P
from experiments.rlscreen import teacher as T
from experiments.rlscreen.env import NONE, SCREEN, ZOOM, Episode, observation, snapshot
from experiments.rlscreen.evaluate import agg
from experiments.rlscreen.policy import ScreenPolicy
from experiments.rlscreen.rollout import load_tables, play

SCALE = 10.0      # targets are n * share / SCALE
CAP = 300.0       # n * share capped here (a whole-slide lesion in one unit); only affects the extreme tail


def prices(backbone):
    """Per-unit prices in units of n * (cost fraction), and the fixed per-round price."""
    return C.ZOOM_PER_UNIT / C.FULL_PER_UNIT, C.L3_PER_UNIT[backbone] / C.FULL_PER_UNIT, C.ROUND / C.FULL_PER_UNIT


def values(net, tab, s, state):
    planes, scal = observation(tab, s, state)
    out = net(planes[None], scal[None], tab.tissue[None])[0][0]       # (hu, wu, 3)
    return out[..., 0] * SCALE, out[..., 1] * SCALE


def make_actor(net, state, eps=0.0, rng=None):
    def actor(ep):
        t = ep.t
        kz, ks, kr = prices(t.backbone)
        with torch.no_grad():
            g, qs = values(net, t, snapshot(ep), state)
        allowed = ep.allowed()
        net_v = torch.stack([torch.zeros_like(g), qs - ep.lam * ks, g - ep.lam * kz], -1).masked_fill(~allowed, -1e9)
        a = net_v.argmax(-1)
        a = torch.where(t.tissue, a, torch.full_like(a, NONE))
        if eps > 0:
            flip = torch.from_numpy(rng.random(a.shape) < eps).to(a.device) & allowed[..., ZOOM]
            a = torch.where(flip, torch.full_like(a, ZOOM), a)
        gain = net_v.max(-1).values.clamp(min=0)[t.tissue & (a != NONE)].sum()
        stop = bool((a == NONE).all()) or float(gain) <= ep.lam * kr
        return stop, a
    return actor


def collect(tables, actor_fn, rng, a, n_lam=1):
    """Plays episodes and keeps every visited state as (table, snapshot)."""
    data = []
    for tab in tables:
        for _ in range(n_lam):
            lam = float(np.exp(rng.uniform(np.log(a.lam_min), np.log(a.lam_max))))
            ep = Episode(tab, lam, a.metric, a.S)
            while not ep.done:
                s = snapshot(ep)
                data.append((tab, s))
                stop, act = actor_fn(ep)
                ep.step(stop, act)
    return data


def screened_all(tab, s):
    s2 = dict(s)
    s2['screened'] = s['screened'] | (tab.tissue & ~s['zoomed'])
    return s2


def train_epoch(net, target, data, a, opt, rng):
    losses = []
    for i in rng.permutation(len(data)):
        tab, s = data[i]
        s = dict(s)
        s['lam'] = float(np.exp(rng.uniform(np.log(a.lam_min), np.log(a.lam_max))))   # lam relabelling is free
        free = tab.tissue & ~s['zoomed']
        if not free.any():
            continue
        y = torch.from_numpy(np.minimum(tab.share[a.metric] * tab.n_units, CAP).reshape(tab.hu, tab.wu)).float().to(free.device)
        planes, scal = observation(tab, s, a.state)
        out = net(planes[None], scal[None], tab.tissue[None])[0][0]
        loss = ((out[..., 0] - y / SCALE)[free] ** 2).mean()
        if tab.design == 'coarse':
            m = free & ~s['screened']
            if m.any():
                kz = prices(tab.backbone)[0]
                with torch.no_grad():
                    g_after, _ = values(target, tab, screened_all(tab, s), a.state)
                    y_s = (g_after - s['lam'] * kz).clamp(min=0) / SCALE
                loss = loss + ((out[..., 1] - y_s)[m] ** 2).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        losses.append(float(loss))
    return float(np.mean(losses))


def evaluate(net, tabs, a, lams):
    rows = {}
    for lam in lams:
        rows[f'q lam={lam:g}'] = agg([play(t, lam, make_actor(net, a.state), a.metric, a.S)[0] for t in tabs])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--design', choices=('coarse', 'full'), required=True)
    ap.add_argument('--g_level', type=int, default=6)
    ap.add_argument('--backbone', default='kaiko_vitl14')
    ap.add_argument('--state', choices=('full', 'light'), default='light')
    ap.add_argument('--metric', choices=('area', 'lesion', 'mix'), default='mix')
    ap.add_argument('--S', type=float, default=4.0)
    ap.add_argument('--lam_min', type=float, default=0.03)
    ap.add_argument('--lam_max', type=float, default=30.0)
    ap.add_argument('--taus', type=float, nargs='*', default=[0.2, 0.35, 0.5, 0.7, 0.9])
    ap.add_argument('--iters', type=int, default=4, help='data-aggregation iterations after the teacher round')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--eps', type=float, default=0.01, help='random extra zooms in the agent rollouts')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    out = P.run_dir(a.tag)
    os.makedirs(out, exist_ok=True)
    json.dump(vars(a), open(f'{out}/args.json', 'w'), indent=1)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    tables = load_tables('train', a.design, a.g_level, a.backbone, a.device, trusted_only=P.NON_TRUSTED_SKIP)
    val_tabs = load_tables('val', a.design, a.g_level, a.backbone, a.device)
    logf = open(f'{out}/train.jsonl', 'w')

    def log(row):
        logf.write(json.dumps(row) + '\n'); logf.flush()
        print(row, flush=True)

    net = ScreenPolicy(a.state).to(a.device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    data = []
    for tau in a.taus:
        data += collect(tables, lambda ep, tau=tau: T.act(ep, tau=tau, follow=True), rng, a)
    for it in range(a.iters + 1):
        t0 = time.time()
        if it > 0:
            data += collect(tables, make_actor(net, a.state, a.eps, rng), rng, a, n_lam=2)
        target = copy.deepcopy(net).eval()
        for e in range(a.epochs):
            loss = train_epoch(net, target, data, a, opt, rng)
            target = copy.deepcopy(net).eval()
        net.eval()
        v = evaluate(net, val_tabs, a, tuple(np.geomspace(a.lam_min, a.lam_max, 4)[1:3].round(4)))
        net.train()
        log({'iter': it, 'states': len(data), 'loss': loss, 'sec': time.time() - t0,
             'val': {k: {c: round(x, 3) for c, x in r.items() if c in ('ideal_dice_tumour', 'zoom_frac', 'screen_frac', 'rounds', 'seconds')} for k, r in v.items()}})
        torch.save(net.state_dict(), f'{out}/it{it}.pt')
    torch.save(net.state_dict(), f'{out}/final.pt')
    net.eval()
    test = load_tables('test', a.design, a.g_level, a.backbone, a.device)
    rows = evaluate(net, test, a, tuple(float(x) for x in np.geomspace(a.lam_min, a.lam_max, 7).round(4)))
    json.dump(rows, open(f'{P.ROOT}/offline_{a.tag}.json', 'w'), indent=1)
    keys = ('ideal_dice_tumour', 'area_recall', 'lesions_found', 'zoom_frac', 'screen_frac', 'rounds', 'seconds', 'seconds_normal')
    lines = [f'# Offline (ideal segmenter, cost-model A100 s), {a.tag}', '', '| policy | ' + ' | '.join(keys) + ' |', '|' + '---|' * (len(keys) + 1)]
    lines += [f'| {k} | ' + ' | '.join(f'{v[c]:.3f}' for c in keys) + ' |' for k, v in rows.items()]
    open(f'{P.ROOT}/offline_{a.tag}.md', 'w').write('\n'.join(lines))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
