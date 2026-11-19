# The RL cascade, second version: the cascade's three decisions (read a unit at L3, segment it at L2, paint it as
# core) are made by small policy networks trained with PPO on the reward the real segmenter's outcome gives, in
# the fast simulator. What changed from the first RL cascade: the policies do not read raw tokens of the slides
# they train on. Their state is the output of frozen readers scored out of fold (the supervised L6 and L3 heads,
# the token-level L3 head, optionally a second L6 head), so nothing in the state is memorised, and the networks
# are two-layer MLPs.
#   L6 policy: per tissue unit, screen at L3 or not.            L3 policy: per screened unit, leave / zoom / paint.
# Reward per unit (the first RL cascade's): share of the slide's cleaned Dice + W_LES per lesion whose best
# candidate lies in the unit - W_FP x score per false-positive candidate - lam x the unit's own seconds. Border
# tracing runs after the policy as in the supervised cascade. --init rule clones the supervised rule first
# (cross-entropy on its decisions), --init consensus clones the consensus policy (vsel.py), --init none starts
# from random weights. --state sup+tok | tok | sup picks
# the L3 readers in the state. Cross-fitted: five policies each trained without one fold play that fold's slides
# (greedy); a sixth trained on every train slide is the one the test slides get.
# Usage: python -m experiments.rlpolicy.rl3 --tok3 t1 --out NAME --lams 0.5 1 2 [--tok6 g0] [--init rule]

import argparse
import os
import pickle
from multiprocessing import get_context

import numpy as np
import torch
import torch.nn as nn

from experiments.rlpolicy import sim as S
from experiments.rlpolicy import where as W
from experiments.rlprobe.cost import L2, L3

W_LES, W_FP = 0.248, 0.0497
LEAVE, ZOOM, PAINT = 0, 1, 2
SLIDES = None


def lg(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)) / 5


def rank(v, m):
    out = np.zeros(v.shape, np.float32)
    if m.sum() > 1:
        out[m] = np.argsort(np.argsort(v[m])) / (m.sum() - 1)
    return out


def states(s, kind, tok6):
    """(x6, x3): one row per tissue unit, in the order of np.flatnonzero(tissue)."""
    m = s.tissue
    c6 = [lg(s.pg), rank(s.pg, m), np.full(m.shape, np.log(m.sum() + 1.0) / 8)] + ([lg(s.g_max), lg(s.g_mean)] if tok6 else [])
    c3 = [lg(s.pg)] + ([lg(s.g_max)] if tok6 else [])
    if 'sup' in kind:
        c3 += [lg(s.p3), lg(s.p_core)]
    if 'tok' in kind:
        c3 += [lg(s.t_top4), lg(s.t_top16), lg(s.t_max), lg(s.t_unit), lg(s.t_mean), lg(s.t_core)]
    f = lambda cols: torch.from_numpy(np.stack([c[m] for c in cols], 1).astype(np.float32))
    return f(c6), f(c3)


def mlp(d_in, d_out, h=64):
    return nn.Sequential(nn.Linear(d_in, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, d_out))


class Nets(nn.Module):
    def __init__(self, d6, d3):
        super().__init__()
        self.pi6, self.v6, self.pi3, self.v3 = mlp(d6, 1), mlp(d6, 1), mlp(d3, 3), mlp(d3, 1)


def rule_actions(s, veto=0.0):
    """The supervised rule's decisions per tissue unit; with veto > 0 the consensus policy's (no zoom or paint
    where the token-level head's top-16 score is below it)."""
    m = s.tissue
    scr = (s.pg >= 0.6)[m]
    want = (s.p3 >= 0.5)[m] & (s.t_top16 >= veto)[m]
    a3 = np.where(want, np.where((s.p_core >= 0.99)[m], PAINT, ZOOM), LEAVE)
    return torch.from_numpy(scr.astype(np.float32)), torch.from_numpy(a3)


def clone(nets, data, steps=3000, seed=0, veto=0.0):
    """Behaviour cloning of the teacher (supervised rule or consensus policy): both policies fitted to its decisions."""
    x6 = torch.cat([d['x6'] for d in data]); x3 = torch.cat([d['x3'] for d in data])
    t6, t3 = (torch.cat(v) for v in zip(*[rule_actions(d['s'], veto) for d in data]))
    t6, t3 = t6.to(x6.device), t3.to(x6.device)
    opt = torch.optim.Adam(list(nets.pi6.parameters()) + list(nets.pi3.parameters()), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(len(x6), (8192,), generator=g).to(x6.device)
        loss = nn.functional.binary_cross_entropy_with_logits(nets.pi6(x6[i])[:, 0], t6[i]) + nn.functional.cross_entropy(nets.pi3(x3[i]), t3[i])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()


def route(s, scr_t, a3_t):
    """Tissue-ordered decisions -> the played route (simulator record with border tracing)."""
    idx = np.flatnonzero(s.tissue.ravel())
    scr, zoom, paint = (np.zeros(s.tissue.size, bool) for _ in range(3))
    scr[idx] = scr_t
    zoom[idx], paint[idx] = scr_t & (a3_t == ZOOM), scr_t & (a3_t == PAINT)
    sh = s.tissue.shape
    return S.finish(s, scr.reshape(sh), zoom.reshape(sh), paint.reshape(sh))


@torch.no_grad()
def greedy(nets, d):
    scr = (nets.pi6(d['x6'])[:, 0] >= 0).cpu().numpy()
    a3 = nets.pi3(d['x3']).argmax(1).cpu().numpy()
    return route(d['s'], scr, a3)


def rollout(nets, d, lam, rng, W=(W_LES, W_FP)):
    with torch.no_grad():
        p6 = torch.sigmoid(nets.pi6(d['x6'])[:, 0])
        lp3 = torch.log_softmax(nets.pi3(d['x3']), 1)
    dev = p6.device
    g = torch.Generator(device=dev).manual_seed(int(rng.integers(2 ** 31)))
    scr = torch.rand(len(p6), generator=g, device=dev) < p6
    a3 = torch.multinomial(lp3.exp(), 1, generator=g)[:, 0]
    s = d['s']
    out = route(s, scr.cpu().numpy(), a3.cpu().numpy())
    q = torch.from_numpy(S.unit_rewards(s, out, W[0], W[1])[s.tissue].astype(np.float32)).to(dev)
    act = a3 != LEAVE
    r3 = q * act - lam * L2 * (a3 == ZOOM)
    r6 = scr * (r3 - lam * L3)
    lp6 = torch.where(scr, torch.log(p6.clamp(min=1e-8)), torch.log((1 - p6).clamp(min=1e-8)))
    return dict(x6=d['x6'], a6=scr.float(), lp6=lp6, r6=r6, x3=d['x3'][scr], a3=a3[scr], lp3=lp3[scr].gather(1, a3[scr][:, None])[:, 0], r3=r3[scr])


def ppo(data, lam, a, seed, log=None, veto=0.0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    nets = Nets(data[0]['x6'].shape[1], data[0]['x3'].shape[1]).to(data[0]['x6'].device)
    if a.init != 'none':
        clone(nets, data, seed=seed, veto=veto)
    opt = torch.optim.Adam([dict(params=list(nets.pi6.parameters()) + list(nets.pi3.parameters()), lr=a.lr),
                            dict(params=list(nets.v6.parameters()) + list(nets.v3.parameters()), lr=a.critic_lr)])
    tum = [i for i, d in enumerate(data) if d['s'].tumour]
    nor = [i for i, d in enumerate(data) if not d['s'].tumour]
    norm = lambda x: (x - x.mean()) / (x.std() + 1e-8)
    if log:
        m = S.summarise([greedy(nets, d) for d in data])
        log(f"    start ({a.init}): greedy on its train slides {m['sec']:.3f} s  Dice {m['dice']:.4f}  FROC {m['froc']:.4f}  L3 {m['l3']:.0f}  L2 {m['l2']:.0f}  paint {m['paint']:.1f}")
    for up in range(1, a.updates + 1):
        pick = np.r_[rng.choice(tum, a.episodes // 2), rng.choice(nor, a.episodes - a.episodes // 2)]
        b = [rollout(nets, data[i], lam, rng, (a.w_les, a.w_fp)) for i in pick]
        cat = {k: torch.cat([x[k] for x in b]) for k in b[0]}
        with torch.no_grad():
            adv6, adv3 = norm(cat['r6'] - nets.v6(cat['x6'])[:, 0] * a.vscale), norm(cat['r3'] - nets.v3(cat['x3'])[:, 0] * a.vscale)
        for _ in range(a.epochs):
            lp6 = -nn.functional.binary_cross_entropy_with_logits(nets.pi6(cat['x6'])[:, 0], cat['a6'], reduction='none')
            ls3 = torch.log_softmax(nets.pi3(cat['x3']), 1)
            lp3 = ls3.gather(1, cat['a3'][:, None])[:, 0]
            surr = lambda lp, old, adv: -torch.min((lp - old).exp() * adv, (lp - old).exp().clamp(1 - a.clip, 1 + a.clip) * adv).mean()
            p6 = torch.sigmoid(nets.pi6(cat['x6'])[:, 0]).clamp(1e-6, 1 - 1e-6)
            ent = -(p6 * p6.log() + (1 - p6) * (1 - p6).log()).mean() - (ls3.exp() * ls3).sum(1).mean()
            vl = ((nets.v6(cat['x6'])[:, 0] - cat['r6'] / a.vscale) ** 2).mean() + ((nets.v3(cat['x3'])[:, 0] - cat['r3'] / a.vscale) ** 2).mean()
            loss = surr(lp6, cat['lp6'], adv6) + surr(lp3, cat['lp3'], adv3) + vl - a.ent * ent
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(nets.parameters(), 1.0)
            opt.step()
        if log and up % a.log_every == 0:
            m = S.summarise([greedy(nets, d) for d in data])
            log(f"    update {up}: greedy on its train slides {m['sec']:.3f} s  Dice {m['dice']:.4f}  FROC {m['froc']:.4f}  L3 {m['l3']:.0f}  L2 {m['l2']:.0f}  paint {m['paint']:.1f}")
    return nets


def player(run):
    """A saved run's all-train policy as a function slide -> played route (greedy)."""
    def play(s):
        x6, x3 = states(s, run['state'], run['tok6'])
        nets = Nets(x6.shape[1], x3.shape[1])
        nets.load_state_dict(run['nets'])
        return greedy(nets, dict(s=s, x6=x6, x3=x3))
    return play


def job(task):
    lam, k, a = task
    fold = W.folds()
    torch.set_num_threads(1)
    on = lambda d: dict(s=d['s'], x6=d['x6'].to(a.device), x3=d['x3'].to(a.device))
    lines = []
    say = lambda t: (lines.append(t), print(f'  [lam {lam}]' + t, flush=True))
    nets = ppo([on(d) for d in SLIDES if fold[d['s'].name] != k], lam, a, seed=a.seed + 10 * (k + 1), log=say if k < 0 else None, veto=a.vetoes[k] if a.init == 'consensus' else 0.0)
    held = [(d['s'].name, greedy(nets, on(d))) for d in SLIDES if fold[d['s'].name] == k]
    return lam, k, {n: v.detach().cpu().clone() for n, v in nets.state_dict().items()}, held, lines


def main():
    global SLIDES
    ap = argparse.ArgumentParser()
    ap.add_argument('--tok3', required=True)
    ap.add_argument('--tok6', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--lams', type=float, nargs='+', required=True)
    ap.add_argument('--state', default='sup+tok')
    ap.add_argument('--init', choices=('rule', 'consensus', 'none'), default='rule')
    ap.add_argument('--teacher', default='vsel_t1@t_top16:0.5', help='--init consensus: the vsel run whose fold vetoes the clones imitate')
    ap.add_argument('--updates', type=int, default=400)
    ap.add_argument('--episodes', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--critic_lr', type=float, default=1e-3)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--ent', type=float, default=0.003)
    ap.add_argument('--vscale', type=float, default=0.05)
    ap.add_argument('--log_every', type=int, default=100)
    ap.add_argument('--w_les', type=float, default=W_LES, help='reward per lesion detected')
    ap.add_argument('--w_fp', type=float, default=W_FP, help='penalty per false-positive candidate (times its score)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--device', default='cpu', help='where the networks live (the simulator stays on the CPU)')
    a = ap.parse_args()
    slides = S.load('train', a.tok3, tok6=a.tok6)
    a.vetoes = None
    if a.init == 'consensus':       # each fold's clone imitates the veto fitted without that fold; index -1 = all slides
        name, rk = a.teacher.split('@')
        t = pickle.load(open(f'{W.ROOT}/search/{name}/res.pkl', 'rb'))['runs'][rk]
        a.vetoes = [float(rk.split(':')[1]) * v for v in t['v_fold']] + [float(t['theta'].veto)]
    SLIDES = [dict(s=s, **dict(zip(('x6', 'x3'), states(s, a.state, a.tok6)))) for s in slides]
    rule = [S.rule(s) for s in slides]
    tum = [i for i, s in enumerate(slides) if s.tumour]
    out_dir = f'{W.ROOT}/search/{a.out}'
    os.makedirs(out_dir, exist_ok=True)
    fmt = lambda m: f"{m['sec']:.3f} s  Dice {m['dice']:.4f}  FROC {m['froc']:.4f}  L3 {m['l3']:.0f}  L2 {m['l2']:.0f}  paint {m['paint']:.1f}  FP {m['fp']:.2f}"
    print('rule            ', fmt(S.summarise(rule)), flush=True)
    res = dict(names=[s.name for s in slides], rule=rule, runs={}, args=vars(a))
    tasks = [(lam, k, a) for lam in a.lams for k in list(range(W.N_FOLDS)) + [-1]]
    got = {}
    with get_context('fork').Pool(min(a.workers, len(tasks))) as pool:
        for lam, k, sd, held, lines in pool.imap_unordered(job, tasks):
            got[(lam, k)] = (sd, dict(held), lines)
            if all((lam, j) in got for j in list(range(W.N_FOLDS)) + [-1]):
                by = {}
                for j in range(W.N_FOLDS):
                    by.update(got[(lam, j)][1])
                cv = [by[s.name] for s in slides]
                print(f'lam {lam}: cross-fitted {fmt(S.summarise(cv))}  lost {sum(max(0, rule[i]["found"] - cv[i]["found"]) for i in tum)}  gained '
                      f'{sum(max(0, cv[i]["found"] - rule[i]["found"]) for i in tum)}  Dice below rule on {sum(cv[i]["dice"] < rule[i]["dice"] - 1e-9 for i in tum)}', flush=True)
                res['runs'][f'lam{lam}'] = dict(cv=cv, nets=got[(lam, -1)][0], fold_nets=[got[(lam, j)][0] for j in range(W.N_FOLDS)], state=a.state, tok6=a.tok6)
                pickle.dump(res, open(f'{out_dir}/res.pkl', 'wb'))
    print('done', flush=True)


if __name__ == '__main__':
    main()
