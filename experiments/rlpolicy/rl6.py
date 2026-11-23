# The cascade without its L3 stage: a unit that passes the L6 decision goes straight to the segmenter (no L3 read,
# so no core painting; border tracing kept). Three screeners for that pipeline, all from the L6 pass alone:
#   thr    the supervised L6 head with a threshold re-chosen for this pipeline: the one with the highest Dice +
#          FROC on the out-of-fold train slides (ties to the faster), in the simulator with the real segmenter's
#          out-of-fold maps. It is also the teacher of the PPO screener.
#   clone  a small MLP cloned from that teacher (the PPO screener's starting point).
#   ppo    the clone fine-tuned with PPO on the segmenter-outcome reward (rl3's reward, recipe of the best agent:
#          lesion weight 8, lambda 0.3, KL anchor 0.02 to the clone), zoom cost = an L2 unit, no L3 cost.
# State of the MLPs: the L6 score, its rank among the slide's tissue units, the slide's size (rl3's L6 state).
# Cross-fitted: every train slide is played by a threshold / clone / PPO policy fitted without its fold; the
# all-train versions are the ones the test slides get.
#   python -m experiments.rlpolicy.rl6 fit                 thresholds, clones, PPO; cross-fitted simulator rows
#   python -m experiments.rlpolicy.rl6 score --split train exact scorer on the cross-fitted routes
#   python -m experiments.rlpolicy.rl6 score --split test  exact scorer, all-train policies, test read once

import argparse
import os
import pickle
import time
from multiprocessing import get_context

import numpy as np
import torch
import torch.nn as nn

from experiments.rlpolicy import sim as S
from experiments.rlpolicy import where as W
from experiments.rlpolicy.rl3 import W_FP, lg, mlp, rank
from experiments.rlprobe.cost import L2, seconds

OUT = f'{W.ROOT}/l6only'
GRID = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995)
KEYS = ('thr', 'clone', 'ppo')
SLIDES = SLIDES_X = None


def state(s):
    """One row per tissue unit, in the order of np.flatnonzero(tissue): rl3's L6 state."""
    m = s.tissue
    cols = [lg(s.pg), rank(s.pg, m), np.full(m.shape, np.log(m.sum() + 1.0) / 8)]
    return torch.from_numpy(np.stack([c[m] for c in cols], 1).astype(np.float32))


def play(s, zoom_t):
    """Tissue-ordered zoom decisions -> the played route: no L3 reads, no painting, border tracing."""
    zoom = np.zeros(s.tissue.size, bool)
    zoom[np.flatnonzero(s.tissue.ravel())] = zoom_t
    zoom = zoom.reshape(s.tissue.shape)
    none = np.zeros_like(zoom)
    out = S.finish(s, none, zoom, none)
    out['sec'] = seconds(0, out['n_l2'])
    return out


def by_thr(s, t):
    return play(s, (s.pg >= t)[s.tissue])


@torch.no_grad()
def by_net(s, pi):
    return play(s, (pi(state(s))[:, 0] >= 0).numpy())


def best_thr(idx):
    """Threshold with the highest Dice + FROC on slides idx (ties to the faster), and the whole curve."""
    curve = [(t, S.summarise([by_thr(SLIDES[i], t) for i in idx])) for t in GRID]
    return max(curve, key=lambda c: (round(c[1]['dice'] + c[1]['froc'], 4), -c[1]['sec']))[0], curve


def clone(pi, data, t, steps=3000, seed=0):
    x = torch.cat([d['x'] for d in data])
    y = torch.cat([torch.from_numpy((d['s'].pg >= t)[d['s'].tissue].astype(np.float32)) for d in data])
    opt = torch.optim.Adam(pi.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(len(x), (8192,), generator=g)
        loss = nn.functional.binary_cross_entropy_with_logits(pi(x[i])[:, 0], y[i])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()


def ppo(pi, data, a, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    v, ref = mlp(3, 1), mlp(3, 1)
    ref.load_state_dict(pi.state_dict())
    for q in ref.parameters():
        q.requires_grad_(False)
    opt = torch.optim.Adam([dict(params=pi.parameters(), lr=a.lr), dict(params=v.parameters(), lr=a.critic_lr)])
    tum = [i for i, d in enumerate(data) if d['s'].tumour]
    nor = [i for i, d in enumerate(data) if not d['s'].tumour]
    for _ in range(a.updates):
        xs, acts, lps, rs = [], [], [], []
        for i in np.r_[rng.choice(tum, a.episodes // 2), rng.choice(nor, a.episodes - a.episodes // 2)]:
            d = data[i]
            with torch.no_grad():
                p = torch.sigmoid(pi(d['x'])[:, 0])
            z = torch.rand(len(p), generator=torch.Generator().manual_seed(int(rng.integers(2 ** 31)))) < p
            s = d['s']
            q = torch.from_numpy(S.unit_rewards(s, play(s, z.numpy()), a.w_les, W_FP)[s.tissue].astype(np.float32))
            xs.append(d['x']); acts.append(z.float()); rs.append(z * (q - a.lam * L2))
            lps.append(torch.where(z, torch.log(p.clamp(min=1e-8)), torch.log((1 - p).clamp(min=1e-8))))
        x, act, lp0, r = (torch.cat(t) for t in (xs, acts, lps, rs))
        with torch.no_grad():
            adv = r - v(x)[:, 0] * a.vscale
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            r0 = torch.sigmoid(ref(x)[:, 0]).clamp(1e-6, 1 - 1e-6)
        for _ in range(a.epochs):
            logit = pi(x)[:, 0]
            lp = -nn.functional.binary_cross_entropy_with_logits(logit, act, reduction='none')
            ratio = (lp - lp0).exp()
            p = torch.sigmoid(logit).clamp(1e-6, 1 - 1e-6)
            ent = -(p * p.log() + (1 - p) * (1 - p).log()).mean()
            kl = (p * (p / r0).log() + (1 - p) * ((1 - p) / (1 - r0)).log()).mean()
            loss = -torch.min(ratio * adv, ratio.clamp(1 - a.clip, 1 + a.clip) * adv).mean() + ((v(x)[:, 0] - r / a.vscale) ** 2).mean() - a.ent * ent + a.kl * kl
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(pi.parameters()) + list(v.parameters()), 1.0)
            opt.step()
    return pi


def job(task):
    k, a = task
    torch.set_num_threads(1)
    folds = W.folds()
    idx = [i for i, d in enumerate(SLIDES_X) if folds[d['s'].name] != k]
    held = [i for i, d in enumerate(SLIDES_X) if folds[d['s'].name] == k]
    t, curve = best_thr(idx)
    data = [SLIDES_X[i] for i in idx]
    pi = mlp(3, 1)
    clone(pi, data, t, seed=k + 1)
    c_sd = {n: p.detach().clone() for n, p in pi.state_dict().items()}
    c_recs = {SLIDES[i].name: by_net(SLIDES[i], pi) for i in held}
    c_in = S.summarise([by_net(d['s'], pi) for d in data])
    ppo(pi, data, a, seed=a.seed + 10 * (k + 1))
    p_recs = {SLIDES[i].name: by_net(SLIDES[i], pi) for i in held}
    p_in = S.summarise([by_net(d['s'], pi) for d in data])
    return k, dict(t=t, curve=curve, thr={SLIDES[i].name: by_thr(SLIDES[i], t) for i in held}, clone=c_recs, ppo=p_recs,
                   clone_sd=c_sd, ppo_sd={n: p.detach().clone() for n, p in pi.state_dict().items()}, clone_in=c_in, ppo_in=p_in)


def fit(a):
    global SLIDES, SLIDES_X
    SLIDES = S.load('train')
    SLIDES_X = [dict(s=s, x=state(s)) for s in SLIDES]
    os.makedirs(OUT, exist_ok=True)
    rule = [S.rule(s) for s in SLIDES]
    tum = [i for i, s in enumerate(SLIDES) if s.tumour]
    fmt = lambda m: f"{m['sec']:.3f} s  Dice {m['dice']:.4f}  FROC {m['froc']:.4f}  L3 {m['l3']:.0f}  L2 {m['l2']:.0f}  FP {m['fp']:.2f}"
    print('full cascade rule', fmt(S.summarise(rule)), flush=True)
    with get_context('fork').Pool(6) as pool:
        got = dict(pool.imap_unordered(job, [(k, a) for k in list(range(W.N_FOLDS)) + [-1]]))
    for t, m in got[-1]['curve']:
        print(f'  L6 >= {t:<6} {fmt(m)}')
    print('thresholds: all-train', got[-1]['t'], 'folds', [got[k]['t'] for k in range(W.N_FOLDS)], flush=True)
    res = dict(names=[s.name for s in SLIDES], rule=rule, t=got[-1]['t'], t_fold=[got[k]['t'] for k in range(W.N_FOLDS)], args=vars(a),
               nets=dict(clone=got[-1]['clone_sd'], ppo=got[-1]['ppo_sd']), cv={})
    for key in KEYS:
        by = {}
        for k in range(W.N_FOLDS):
            by.update(got[k][key])
        cv = [by[s.name] for s in SLIDES]
        res['cv'][key] = cv
        print(f'{key:5s} cross-fitted {fmt(S.summarise(cv))}  lost vs full rule {sum(max(0, rule[i]["found"] - cv[i]["found"]) for i in tum)}', flush=True)
    print('all-train on its own slides: clone', fmt(got[-1]['clone_in']), '| ppo', fmt(got[-1]['ppo_in']), flush=True)
    pickle.dump(res, open(f'{OUT}/res.pkl', 'wb'))


def load_pi(sd):
    pi = mlp(3, 1)
    pi.load_state_dict(sd)
    return pi.eval()


def score(a):
    from experiments.rlcascade import report as RP
    from experiments.rlpolicy import exact as X
    res = pickle.load(open(f'{OUT}/res.pkl', 'rb'))
    slides = S.load(a.split)
    routes = {s.name: {'rule': S.rule(s)} for s in slides}
    if a.split == 'train':
        assert res['names'] == [s.name for s in slides]
        for key in KEYS:
            for s, r in zip(slides, res['cv'][key]):
                routes[s.name][key] = r
    else:
        open(f'{W.ROOT}/TEST_READS.txt', 'a').write(f"{time.strftime('%F %T')} L6-only screeners (thr {res['t']}, clone, ppo)\n")
        pis = {k: load_pi(res['nets'][k]) for k in ('clone', 'ppo')}
        for s in slides:
            routes[s.name].update(thr=by_thr(s, res['t']), clone=by_net(s, pis['clone']), ppo=by_net(s, pis['ppo']))
    ex = X.score_routes(a.split, routes, a.workers)
    label = dict(rule='full cascade, supervised rule (L6 -> L3 -> L2)', thr=f"L6 only, supervised head >= {res['t']} (teacher)", clone='L6 only, clone of the teacher', ppo='L6 only, PPO from the clone')
    same = sum(bool((routes[s.name]['thr']['zoom'] == routes[s.name]['clone']['zoom']).all()) for s in slides)
    text = '\n'.join([f"## {a.split}, {len(slides)} slides, exact scorer ({'cross-fitted' if a.split == 'train' else 'all-train policies, test read once'}); L6-only rows: no L3 read, no painting; seconds = cost model\n",
                      RP.table(ex, ['rule'] + list(KEYS), label), '', RP.BOOT_HEAD] + [RP.boot_line(ex, k, label[k]) for k in KEYS]
                     + ['', f'clone and teacher segment the same units on {same} of {len(slides)} slides']) + '\n'
    open(f'{OUT}/{a.split}.md', 'w').write(text)
    pickle.dump(ex, open(f'{OUT}/{a.split}.pkl', 'wb'))
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=('fit', 'score'))
    ap.add_argument('--split', choices=('train', 'test'), default='train')
    ap.add_argument('--lam', type=float, default=0.3)
    ap.add_argument('--w_les', type=float, default=8.0)
    ap.add_argument('--kl', type=float, default=0.02)
    ap.add_argument('--updates', type=int, default=800)
    ap.add_argument('--episodes', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--critic_lr', type=float, default=1e-3)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--ent', type=float, default=0.003)
    ap.add_argument('--vscale', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=20)
    a = ap.parse_args()
    (fit if a.step == 'fit' else score)(a)


if __name__ == '__main__':
    main()
