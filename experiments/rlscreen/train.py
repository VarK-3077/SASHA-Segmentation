# Trains a screening policy in two stages. Behaviour cloning: the teacher plays training episodes
# (later DAgger rounds let the student play while the teacher labels its states) and the student
# learns the teacher's per-unit actions and stop by cross-entropy, units balanced by action.
# Then PPO with lambda drawn per episode (log-uniform), so one policy covers every speed/quality
# trade-off: per-unit clipped surrogates sharing the round advantage (GAE over rounds), a stop
# surrogate, entropy, value loss, and KL(policy || frozen cloned policy) holding it near the
# clone. The final checkpoint is the result; val episodes are only logged.
# Usage: python -m experiments.rlscreen.train --design coarse --g_level 6 --tag NAME [--state light] [--kl 0.1]

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.rlscreen import paths as P
from experiments.rlscreen import teacher as T
from experiments.rlscreen.env import Episode, observation
from experiments.rlscreen.policy import ScreenPolicy, distributions
from experiments.rlscreen.rollout import load_tables, play


def lam_sample(rng, a):
    return float(np.exp(rng.uniform(np.log(a.lam_min), np.log(a.lam_max))))


def pick_slides(rng, tables, n):
    tum = [t for t in tables if t.has_tumour]
    nor = [t for t in tables if not t.has_tumour]
    k = n // 2
    return list(rng.choice(tum, k, replace=False)) + list(rng.choice(nor, n - k, replace=False))


def forward(policy, tab, tr, state):
    planes, scal = observation(tab, tr['s'], state)
    logits, stop_logit, value = policy(planes[None], scal[None], tab.tissue[None])
    return distributions(logits[0], stop_logit[0], tr['allowed']), value[0]


def bc(policy, tables, a, rng, log):
    opt = torch.optim.AdamW(policy.parameters(), lr=a.bc_lr, weight_decay=1e-4)
    for it in range(a.bc_iters):
        data = []
        for tab in tables:
            lam = lam_sample(rng, a)
            actor = 'teacher' if it == 0 else 'policy'
            _, trans = play(tab, lam, actor, a.metric, a.S, policy=policy, state=a.state, tau=a.tau)
            if actor == 'policy':  # DAgger: relabel the student's states with the teacher's choice
                for tr in trans:
                    ep = Episode(tab, tr['s']['lam'], a.metric, a.S)
                    ep.screened, ep.zoomed, ep.round, ep.spent = tr['s']['screened'], tr['s']['zoomed'], tr['s']['round'], tr['s']['spent']
                    tr['stop'], tr['a'] = T.act(ep, tau=a.tau)
            data += [(tab, tr) for tr in trans]
        for ep_i in range(a.bc_epochs):
            rng.shuffle(data)
            losses = []
            for tab, tr in data:
                (ud, sd), _ = forward(policy, tab, tr, a.state)
                tgt = tr['a']
                m = tab.tissue & ~torch.tensor(tr['stop'], device=tgt.device)
                lp = ud.log_prob(tgt)
                w, n_cls = torch.zeros_like(lp), 0
                for c in range(3):  # balance the rare screen / zoom labels against none
                    sel = m & (tgt == c)
                    if sel.any():
                        w[sel] = 1.0 / sel.sum(); n_cls += 1
                loss_u = -(lp * w).sum() / max(1, n_cls)
                loss_s = F.binary_cross_entropy_with_logits(sd.logits, torch.tensor(float(tr['stop']), device=tgt.device))
                loss = loss_u + loss_s
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
                losses.append(loss.item())
            log({'stage': 'bc', 'iter': it, 'epoch': ep_i, 'loss': float(np.mean(losses)), 'transitions': len(data)})


def gae(rewards, values, lam=0.95):
    adv, last = np.zeros(len(rewards)), 0.0
    for i in reversed(range(len(rewards))):
        nxt = values[i + 1] if i + 1 < len(values) else 0.0
        last = rewards[i] + nxt - values[i] + lam * last
        adv[i] = last
    return adv, adv + np.asarray(values)


def ppo(policy, ref, tables, a, rng, log, val, save):
    opt = torch.optim.AdamW(policy.parameters(), lr=a.lr, weight_decay=1e-4)
    for u in range(a.updates):
        t0 = time.time()
        batch, stats, returns = [], [], []
        for tab in pick_slides(rng, tables, a.episodes):
            summ, trans = play(tab, lam_sample(rng, a), 'policy', a.metric, a.S, policy=policy, state=a.state)
            adv, ret = gae([tr['r'] for tr in trans], [tr['v'] for tr in trans])
            for tr, ad, rt in zip(trans, adv, ret):
                tr['adv'], tr['ret'] = ad, rt
            batch += [(tab, tr) for tr in trans]
            stats.append(summ)
            returns.append(sum(tr['r'] for tr in trans))
        advs = np.array([tr['adv'] for _, tr in batch])
        mu, sd = advs.mean(), advs.std() + 1e-8
        for _ in range(a.epochs):
            order = rng.permutation(len(batch))
            for i in order:
                tab, tr = batch[i]
                (ud, sdist), v = forward(policy, tab, tr, a.state)
                A = (tr['adv'] - mu) / sd
                st = torch.tensor(float(tr['stop']), device=v.device)
                r_s = torch.exp(sdist.log_prob(st) - tr['lp_s'])
                loss = -torch.min(r_s * A, r_s.clamp(1 - a.clip, 1 + a.clip) * A)
                m = tab.tissue
                ent = sdist.entropy()
                if not tr['stop']:
                    r_u = torch.exp(ud.log_prob(tr['a']) - tr['lp_u'])[m]
                    loss = loss - torch.min(r_u * A, r_u.clamp(1 - a.clip, 1 + a.clip) * A).mean()
                    ent = ent + ud.entropy()[m].mean()
                loss = loss + 0.5 * (v - tr['ret']) ** 2 - a.ent * ent
                if ref is not None and a.kl > 0:
                    with torch.no_grad():
                        (rud, rsd), _ = forward(ref, tab, tr, a.state)
                    kl = torch.distributions.kl_divergence(sdist, rsd) + torch.distributions.kl_divergence(ud, rud)[m].mean()
                    loss = loss + a.kl * kl
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5); opt.step()
        row = {'stage': 'ppo', 'update': u, 'sec': time.time() - t0,
               'return': float(np.mean(returns)),
               'zoom_frac': float(np.mean([s['zoom_frac'] for s in stats])), 'screen_frac': float(np.mean([s['screen_frac'] for s in stats])),
               'area_recall_tumour': float(np.mean([s['area_recall'] for s in stats if s['tumour']] or [0])),
               'rounds': float(np.mean([s['rounds'] for s in stats]))}
        if u % a.eval_every == 0 or u == a.updates - 1:
            row['val'] = val()
            save(f'u{u}')
        log(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--design', choices=('coarse', 'full'), required=True)
    ap.add_argument('--g_level', type=int, default=6)
    ap.add_argument('--backbone', default='kaiko_vitl14')
    ap.add_argument('--state', choices=('full', 'light'), default='full')
    ap.add_argument('--metric', choices=('area', 'lesion', 'mix'), default='mix')
    ap.add_argument('--S', type=float, default=4.0)
    ap.add_argument('--lam_min', type=float, default=0.1)
    ap.add_argument('--lam_max', type=float, default=10.0)
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--bc_iters', type=int, default=2)
    ap.add_argument('--bc_epochs', type=int, default=3)
    ap.add_argument('--bc_lr', type=float, default=1e-3)
    ap.add_argument('--updates', type=int, default=300)
    ap.add_argument('--episodes', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--ent', type=float, default=0.001)
    ap.add_argument('--kl', type=float, default=0.1, help='weight of KL(policy || cloned policy); 0 = none')
    ap.add_argument('--no_bc', action='store_true', help='PPO from scratch, no cloning and no KL anchor')
    ap.add_argument('--eval_every', type=int, default=25)
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
    print(f'{a.tag}: {len(tables)} train slides ({sum(t.has_tumour for t in tables)} tumour), {len(val_tabs)} val', flush=True)
    logf = open(f'{out}/train.jsonl', 'a')

    def log(row):
        logf.write(json.dumps(row) + '\n'); logf.flush()
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != 'val'}, row.get('val', ''), flush=True)

    def val():
        res = [play(t, 1.0, 'policy', a.metric, a.S, policy=policy, state=a.state, greedy=True)[0] for t in val_tabs]
        return {'area_recall_tumour': float(np.mean([r['area_recall'] for r in res if r['tumour']])),
                'lesion_recall_tumour': float(np.mean([r['lesion_recall'] for r in res if r['tumour']])),
                'zoom_frac': float(np.mean([r['zoom_frac'] for r in res])), 'seconds': float(np.mean([r['seconds'] for r in res]))}

    policy = ScreenPolicy(a.state).to(a.device)
    save = lambda name: torch.save(policy.state_dict(), f'{out}/{name}.pt')
    ref = None
    if not a.no_bc:
        bc(policy, tables, a, rng, log)
        save('bc')
        log({'stage': 'bc_done', 'val': val()})
        ref = copy.deepcopy(policy).eval()
        for p in ref.parameters():
            p.requires_grad = False
    ppo(policy, ref, tables, a, rng, log, val, save)
    save('final')
    print(f'{a.tag} done', flush=True)


if __name__ == '__main__':
    main()
