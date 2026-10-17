# Episode collection and the clipped PPO update. An episode is replayed for the update from
# its per-step status vectors alone, since every other input is static per slide.

import numpy as np
import torch

from experiments.rlbin.env import Env
from experiments.rlbin.policy import action_mask

GAMMA, LAM, CLIP, ENT, VCOEF = 1.0, 0.95, 0.2, 0.01, 0.5


def run_episode(policy, slide, device, deterministic=False):
    env = Env(slide)
    status, glob = env.reset()
    rec = {k: [] for k in ('status', 'glob', 'action', 'logp', 'value', 'reward')}
    while not env.done:
        st = torch.tensor(status, device=device)
        gl = torch.tensor(glob, device=device)
        with torch.no_grad():
            logits, v = policy(slide.static, slide.dyn_features(st), gl)
            logits = logits.masked_fill(~action_mask(st, slide.cap, slide.n_act), float('-inf'))
            dist = torch.distributions.Categorical(logits=logits)
            a = logits.argmax() if deterministic else dist.sample()
            lp = dist.log_prob(a)
        r = env.step(int(a))
        rec['status'].append(status); rec['glob'].append(glob); rec['action'].append(int(a))
        rec['logp'].append(float(lp)); rec['value'].append(float(v)); rec['reward'].append(r)
        status, glob = env.obs()
    rec = {k: np.array(v) for k, v in rec.items()}
    rec['adv'], rec['ret'] = gae(rec['reward'], rec['value'])
    rec['final_status'] = env.status.copy()
    return rec, env


def gae(rewards, values):
    adv = np.zeros(len(rewards), np.float32)
    last = 0.0
    for t in reversed(range(len(rewards))):
        nxt = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + GAMMA * nxt - values[t]
        last = delta + GAMMA * LAM * last
        adv[t] = last
    return adv, adv + values.astype(np.float32)


ROWS_PER_CHUNK = 96 * 1000   # steps x tiles per forward pass; a 14k-tile slide gets 6-step chunks


def update(policy, opt, episodes, device, epochs=4, ent=ENT):
    """episodes: list of (slide, rec). Returns mean loss parts."""
    all_adv = np.concatenate([r['adv'] for _, r in episodes])
    mu, sd = all_adv.mean(), all_adv.std() + 1e-8
    stats, rng = [], np.random.default_rng()
    chunk_of = lambda s: max(4, ROWS_PER_CHUNK // s.n)
    for _ in range(epochs):
        pieces = [(s, r, i) for s, r in episodes for i in range(0, len(r['action']), chunk_of(s))]
        rng.shuffle(pieces)
        for s, r, i in pieces:
            sl = slice(i, i + chunk_of(s))
            st = torch.tensor(r['status'][sl], device=device)
            gl = torch.tensor(r['glob'][sl], device=device)
            logits, v = policy(s.static.expand(len(st), -1, -1), s.dyn_features(st), gl)
            logits = logits.masked_fill(~action_mask(st, s.cap, s.n_act), float('-inf'))
            dist = torch.distributions.Categorical(logits=logits)
            a = torch.tensor(r['action'][sl], device=device)
            ratio = torch.exp(dist.log_prob(a) - torch.tensor(r['logp'][sl], device=device))
            adv = torch.tensor((r['adv'][sl] - mu) / sd, device=device)
            pl = -torch.min(ratio * adv, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv).mean()
            vl = torch.nn.functional.mse_loss(v, torch.tensor(r['ret'][sl], device=device))
            entropy = dist.entropy().mean()
            loss = pl + VCOEF * vl - ent * entropy
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            stats.append((pl.item(), vl.item(), entropy.item()))
    return dict(zip(('policy_loss', 'value_loss', 'entropy'), np.mean(stats, 0).tolist()))
