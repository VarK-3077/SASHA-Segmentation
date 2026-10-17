# Behaviour-cloning warm start from the oracle routing policy: zoom every unassigned boundary
# or small tile, mark every unassigned core tile, stop when none are left. States are sampled
# along oracle trajectories (random order within the allowed set) and the policy is trained to
# put its probability mass on the allowed action set. Writes a checkpoint train.py can start from.

import argparse
import os
import time

import numpy as np
import torch

from experiments.rlbin import paths as P
from experiments.rlbin.env import CORE, UNASSIGNED, ZOOMED
from experiments.rlbin.policy import Policy, action_mask
from experiments.rlbin.train import evaluate, load_slides
from experiments.sep import buckets as B


def oracle_trajectory(slide, rng, n_states):
    """Status snapshots along one random oracle roll-out, plus the allowed-action mask at each."""
    status = np.zeros(slide.n, np.int8)
    todo = [(i, ZOOMED) for i in np.flatnonzero(np.isin(slide.bucket, (B.BOUNDARY, B.SMALL)))]
    todo += [(i, CORE) for i in np.flatnonzero(slide.bucket == B.CORE)]
    rng.shuffle(todo)
    keep = sorted(rng.choice(len(todo) + 1, min(n_states, len(todo) + 1), replace=False))
    snaps, allowed = [], []
    k = 0
    for t in range(len(todo) + 1):
        if k < len(keep) and keep[k] == t:
            snaps.append(status.copy())
            allowed.append(allowed_set(slide, status))
            k += 1
        if t < len(todo):
            i, a = todo[t]
            status[i] = a
    return np.stack(snaps), np.stack(allowed)


def allowed_set(slide, status):
    free = status == UNASSIGNED
    zoom = free & np.isin(slide.bucket, (B.BOUNDARY, B.SMALL))
    core = free & (slide.bucket == B.CORE)
    stop = np.array([not (zoom.any() or core.any())])
    return np.concatenate([zoom, core, stop])


def glob_of(slide, status):
    zooms = (status == ZOOMED).sum(-1, keepdims=True).astype(np.float32)
    steps = ((status != UNASSIGNED).sum(-1, keepdims=True)).astype(np.float32)
    return np.concatenate([zooms / slide.n, zooms / max(slide.cap, 1), steps / slide.max_steps], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--level', type=int, default=3, choices=(3, 2))
    ap.add_argument('--state', default='oracle', choices=('none', 'pred', 'oracle'))
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--states_per_slide', type=int, default=24)
    ap.add_argument('--tag', default='bc')
    a = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    train = load_slides('train', a.device, trusted_only=True, state=a.state, level=a.level)
    val = load_slides('val', a.device, trusted_only=False, state=a.state, level=a.level)
    policy = Policy(d_emb=train[0].d_emb).to(a.device)
    opt = torch.optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=1e-4)
    t0 = time.time()
    for ep in range(a.epochs):
        order = rng.permutation(len(train))
        losses = []
        for si in order:
            s = train[si]
            snaps, allowed = oracle_trajectory(s, rng, a.states_per_slide)
            st = torch.tensor(snaps, device=a.device)
            gl = torch.tensor(np.stack([glob_of(s, x) for x in snaps]), device=a.device)
            logits, _ = policy(s.static.expand(len(st), -1, -1), s.dyn_features(st), gl)
            logits = logits.masked_fill(~action_mask(st, s.cap), float('-inf'))
            logp = torch.log_softmax(logits, -1)
            ok = torch.tensor(allowed, device=a.device)
            loss = -torch.logsumexp(logp.masked_fill(~ok, float('-inf')), -1).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        agg, _ = evaluate(policy, val, a.device)
        print(f"bc epoch {ep} loss {np.mean(losses):.3f} | val dice {agg['dice_all']:.3f} tumour {agg['dice_tumour']:.3f} "
              f"zoom {agg['zoom_frac']:.2f} bnd-zoom {agg.get('boundary_zoom', 0):.2f} small-zoom {agg.get('small_zoom', 0):.2f} "
              f"core-core {agg.get('core_core', 0):.2f} | {time.time() - t0:.0f}s", flush=True)
    torch.save(policy.state_dict(), os.path.join(P.CKPT, f'{a.tag}.pt'))
    print(f'saved {P.CKPT}/{a.tag}.pt', flush=True)


if __name__ == '__main__':
    main()
