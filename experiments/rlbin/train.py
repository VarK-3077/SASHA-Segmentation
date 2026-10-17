# PPO training loop for the binning agent. Each update plays a handful of training slides to
# the end, runs the clipped update, and every few updates scores the val slides with the
# deterministic policy; the best val Dice checkpoint is kept.

import argparse
import json
import os
import time

import numpy as np
import torch

from experiments.rlbin import paths as P
from experiments.rlbin.env import Slide
from experiments.rlbin.metrics import aggregate, episode_metrics
from experiments.rlbin.policy import Policy
from experiments.rlbin.ppo import run_episode, update


def load_slides(split, device, trusted_only):
    out = []
    for f in sorted(os.listdir(P.CACHE)):
        c = np.load(os.path.join(P.CACHE, f))
        if str(c['split']) != split or (trusted_only and not bool(c['trust_normals'])):
            continue
        out.append(Slide(f[:-4], c, device))
    return out


def evaluate(policy, slides, device):
    rows = []
    for s in slides:
        _, env = run_episode(policy, s, device, deterministic=True)
        rows.append(episode_metrics(s, env.status, env.steps))
    return aggregate(rows), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--updates', type=int, default=600)
    ap.add_argument('--episodes', type=int, default=8)
    ap.add_argument('--eval_every', type=int, default=10)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--tag', default='l3')
    a = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    train = load_slides('train', a.device, trusted_only=True)
    val = load_slides('val', a.device, trusted_only=False)
    print(f'train {len(train)} slides ({sum(s.gt_pos > 0 for s in train)} tumour), val {len(val)}', flush=True)
    policy = Policy().to(a.device)
    opt = torch.optim.AdamW(policy.parameters(), lr=a.lr, weight_decay=1e-4)
    log = open(os.path.join(P.ROOT, f'train_{a.tag}.jsonl'), 'a')
    best = -1
    for u in range(a.updates):
        t0 = time.time()
        episodes = []
        for s in rng.choice(train, a.episodes, replace=False):
            rec, env = run_episode(policy, s, a.device)
            episodes.append((s, rec))
        stats = update(policy, opt, episodes, a.device)
        stats.update(update=u, ret=float(np.mean([r['reward'].sum() for _, r in episodes])),
                     dice=float(np.mean([final_dice(s, r) for s, r in episodes])),
                     zoom_frac=float(np.mean([(r['final_status'] == 1).mean() for _, r in episodes])),
                     steps=float(np.mean([len(r['action']) for _, r in episodes])), sec=time.time() - t0)
        if u % a.eval_every == 0 or u == a.updates - 1:
            agg, _ = evaluate(policy, val, a.device)
            stats['val'] = agg
            if agg['dice_all'] > best:
                best = agg['dice_all']
                torch.save(policy.state_dict(), os.path.join(P.CKPT, f'best_{a.tag}.pt'))
            torch.save(policy.state_dict(), os.path.join(P.CKPT, f'last_{a.tag}.pt'))
            print(f"u{u} ret {stats['ret']:.3f} zoom {stats['zoom_frac']:.2f} steps {stats['steps']:.0f} | "
                  f"val dice {agg['dice_all']:.3f} tumour {agg['dice_tumour']:.3f} zoom {agg['zoom_frac']:.2f} "
                  f"bnd-zoom {agg.get('boundary_zoom', 0):.2f} small-zoom {agg.get('small_zoom', 0):.2f} "
                  f"core-core {agg.get('core_core', 0):.2f} | {stats['sec']:.0f}s", flush=True)
        log.write(json.dumps(stats) + '\n'); log.flush()


def final_dice(slide, rec):
    """Final Dice of a recorded episode: empty-mask Dice plus the summed rewards."""
    empty = 1.0 if slide.gt_pos == 0 else 0.0
    return empty + float(rec['reward'].sum())


if __name__ == '__main__':
    main()
