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


def load_slides(split, device, trusted_only, state='none', level=3, mode='neg'):
    cache = P.CACHE if level == 3 else P.CACHE_L2
    out = []
    for f in sorted(os.listdir(cache)):
        c = np.load(os.path.join(cache, f))
        if str(c['split']) != split or (trusted_only and not bool(c['trust_normals'])):
            continue
        c = dict(c)
        if state == 'pred':
            c['p_bucket'] = np.load(os.path.join(P.PBUCKET if level == 3 else P.PBUCKET + '_l2', f[:-4] + '.npy'))
        out.append(Slide(f[:-4], c, device, state, mode))
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
    ap.add_argument('--ent', type=float, default=0.01)
    ap.add_argument('--balanced', action='store_true', help='half the episodes on tumour slides')
    ap.add_argument('--state', default='none', choices=('none', 'pred', 'oracle'))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--level', type=int, default=3, choices=(3, 2))
    ap.add_argument('--init', default='', help='checkpoint to start the policy from (e.g. a bc.py warm start)')
    ap.add_argument('--mode', default='neg', choices=('neg', 'lr'), help='what an untouched tile means at the end')
    ap.add_argument('--zoom_cost', type=float, default=0.0, help='Dice units charged per zoom in the training reward')
    ap.add_argument('--tag', default='l3')
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    train = load_slides('train', a.device, trusted_only=True, state=a.state, level=a.level, mode=a.mode)
    val = load_slides('val', a.device, trusted_only=False, state=a.state, level=a.level, mode=a.mode)
    json.dump(vars(a), open(os.path.join(P.ROOT, f'args_{a.tag}.json'), 'w'), indent=1)
    print(f'train {len(train)} slides ({sum(s.gt_pos > 0 for s in train)} tumour), val {len(val)}', flush=True)
    policy = Policy(d_emb=train[0].d_emb, n_act=train[0].n_act, n_dyn=train[0].n_dyn).to(a.device)
    if a.init:
        policy.load_state_dict(torch.load(a.init, map_location=a.device))
        agg0, _ = evaluate(policy, val, a.device)
        print(f"init from {a.init}: val dice {agg0['dice_all']:.3f} tumour {agg0['dice_tumour']:.3f} zoom {agg0['zoom_frac']:.2f}", flush=True)
    opt = torch.optim.AdamW(policy.parameters(), lr=a.lr, weight_decay=1e-4)
    log = open(os.path.join(P.ROOT, f'train_{a.tag}.jsonl'), 'a')
    best = -1
    tumour = [s for s in train if s.gt_pos > 0]
    normal = [s for s in train if s.gt_pos == 0]
    for u in range(a.updates):
        t0 = time.time()
        if a.balanced:
            picks = list(rng.choice(tumour, a.episodes // 2, replace=False)) + list(rng.choice(normal, a.episodes - a.episodes // 2, replace=False))
        else:
            picks = rng.choice(train, a.episodes, replace=False)
        episodes = []
        for s in picks:
            rec, env = run_episode(policy, s, a.device, zoom_cost=a.zoom_cost)
            episodes.append((s, rec))
        stats = update(policy, opt, episodes, a.device, ent=a.ent)
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
            torch.save(policy.state_dict(), os.path.join(P.CKPT, f'{a.tag}_u{u}.pt'))
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
