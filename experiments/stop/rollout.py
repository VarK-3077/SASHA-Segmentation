# Plays every visit order on every slide and records the Dice curve after each step, so the
# stopping analysis can read off, per slide, where a perfect stopper would have quit. Also
# builds the tumour-fraction variants of each tumour slide by keeping the lesions and only the
# nearest normal tiles, then plays the same orders on those tile subsets.

import argparse
import os
from collections import deque

import numpy as np
import torch

from experiments.stop import data as D
from experiments.stop import paths as P
from experiments.stop.heads import mlp


def load_heads():
    d = torch.load(P.HEADS, map_location='cpu', weights_only=False)
    lr_head, hr_head = mlp(384, 16), mlp(768, 1)
    lr_head.load_state_dict(d['lr_head']); hr_head.load_state_dict(d['hr_head'])
    return lr_head.eval(), hr_head.eval()


@torch.no_grad()
def predict(heads, s, device):
    lr_head, hr_head = heads
    lr = torch.tensor(s['lr'], device=device)
    hr = torch.tensor(s['hr'], device=device).float()
    p_lr = torch.sigmoid(lr_head(lr))
    x = torch.cat([hr.reshape(-1, 384), lr.repeat_interleave(16, 0)], 1)
    p_hr = torch.sigmoid(hr_head(x)).reshape(-1, 16)
    return p_lr.cpu().numpy(), p_hr.cpu().numpy()


def entropy(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p)).mean(1)


def grid_index(coords):
    g = coords // P.TILE_L0
    return {(int(x), int(y)): i for i, (x, y) in enumerate(g)}, g


def neighbours(gi, g, i):
    x, y = int(g[i, 0]), int(g[i, 1])
    return [gi[(x + dx, y + dy)] for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            if (dx or dy) and (x + dx, y + dy) in gi]


def adaptive_order(ent, pbar, hr_hit, coords):
    """Most-uncertain-first, but after a visit that reveals tumour, its unvisited
    neighbours are taken next (highest low-res tumour score first)."""
    gi, g = grid_index(coords)
    n = len(ent)
    base = np.argsort(-ent)
    visited = np.zeros(n, bool)
    queue, order, ptr = [], [], 0
    for _ in range(n):
        queue = [q for q in queue if not visited[q]]
        if queue:
            pick = max(queue, key=lambda q: pbar[q])
        else:
            while visited[base[ptr]]:
                ptr += 1
            pick = base[ptr]
        visited[pick] = True
        order.append(pick)
        if hr_hit[pick]:
            queue.extend(q for q in neighbours(gi, g, pick) if not visited[q])
    return np.array(order)


def orders_for(p_lr, p_hr, tile_frac, coords, rng):
    ent = entropy(p_lr)
    pbar = p_lr.mean(1)
    hr_hit = (p_hr >= 0.5).any(1)
    return {
        'random': rng.permutation(len(ent)),
        'greedy': np.argsort(-pbar),
        'uncertain': np.argsort(-ent),
        'adaptive': adaptive_order(ent, pbar, hr_hit, coords),
        'oracle': np.lexsort((-ent, -tile_frac)),
    }, ent


def curves(order, pred_lr, pred_hr, gt):
    """Dice and tumour coverage after k visits, k = 0..N. O(N) via cumulative deltas."""
    inter_lr, inter_hr = (pred_lr & gt).sum(1), (pred_hr & gt).sum(1)
    pos_lr, pos_hr = pred_lr.sum(1), pred_hr.sum(1)
    gt_pos = gt.sum()
    inter = inter_lr.sum() + np.concatenate([[0], np.cumsum((inter_hr - inter_lr)[order])])
    pos = pos_lr.sum() + np.concatenate([[0], np.cumsum((pos_hr - pos_lr)[order])])
    dice = 2.0 * inter / np.maximum(pos + gt_pos, 1)
    cov = np.concatenate([[0], np.cumsum(gt.sum(1)[order])]) / max(gt_pos, 1)
    return dice.astype(np.float32), cov.astype(np.float32)


def variants(tile_frac, coords):
    """Tile subsets at the target tumour fractions: all tumour tiles plus the nearest normals."""
    tum = np.flatnonzero(tile_frac > 0)
    out = {'real': np.arange(len(tile_frac))}
    if len(tum) == 0:
        return out
    gi, g = grid_index(coords)
    dist = np.full(len(tile_frac), np.inf)
    dq = deque(tum.tolist())
    dist[tum] = 0
    while dq:
        i = dq.popleft()
        for j in neighbours(gi, g, i):
            if dist[j] == np.inf:
                dist[j] = dist[i] + 1
                dq.append(j)
    normals = np.flatnonzero(tile_frac == 0)
    normals = normals[np.argsort(dist[normals], kind='stable')]
    for f in P.TARGET_FRACS:
        n_norm = int(round(len(tum) * (1 - f) / f))
        if 1 <= n_norm < len(normals):
            out[f'f{f}'] = np.sort(np.concatenate([tum, normals[:n_norm]]))
    return out


def play(s, p_lr, p_hr, rng):
    gt = s['sub_frac'] >= P.SUB_POS_THR
    pred_lr, pred_hr = p_lr >= 0.5, p_hr >= 0.5
    out = {}
    for vname, keep in variants(s['tile_frac'], s['coords']).items():
        ords, ent = orders_for(p_lr[keep], p_hr[keep], s['tile_frac'][keep], s['coords'][keep], rng)
        out[f'{vname}__n'] = len(keep)
        out[f'{vname}__n_tumour'] = int((s['tile_frac'][keep] > 0).sum())
        for oname, order in ords.items():
            dice, cov = curves(order, pred_lr[keep], pred_hr[keep], gt[keep])
            out[f'{vname}__{oname}__dice'] = dice
            out[f'{vname}__{oname}__cov'] = cov
            out[f'{vname}__{oname}__ent'] = ent[order].astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    a = ap.parse_args()
    heads = load_heads()
    heads = tuple(h.to(a.device) for h in heads)
    split = D.split_of()
    rng = np.random.default_rng(0)
    for sp in a.splits:
        os.makedirs(os.path.join(P.ROLL_DIR, sp), exist_ok=True)
        names = sorted(n for n, v in split.items() if v == sp and D.has_data(n))
        for i, n in enumerate(names):
            out = os.path.join(P.ROLL_DIR, sp, n + '.npz')
            if os.path.exists(out):
                continue
            s = D.load_slide(n)
            p_lr, p_hr = predict(heads, s, a.device)
            np.savez_compressed(out, **play(s, p_lr, p_hr, rng))
            if i % 25 == 0:
                print(f'{sp} {i}/{len(names)}', flush=True)
    print('rollouts done', flush=True)


if __name__ == '__main__':
    main()
