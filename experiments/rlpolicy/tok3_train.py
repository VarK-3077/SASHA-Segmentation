# Trains the token-level L3 head on the cached subset (tok3_data.py), leaving one fold of slides out (--fold k)
# or none (--fold -1, the head the test slides get). Batches are half normal units and half tumour units, the
# tumour half spread evenly over tiny (< 10 % tumour), partial and core (>= 90 %) units; random flips and
# quarter turns of the token grid. Loss: token BCE against the tumour share under each token, plus unit BCE
# for "holds tumour" and "is core". Fixed number of steps, last weights kept (no slide is used to pick a step).
# Usage: python -m experiments.rlpolicy.tok3_train --tag t0 --fold 0

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.rlpolicy import where as W
from experiments.rlpolicy.tok3_model import TokHead


def groups(frac):
    return [np.flatnonzero(frac == 0), np.flatnonzero((frac > 0) & (frac < 0.1)), np.flatnonzero((frac >= 0.1) & (frac < 0.9)),
            np.flatnonzero(frac >= 0.9)]


def batch_idx(rng, gs, bs, share=(0.5, 0.2, 0.2, 0.1)):
    return np.sort(np.concatenate([g[rng.integers(len(g), size=int(round(bs * s)))] for g, s in zip(gs, share) if len(g)]))


def augment(x, y, rng):
    k = int(rng.integers(4))
    x, y = torch.rot90(x, k, (2, 3)), torch.rot90(y, k, (1, 2))
    if rng.random() < 0.5:
        x, y = x.flip(3), y.flip(2)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--d', type=int, default=128)
    ap.add_argument('--ctx', type=int, default=3)
    ap.add_argument('--drop', type=float, default=0.2)
    ap.add_argument('--tok_pos', type=float, default=4.0, help='weight of tumour tokens in the token loss')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    out = f'{W.head_dir(a.tag)}/f{a.fold}'
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    meta = np.load(f'{W.SUB}/meta.npz')
    keep = np.flatnonzero(meta['fold'][meta['slide']] != a.fold)
    tok = np.load(f'{W.SUB}/tok.npy', mmap_mode='r')
    t0 = time.time()
    x_all = np.concatenate([tok[i:j] for i, j in runs(keep)])
    lab, frac = meta['lab'][keep], meta['frac'][keep]
    print(f'fold {a.fold}: {len(keep)} units loaded in {time.time() - t0:.0f} s', flush=True)
    gs = groups(frac)
    print('groups', [len(g) for g in gs], flush=True)
    net = TokHead(d=a.d, ctx=a.ctx, p_drop=a.drop).to(a.device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    log = []
    for step in range(1, a.steps + 1):
        i = batch_idx(rng, gs, a.bs)
        x = torch.from_numpy(x_all[i]).to(a.device).float()
        y = torch.from_numpy(lab[i]).to(a.device).float() / 16
        x, y = augment(x, y, rng)
        f = torch.from_numpy(frac[i]).to(a.device)
        t, u = net(x)
        w = 1 + (a.tok_pos - 1) * y
        l_tok = (F.binary_cross_entropy_with_logits(t, y, reduction='none') * w).mean()
        l_unit = F.binary_cross_entropy_with_logits(u[:, 0], (f > 0).float()) + F.binary_cross_entropy_with_logits(u[:, 1], (f >= 0.9).float())
        loss = l_tok + 0.5 * l_unit
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 200 == 0:
            log.append(dict(step=step, tok=float(l_tok), unit=float(l_unit)))
            print(f'{step} tok {float(l_tok):.4f} unit {float(l_unit):.4f} {time.time() - t0:.0f} s', flush=True)
    torch.save(net.state_dict(), f'{out}/head.pt')
    json.dump(dict(vars(a), log=log), open(f'{out}/run.json', 'w'), indent=1)
    open(f'{out}/DONE', 'w').close()


def runs(idx):
    """Consecutive runs of sorted indices as (start, stop) pairs, so a memmap is read in a few long slices."""
    cut = np.flatnonzero(np.diff(idx) != 1) + 1
    return [(int(s[0]), int(s[-1]) + 1) for s in np.split(idx, cut)]


if __name__ == '__main__':
    main()
