# Builds the level-2 per-slide cache for the binning agent: four 1024 px children per level-3
# tile, each with its Kaiko ViT-L embedding, its 4 high-res sub-patches (the same ViT-S
# high-res scorer probabilities as the level-3 cache, re-grouped), the sub-patch ground truth,
# the level-2 bucket, and low-res score probabilities from a freshly trained level-2 head.
#
# Index bookkeeping, verified by assertions on the ground truth: level-3 cache sub-patches are
# in step12 order k = xi*4 + yi; level-2 child c = iy*2 + ix (extract_l2.py) holds the
# sub-patches with xi in {2ix, 2ix+1}, yi in {2iy, 2iy+1}; the sep level-2 labels list the
# children of a tile in the same c order.

import os
import sys

import numpy as np
import torch

from experiments.rlbin import paths as P
from experiments.rlbin.cache import neighbour_mean
from experiments.rlbin.extract_l2 import L2_DIR
from experiments.sep import paths as SP

CACHE_L2 = f'{P.ROOT}/cache_l2'
HEAD_L2 = f'{P.ROOT}/lr_head_l2.pt'

# for child c, the 4 level-3 sub-patch indices (x-major order within the child)
CHILD_SUBS = np.array([[(2 * ix + a) * 4 + (2 * iy + b) for a in (0, 1) for b in (0, 1)]
                       for iy in (0, 1) for ix in (0, 1)])


def regroup(arr16):
    """(N, 16, ...) level-3 sub-patch array -> (N*4, 4, ...) per level-2 child."""
    return arr16[:, CHILD_SUBS].reshape(-1, 4, *arr16.shape[2:])


def load_raw(name):
    c = np.load(os.path.join(P.CACHE, name + '.npz'))
    z = np.load(os.path.join(SP.LABEL_DIR, name + '.npz'))
    f = np.load(os.path.join(L2_DIR, name + '.npz'))
    n3 = len(c['coords'])
    gt = regroup(c['gt'])
    p_hr = regroup(c['p_hr'])
    l2_frac = z['L2_frac'].astype(np.float32).reshape(n3, 4).reshape(-1)
    l1_frac = z['L1_frac'].astype(np.float32).reshape(n3, 16)
    # sep L1 index = (k % 4) * 4 + k // 4 for step12 index k; the child's 4 sub-patches must average to the child's L2 fraction
    sub_frac_hr_order = l1_frac[:, [(k % 4) * 4 + k // 4 for k in range(16)]]
    assert np.allclose(regroup(sub_frac_hr_order).mean(1), l2_frac, atol=1e-2), name
    coords = (c['coords'][:, None, :] + np.array([[0, 0], [1024, 0], [0, 1024], [1024, 1024]])[None]).reshape(-1, 2)
    assert np.array_equal(coords, np.stack([z['L2_x'], z['L2_y']], 1)), name
    return {'lr': f['feat'], 'gt': gt, 'p_hr': p_hr.astype(np.float16), 'bucket': z['L2_bucket'], 'coords': coords,
            'split': str(c['split']), 'trust_normals': bool(c['trust_normals'])}


def train_lr_head(raws, device):
    """Kaiko ViT-L 1024 px embedding -> 4 sub-patch tumour logits. Early stop on val BCE."""
    tr = [r for r in raws.values() if r['split'] == 'train' and r['trust_normals']]
    va = [r for r in raws.values() if r['split'] == 'val']
    X = torch.tensor(np.concatenate([r['lr'] for r in tr]).astype(np.float32), device=device)
    y = torch.tensor(np.concatenate([r['gt'] for r in tr]).astype(np.float32), device=device)
    Xv = torch.tensor(np.concatenate([r['lr'] for r in va]).astype(np.float32), device=device)
    yv = torch.tensor(np.concatenate([r['gt'] for r in va]).astype(np.float32), device=device)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X, Xv = (X - mu) / sd, (Xv - mu) / sd
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(1024, 512), torch.nn.GELU(), torch.nn.Dropout(0.2), torch.nn.Linear(512, 4)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, bad = 1e9, None, 0
    for ep in range(30):
        net.train()
        perm = torch.randperm(len(X), device=device)
        for i in range(0, len(perm), 4096):
            j = perm[i:i + 4096]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(X[j]), y[j])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = torch.nn.functional.binary_cross_entropy_with_logits(net(Xv), yv).item()
        print(f'  lr head epoch {ep} val bce {vl:.4f}', flush=True)
        if vl < best:
            best, bad, best_state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 3:
                break
    net.load_state_dict(best_state)
    torch.save({'state': best_state, 'mu': mu.cpu(), 'sd': sd.cpu(), 'val_bce': best}, HEAD_L2)
    return net.eval(), mu, sd


def main(device):
    os.makedirs(CACHE_L2, exist_ok=True)
    names = sorted(f[:-4] for f in os.listdir(L2_DIR))
    raws = {n: load_raw(n) for n in names}
    print(f'{len(raws)} slides aligned', flush=True)
    net, mu, sd = train_lr_head(raws, device)
    with torch.no_grad():
        for n, r in raws.items():
            x = (torch.tensor(r['lr'].astype(np.float32), device=device) - mu) / sd
            p_lr = torch.sigmoid(net(x)).cpu().numpy()
            np.savez(os.path.join(CACHE_L2, n + '.npz'), lr=r['lr'], p_lr=p_lr.astype(np.float16), p_hr=r['p_hr'],
                     gt=r['gt'], bucket=r['bucket'], coords=r['coords'], nbr_p=neighbour_mean(p_lr.mean(1), r['coords'], step=1024),
                     split=r['split'], trust_normals=r['trust_normals'])
    print('cache_l2 done', flush=True)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'cuda:1')
