# The crude baseline for the ablation: no L6 pre-screen, no token-level head, no painting, no border tracing,
# no hard-negative mining. Every tissue unit is read at L3, the frozen ViT-L's CLS token of that crop goes
# through a two-layer MLP ("holds tumour" or not), and the units it flags are segmented at L2.
#   extract  CLS tokens of every tissue unit of the trusted train slides and the test slides (real L3 reads)
#   train    five fold MLPs and an all-train MLP on all tissue units, sampled as they come, class-weighted BCE;
#            writes each slide's probability grid (train slides out of fold, test slides from the all-train MLP)
# Usage: python -m experiments.rlpolicy.cls3 extract [--workers 24] ; python -m experiments.rlpolicy.cls3 train

import argparse
import os
from multiprocessing import get_context

import numpy as np
import torch
import torch.nn as nn

from experiments.lowsep import paths as LP
from experiments.lowsep.cascade_pipe import read_screen
from experiments.pfa_scannet.data import slide_path
from experiments.rlpolicy import where as W
from experiments.rlpolicy.simtab import names_of

OUT = f'{W.ROOT}/cls3'


@torch.no_grad()
def cls_tokens(enc, imgs):
    """uint8 crops (B, 3, 224, 224) as read_screen returns them -> the encoder's CLS tokens (B, 1024) float32."""
    x = torch.from_numpy(np.stack(imgs)).to(enc.mean.device, non_blocking=True)
    x = (x.float() / 255.0 - enc.mean) / enc.std
    with torch.autocast('cuda', dtype=torch.bfloat16):
        h = enc.model.forward_features(x)
    return h[:, 0].float()


class Head(nn.Module):
    def __init__(self, mu, sd, d=256, p_drop=0.1):
        super().__init__()
        self.register_buffer('mu', mu)
        self.register_buffer('sd', sd)
        self.net = nn.Sequential(nn.Linear(mu.numel(), d), nn.ReLU(), nn.Dropout(p_drop), nn.Linear(d, 1))

    def forward(self, x):
        return self.net((x - self.mu) / self.sd)[:, 0]


def extract(a):
    from experiments.sep.backbones import Encoder
    os.makedirs(f'{OUT}/tok', exist_ok=True)
    pool = get_context('fork').Pool(a.workers)       # readers before CUDA
    enc = Encoder('kaiko_vitl14', a.device)
    names = names_of('train') + names_of('test')
    for i, name in enumerate(names):
        f = f'{OUT}/tok/{name}.npz'
        if os.path.exists(f):
            continue
        rc = np.argwhere(np.load(f'{W.UNITS}/{name}.npz')['tissue'])
        path = slide_path(name) if os.path.exists(slide_path(name)) else LP.slide_path(name)   # some normals live in the extra folder
        out = []
        for k in range(0, len(rc), 256):
            items = pool.map(read_screen, [(path, 3, int(r), int(c)) for r, c in rc[k:k + 256]], chunksize=8)
            out.append(cls_tokens(enc, [it[2] for it in items]).cpu().numpy().astype(np.float16))
        np.savez(f'{f}.tmp.npz', cls=np.concatenate(out) if out else np.zeros((0, 1024), np.float16), rc=rc)
        os.replace(f'{f}.tmp.npz', f)
        if i % 20 == 0:
            print(i, name, len(rc), flush=True)
    open(f'{OUT}/tok/DONE', 'w').close()


def fit(x, y, a, seed):
    torch.manual_seed(seed)
    x, y = x.to(a.device), y.to(a.device)
    net = Head(x.mean(0), x.std(0) + 1e-6).to(a.device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = a.epochs * (len(x) // a.bs + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.05)
    pos_w = (1 - y).sum() / y.sum().clamp(min=1)
    for _ in range(steps):
        i = torch.randint(len(x), (a.bs,), device=a.device)
        loss = nn.functional.binary_cross_entropy_with_logits(net(x[i]), y[i], pos_weight=pos_w)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return net.eval()


def train(a):
    os.makedirs(f'{OUT}/pred', exist_ok=True)
    folds = W.folds()
    tr, te = names_of('train'), names_of('test')
    data = {}
    for n in tr + te:
        z = np.load(f'{OUT}/tok/{n}.npz')
        frac = np.load(f'{W.UNITS}/{n}.npz')['frac']
        data[n] = (torch.from_numpy(z['cls'].astype(np.float32)), z['rc'], torch.from_numpy((frac[z['rc'][:, 0], z['rc'][:, 1]] > 0).astype(np.float32)), frac.shape)
    print('train units', sum(len(data[n][0]) for n in tr), 'tumour share', float(torch.cat([data[n][2] for n in tr]).mean()), flush=True)

    @torch.no_grad()
    def write(net, names):
        for n in names:
            x, rc, _, shape = data[n]
            g = np.zeros(shape, np.float32)
            if len(x):
                g[rc[:, 0], rc[:, 1]] = torch.sigmoid(net(x.to(a.device))).cpu().numpy()
            np.save(f'{OUT}/pred/{n}.npy', g)

    for k in list(range(W.N_FOLDS)) + [-1]:
        keep = [n for n in tr if folds[n] != k]
        net = fit(torch.cat([data[n][0] for n in keep]), torch.cat([data[n][2] for n in keep]), a, seed=k + 1)
        torch.save(net.state_dict(), f'{OUT}/f{k}.pt')
        write(net, [n for n in tr if folds[n] == k] if k >= 0 else te)
        print('fold', k, flush=True)
    open(f'{OUT}/pred/DONE', 'w').close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=('extract', 'train'))
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--bs', type=int, default=4096)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    (extract if a.step == 'extract' else train)(a)


if __name__ == '__main__':
    main()
