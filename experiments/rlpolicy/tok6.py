# A second L6 screen head for the consensus policy: token-level, with context. It reads the whole-slide level-6
# token map the cascade computes anyway (2 x 2 tokens per unit), mixes neighbouring tokens with two 3 x 3
# residual blocks and gives one tumour logit per token, trained against the tumour share under each token (the
# scorer's level-5 ground truth). Trained on random 48 x 48 token windows of the trusted train slides, half of
# them centred on a tumour token, one fold of slides left out (five heads) or none (the all-train head); then
# every slide is scored by the head that did not see it and the per-unit maximum / mean are saved as grids.
# Usage: python -m experiments.rlpolicy.tok6 --tag g0 [--steps 3000]

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.lowsep.maps import SlideMaps
from experiments.rlpolicy import where as W
from experiments.rlpolicy.simtab import names_of
from experiments.rlpolicy.tok3_data import gt_mask

WIN, T5 = 48, 32          # window side in tokens; level-5 cells per level-6 token side


class Tok6(nn.Module):
    def __init__(self, d=64, p_drop=0.3):
        super().__init__()
        self.proj = nn.Conv2d(256, d, 1)
        self.b1, self.b2 = nn.Conv2d(d, d, 3, padding=1), nn.Conv2d(d, d, 3, padding=1)
        self.drop = nn.Dropout2d(p_drop)
        self.out = nn.Conv2d(d, 1, 1)

    def forward(self, x):
        h = F.gelu(self.proj(x))
        h = F.gelu(h + self.b1(h))
        h = F.gelu(h + self.b2(h))
        return self.out(self.drop(h))[:, 0]


def load_slide(name):
    """Token map (256, H, W) float16, token labels (H, W) float32 (tumour share) and the tissue-token mask."""
    m = SlideMaps(name, 6)
    tk = m.full()[0]
    h, w = tk.shape[1:]
    lab = np.zeros((h, w), np.float32)
    g = gt_mask(name)
    if g is not None:
        g = (g > 0)[:h * T5, :w * T5]
        p = np.zeros((h * T5, w * T5), np.float32)
        p[:g.shape[0], :g.shape[1]] = g
        lab = p.reshape(h, T5, w, T5).mean((1, 3))
    tis = np.zeros((h, w), bool)
    tu = np.repeat(np.repeat(m.tissue, 2, 0), 2, 1)
    tis[:tu.shape[0], :tu.shape[1]] = tu
    return tk, lab, tis, m.tissue.shape


def window(rng, slides, pos, tis):
    k = rng.integers(len(slides))
    tk, lab, t, _ = slides[k]
    pool = pos[k] if len(pos[k]) and rng.random() < 0.5 else tis[k]
    r, c = pool[rng.integers(len(pool))] - WIN // 2 + rng.integers(-WIN // 4, WIN // 4 + 1, 2)
    x, y, m = np.zeros((256, WIN, WIN), np.float16), np.zeros((WIN, WIN), np.float32), np.zeros((WIN, WIN), bool)
    r0, c0, r1, c1 = max(r, 0), max(c, 0), min(r + WIN, tk.shape[1]), min(c + WIN, tk.shape[2])
    if r1 > r0 and c1 > c0:
        x[:, r0 - r:r1 - r, c0 - c:c1 - c] = tk[:, r0:r1, c0:c1]
        y[r0 - r:r1 - r, c0 - c:c1 - c], m[r0 - r:r1 - r, c0 - c:c1 - c] = lab[r0:r1, c0:c1], t[r0:r1, c0:c1]
    k4 = int(rng.integers(4))
    x, y, m = np.rot90(x, k4, (1, 2)), np.rot90(y, k4), np.rot90(m, k4)
    if rng.random() < 0.5:
        x, y, m = x[:, :, ::-1], y[:, ::-1], m[:, ::-1]
    return x.copy(), y.copy(), m.copy()


def train(slides, a, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    pos = [np.argwhere(s[1] > 0) for s in slides]
    tis = [np.argwhere(s[2]) for s in slides]
    net = Tok6(a.d, a.drop).to(a.device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    for step in range(1, a.steps + 1):
        b = [window(rng, slides, pos, tis) for _ in range(a.bs)]
        x, y, m = (torch.from_numpy(np.stack(v)).to(a.device) for v in zip(*b))
        loss = F.binary_cross_entropy_with_logits(net(x.float()), y, reduction='none') * (1 + (a.pos - 1) * y)
        loss = (loss * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0:
            print(f'  step {step} loss {float(loss):.4f}', flush=True)
    return net.eval()


@torch.no_grad()
def unit_scores(net, slide, device):
    tk, _, _, (hu, wu) = slide
    p = torch.sigmoid(net(torch.from_numpy(tk[None]).to(device).float()))[0]
    p = F.pad(p, (0, max(0, 2 * wu - p.shape[1]), 0, max(0, 2 * hu - p.shape[0])))[:2 * hu, :2 * wu].reshape(hu, 2, wu, 2)
    return p.amax((1, 3)).cpu().numpy(), p.mean((1, 3)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--drop', type=float, default=0.3)
    ap.add_argument('--pos', type=float, default=8.0)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    out = f'{W.ROOT}/tok6/{a.tag}'
    os.makedirs(f'{out}/pred', exist_ok=True)
    json.dump(vars(a), open(f'{out}/run.json', 'w'))
    folds = W.folds()
    tr, te = names_of('train'), names_of('test')
    data = {n: load_slide(n) for n in tr}
    print(len(data), 'train slides loaded', flush=True)
    for k in list(range(W.N_FOLDS)) + [-1]:
        print('fold', k, flush=True)
        net = train([data[n] for n in tr if folds[n] != k], a, k + 10)
        torch.save(net.state_dict(), f'{out}/f{k}.pt')
        for n in ([x for x in tr if folds[x] == k] if k >= 0 else te):
            mx, mean = unit_scores(net, data[n] if k >= 0 else load_slide(n), a.device)
            np.savez(f'{out}/pred/{n}.npz', g_max=mx, g_mean=mean)
    open(f'{out}/DONE', 'w').close()


if __name__ == '__main__':
    main()
