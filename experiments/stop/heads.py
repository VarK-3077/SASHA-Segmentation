# Trains the two fresh scorers the rollouts use: a low-res head that predicts all 16
# sub-patch labels of a tile from the tile embedding alone, and a high-res head that
# predicts one sub-patch label from the sub-patch embedding plus its tile embedding.
# Plain BCE so the 0.5 threshold is meaningful; early stop on val BCE.

import time

import numpy as np
import torch

from experiments.stop import data as D
from experiments.stop import paths as P

DEVICE = 'cuda:1'


def mlp(d_in, d_out, h=512):
    return torch.nn.Sequential(torch.nn.Linear(d_in, h), torch.nn.GELU(), torch.nn.Dropout(0.2),
                               torch.nn.Linear(h, d_out))


def gather(names):
    lr, hr, y = [], [], []
    for n in names:
        if not D.has_data(n):
            continue
        s = D.load_slide(n)
        keep = np.ones(len(s['lr']), bool) if s['trust_normals'] else s['tile_frac'] > 0
        lr.append(s['lr'][keep].astype(np.float16))
        hr.append(s['hr'][keep])
        y.append((s['sub_frac'][keep] >= P.SUB_POS_THR).astype(np.float16))
    return np.concatenate(lr), np.concatenate(hr), np.concatenate(y)


def fit(net, X_fn, y, n, epochs, device, seed=0, batch=4096, patience=2):
    """X_fn(idx) builds a float batch; y is (n, d_out) on device."""
    torch.manual_seed(seed)
    net = net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    tr, va = n['train'], n['val']
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(tr), device=device)
        for i in range(0, len(perm), batch):
            j = tr[perm[i:i + batch]]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(X_fn(j)), y[j])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = np.mean([torch.nn.functional.binary_cross_entropy_with_logits(net(X_fn(va[i:i + 16384])), y[va[i:i + 16384]]).item()
                          for i in range(0, len(va), 16384)])
        print(f'  epoch {ep} val bce {vl:.4f}', flush=True)
        if vl < best:
            best, bad, best_state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return net.eval().cpu(), best


def main():
    split = D.split_of()
    names = {k: sorted(s for s, v in split.items() if v == k) for k in ('train', 'val')}
    t0 = time.time()
    parts = {k: gather(v) for k, v in names.items()}
    print(f'gathered in {time.time() - t0:.0f}s: ' + ', '.join(f'{k} {len(v[0])} tiles' for k, v in parts.items()), flush=True)
    dev = DEVICE
    lr = torch.tensor(np.concatenate([parts['train'][0], parts['val'][0]]), device=dev)
    hr = torch.tensor(np.concatenate([parts['train'][1], parts['val'][1]]), device=dev)
    y = torch.tensor(np.concatenate([parts['train'][2], parts['val'][2]]), device=dev)
    n_tr = len(parts['train'][0])
    idx = {'train': torch.arange(n_tr, device=dev), 'val': torch.arange(n_tr, len(lr), device=dev)}

    print('low-res head', flush=True)
    lr_head, lr_bce = fit(mlp(384, 16), lambda j: lr[j].float(), y.float(), idx, 30, dev)

    # high-res head: one row per sub-patch, tile embedding appended
    n_sub = len(lr) * 16
    y_sub = y.reshape(-1, 1).float()
    idx_sub = {k: (v[:, None] * 16 + torch.arange(16, device=dev)[None]).reshape(-1) for k, v in idx.items()}

    def x_sub(j):
        return torch.cat([hr.reshape(n_sub, 384)[j].float(), lr[j // 16].float()], 1)

    print('high-res head', flush=True)
    hr_head, hr_bce = fit(mlp(768, 1), x_sub, y_sub, idx_sub, 12, dev)
    torch.save({'lr_head': lr_head.state_dict(), 'hr_head': hr_head.state_dict(),
                'val_bce': {'lr': lr_bce, 'hr': hr_bce}}, P.HEADS)
    print(f'saved {P.HEADS}: val bce lr {lr_bce:.4f} hr {hr_bce:.4f}', flush=True)


if __name__ == '__main__':
    main()
