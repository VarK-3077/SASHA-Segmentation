# Trains a 4-class bucket classifier (normal / core / boundary / small) on the level-3 tile
# embeddings of the training slides and writes its probabilities per slide, so the
# agent can be given predicted buckets as state. Balanced CE, selected on
# val macro-recall, same recipe as the separability probes. Written next to the cache rather
# than into it, so a running extraction that reads the cache never sees a half-written file.

import os
import sys

import numpy as np
import torch

from experiments.rlbin import paths as P


def load_all():
    data = {}
    for f in sorted(os.listdir(P.CACHE)):
        c = np.load(os.path.join(P.CACHE, f))
        data[f[:-4]] = {'lr': c['lr'].astype(np.float32), 'bucket': c['bucket'].astype(np.int64),
                        'split': str(c['split']), 'trust': bool(c['trust_normals'])}
    return data


def macro_recall(y, pred):
    return float(np.mean([(pred[y == c] == c).mean() for c in range(4) if (y == c).any()]))


def main(device):
    torch.manual_seed(0)
    data = load_all()
    tr = [d for d in data.values() if d['split'] == 'train' and d['trust']]
    va = [d for d in data.values() if d['split'] == 'val']
    X = torch.tensor(np.concatenate([d['lr'] for d in tr]), device=device)
    y = torch.tensor(np.concatenate([d['bucket'] for d in tr]), device=device)
    Xv = torch.tensor(np.concatenate([d['lr'] for d in va]), device=device)
    yv = np.concatenate([d['bucket'] for d in va])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X, Xv = (X - mu) / sd, (Xv - mu) / sd
    w = 1.0 / np.maximum(np.bincount(y.cpu().numpy(), minlength=4), 1)
    w = torch.tensor(w / w.sum() * 4, device=device, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(384, 512), torch.nn.GELU(), torch.nn.Dropout(0.2), torch.nn.Linear(512, 4)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, bad = -1, None, 0
    for ep in range(60):
        net.train()
        perm = torch.randperm(len(X), device=device)
        for i in range(0, len(perm), 2048):
            j = perm[i:i + 2048]
            loss = torch.nn.functional.cross_entropy(net(X[j]), y[j], weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            mr = macro_recall(yv, net(Xv).argmax(1).cpu().numpy())
        if mr > best:
            best, bad, best_state = mr, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 8:
                break
    net.load_state_dict(best_state)
    net.eval()
    print(f'bucket head: val macro-recall {best:.3f} after {ep + 1} epochs', flush=True)
    os.makedirs(P.PBUCKET, exist_ok=True)
    with torch.no_grad():
        for name, d in data.items():
            p = torch.softmax(net((torch.tensor(d['lr'], device=device) - mu) / sd), 1).cpu().numpy().astype(np.float16)
            np.save(os.path.join(P.PBUCKET, name + '.npy'), p)
    te = [d for d in data.values() if d['split'] == 'test']
    with torch.no_grad():
        pred = net((torch.tensor(np.concatenate([d['lr'] for d in te]), device=device) - mu) / sd).argmax(1).cpu().numpy()
    yt = np.concatenate([d['bucket'] for d in te])
    print('test per-bucket recall:', {b: round(float((pred[yt == b] == b).mean()), 3) for b in range(4)}, flush=True)
    print(f'p_bucket written per slide under {P.PBUCKET}', flush=True)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'cuda:0')
