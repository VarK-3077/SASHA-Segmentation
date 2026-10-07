# Trains the probes of the separability study: for every (level, backbone) a linear
# classifier and a small MLP over the four buckets, slide-grouped on the repo's split,
# selected on val macro-recall, scored on test. Writes one json per combination with the
# confusion matrix, per-bucket recall and AUC, and the recall of each tumour bucket and
# lesion-size bin at fixed specificity on normals.

import argparse
import json
import os

import h5py
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from experiments.sep import buckets as B
from experiments.sep import paths as P
from experiments.sep.backbones import LOADERS

SPECS = (0.95, 0.99)
SEEDS = (0, 1, 2)


def load(level, backbone):
    z = np.load(P.crops_npz(level))
    with h5py.File(P.feats_h5(level), 'r') as f:
        X = f[backbone][:].astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    tr = z['split'] == 'train'
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    return (X - mu) / sd, z


def make(kind, d, h=512):
    if kind == 'linear':
        return torch.nn.Linear(d, 4)
    return torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.GELU(), torch.nn.Dropout(0.2),
                               torch.nn.Linear(h, 4))


def macro_recall(y, pred):
    return float(np.mean([(pred[y == c] == c).mean() for c in range(4) if (y == c).any()]))


def fit(kind, X, y, tr, va, device, seed, epochs=60, patience=8):
    torch.manual_seed(seed)
    net = make(kind, X.shape[1]).to(device)
    Xt, yt = torch.tensor(X[tr], device=device), torch.tensor(y[tr], device=device, dtype=torch.long)
    Xv = torch.tensor(X[va], device=device)
    w = 1.0 / np.maximum(np.bincount(y[tr], minlength=4), 1)
    w = torch.tensor(w / w.sum() * 4, device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt), device=device)
        for i in range(0, len(perm), 2048):
            j = perm[i:i + 2048]
            loss = torch.nn.functional.cross_entropy(net(Xt[j]), yt[j], weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            mr = macro_recall(y[va], net(Xv).argmax(1).cpu().numpy())
        if mr > best:
            best, bad = mr, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, best, ep + 1


def recall_at_spec(score, y, groups, spec):
    """score is the 'not normal' probability; threshold set on test normals."""
    thr = np.quantile(score[y == B.NORMAL], spec)
    hit = score >= thr
    out = {}
    for gname, g in groups.items():
        out[gname] = float(hit[g].mean()) if g.any() else None
    return thr, out


def evaluate(prob, y, z, te):
    y, prob = y[te], prob[te]
    slide, size_bin = z['slide'][te], z['size_bin'][te]
    tumour_slides = set(slide[y > 0])
    on_tumour = np.isin(slide, list(tumour_slides))
    pred = prob.argmax(1)
    cm = np.array([[int(((y == i) & (pred == j)).sum()) for j in range(4)] for i in range(4)])
    r = {'confusion': cm.tolist(),
         'recall': {B.NAMES[c]: float((pred[y == c] == c).mean()) for c in range(4) if (y == c).any()},
         'macro_recall': macro_recall(y, pred),
         'auc_tumour_slides': {}, 'at_spec': {}}
    for c in range(4):
        yy = (y[on_tumour] == c)
        if yy.any() and not yy.all():
            r['auc_tumour_slides'][B.NAMES[c]] = float(roc_auc_score(yy, prob[on_tumour, c]))
    attend = 1.0 - prob[:, B.NORMAL]
    groups = {B.NAMES[c]: y == c for c in (B.CORE, B.BOUNDARY, B.SMALL)}
    groups.update({f'size_{B.SIZE_NAMES[s]}': (y > 0) & (size_bin == s) for s in range(3)})
    for spec in SPECS:
        thr, rec = recall_at_spec(attend, y, groups, spec)
        r['at_spec'][str(spec)] = {'thr': float(thr), 'recall': rec}
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--levels', nargs='+', type=int, default=list(P.LEVELS))
    ap.add_argument('--backbones', nargs='+', default=list(LOADERS))
    a = ap.parse_args()
    for L in a.levels:
        for bb in a.backbones:
            out = os.path.join(P.PROBE_DIR, f'L{L}_{bb}.json')
            if os.path.exists(out):
                continue
            with h5py.File(P.feats_h5(L), 'r') as f:
                if not f.attrs.get(f'{bb}_done', False):
                    print(f'skip L{L} {bb}: no features'); continue
            X, z = load(L, bb)
            y = z['bucket'].astype(np.int64)
            tr, va, te = (z['split'] == s for s in ('train', 'val', 'test'))
            res = {'level': L, 'backbone': bb, 'n': {s: int(m.sum()) for s, m in zip(('train', 'val', 'test'), (tr, va, te))},
                   'counts_test': np.bincount(y[te], minlength=4).tolist(), 'models': {}}
            for kind in ('linear', 'mlp'):
                runs = []
                for seed in SEEDS:
                    net, val_mr, n_ep = fit(kind, X, y, tr, va, a.device, seed)
                    with torch.no_grad():
                        prob = torch.softmax(net(torch.tensor(X, device=a.device)), 1).cpu().numpy()
                    r = evaluate(prob, y, z, te)
                    r.update({'val_macro_recall': val_mr, 'epochs': n_ep, 'seed': seed})
                    runs.append(r)
                    if seed == SEEDS[0]:
                        np.savez_compressed(os.path.join(P.PROBE_DIR, f'L{L}_{bb}_{kind}_testprob.npz'),
                                            prob=prob[te].astype(np.float16), y=y[te], slide=z['slide'][te],
                                            x=z['x'][te], y_coord=z['y'][te], size_bin=z['size_bin'][te])
                res['models'][kind] = runs
                print(f'L{L} {bb} {kind}: macro-recall {np.mean([r["macro_recall"] for r in runs]):.3f}, '
                      f'small@99spec {np.mean([r["at_spec"]["0.99"]["recall"]["small"] or 0 for r in runs]):.3f}', flush=True)
            json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
