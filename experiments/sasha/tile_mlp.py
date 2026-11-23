"""Supervised slide classification without RL or high-res reads: a tile classifier on SASHA's own low-res (level 3)
ViT-S features, trained on tile labels (tile holds tumour), and a pooled slide score.
  linear     CLS feature (384) -> logistic regression, slide score = max over tiles
  mlp        CLS feature (384) -> 256 -> 1 MLP,       slide score = max over tiles
  juiced     mean + max of the patch tokens (768) -> 256 -> 1 MLP, slide score = mean of the top-k tile scores
Train slides of SASHA's split (split_4) train the tiles (normal tiles of non-exhaustively annotated slides left out);
val slides pick the epoch (slide AUC), k, and the threshold (largest validation Accuracy + F1, SASHA's own selection
criterion). Test is not read here. Each model is saved as a bundle the timed pipeline loads (time_tilemlp).
Usage: python -m experiments.sasha.tile_mlp --kind juiced --tokens DIR --out DIR"""

import argparse
import json
import os
import pickle
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

LR_H5 = '/data2/tej/SASHA/camelyon_lr_features/patch_feats_pretrain_medical_ssl.h5'
LABELS = os.path.expanduser('~/seg_outputs/patch_cls/labels.pkl')
SPLIT = 'dataset_csv/camelyon16/splits/split_4.json'
KS = (1, 2, 4, 8, 16, 32)


class TileNet(nn.Module):
    def __init__(self, mu, sd, hidden):
        super().__init__()
        self.register_buffer('mu', mu)
        self.register_buffer('sd', sd)
        d = mu.numel()
        self.net = nn.Linear(d, 1) if hidden == 0 else nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net((x - self.mu) / self.sd)[:, 0]


def pool(p, agg, k):
    """Tile probabilities (N,) -> slide score."""
    if agg == 'max' or len(p) == 0:
        return float(p.max()) if len(p) else 0.0
    return float(np.sort(p)[::-1][:k].mean())


def pick_threshold(y, s):
    """Threshold with the largest Accuracy + F1 on (y, s); ties go to the one nearest 0.5."""
    u = np.unique(s)
    cand = np.r_[(u[:-1] + u[1:]) / 2, u[0] - 1e-6, u[-1] + 1e-6] if len(u) > 1 else np.array([0.5])
    score = [accuracy_score(y, s >= t) + f1_score(y, s >= t, zero_division=0) for t in cand]
    best = np.max(score)
    return float(min((t for t, v in zip(cand, score) if v == best), key=lambda t: abs(t - 0.5)))


def load_split():
    sp = json.load(open(SPLIT))
    return sp['train_names'], sp['val_names'], sp['test_names']


def load_data(names, kind, tokens):
    labels = pickle.load(open(LABELS, 'rb'))
    out = {}
    with h5py.File(LR_H5, 'r') as f:
        for n in names:
            if n not in f:
                continue
            x = f[n]['feat'][:] if kind != 'juiced' else np.load(f'{tokens}/{n}.npz')['meanmax']
            y = np.asarray(labels[n]['tile_frac']) > 0
            assert len(x) == len(y), (n, len(x), len(y))
            out[n] = (x.astype(np.float32), y, int(labels[n]['slide_label']))
    return out


def slide_scores(net, data, agg, k, device):
    with torch.no_grad():
        return {n: pool(torch.sigmoid(net(torch.from_numpy(x).to(device))).cpu().numpy(), agg, k) for n, (x, _, _) in data.items()}


def train(a):
    from experiments.patch_common.ncrf_non_exhaustive import NON_EXHAUSTIVE_TUMOR_SLIDES
    tr_n, va_n, _ = load_split()
    tr, va = load_data(tr_n, a.kind, a.tokens), load_data(va_n, a.kind, a.tokens)
    keep = [(x, y) if n not in NON_EXHAUSTIVE_TUMOR_SLIDES else (x[y], y[y]) for n, (x, y, _) in tr.items()]
    X = torch.from_numpy(np.concatenate([k[0] for k in keep])).to(a.device)
    Y = torch.from_numpy(np.concatenate([k[1] for k in keep]).astype(np.float32)).to(a.device)
    torch.manual_seed(a.seed)
    net = TileNet(X.mean(0), X.std(0) + 1e-6, 0 if a.kind == 'linear' else 256).to(a.device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    pos_w = (1 - Y).sum() / Y.sum()
    agg = 'max' if a.kind != 'juiced' else 'topk'
    yv = np.array([va[n][2] for n in sorted(va)])
    best, log = None, []
    for ep in range(1, a.epochs + 1):
        net.train()
        perm = torch.randperm(len(X), device=a.device)
        for i in range(0, len(X), a.bs):
            j = perm[i:i + a.bs]
            loss = nn.functional.binary_cross_entropy_with_logits(net(X[j]), Y[j], pos_weight=pos_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        for k in (KS if agg == 'topk' else (1,)):
            s = slide_scores(net, va, agg, k, a.device)
            auc = roc_auc_score(yv, [s[n] for n in sorted(va)])
            log.append(dict(epoch=ep, k=k, val_auc=auc))
            if best is None or auc > best[0] + 1e-9:
                best = (auc, ep, k, {n: v.detach().clone() for n, v in net.state_dict().items()})
        print(f'{a.kind} epoch {ep}: val slide AUC ' + ' '.join(f"k{r['k']}={r['val_auc']:.4f}" for r in log if r['epoch'] == ep), flush=True)
    auc, ep, k, sd = best
    net.load_state_dict(sd)
    s = slide_scores(net, va, agg, k, a.device)
    thr = pick_threshold(yv, np.array([s[n] for n in sorted(va)]))
    os.makedirs(a.out, exist_ok=True)
    torch.save(net.state_dict(), f'{a.out}/model.pt')
    cfg = dict(kind=a.kind, feat='meanmax' if a.kind == 'juiced' else 'cls', hidden=0 if a.kind == 'linear' else 256, agg=agg, k=k, thr=thr,
               val_auc=auc, epoch=ep, n_train_tiles=int(len(X)), n_pos=int(Y.sum().item()), seed=a.seed, log=log)
    json.dump(cfg, open(f'{a.out}/cfg.json', 'w'), indent=1)
    print(f'{a.kind}: picked epoch {ep}, k {k}, val AUC {auc:.4f}, threshold {thr:.4f}', flush=True)


class Bundle:
    def __init__(self, d, device):
        self.cfg = json.load(open(f'{d}/cfg.json'))
        sd = torch.load(f'{d}/model.pt', map_location=device)
        self.net = TileNet(sd['mu'], sd['sd'], self.cfg['hidden']).to(device).eval()
        self.net.load_state_dict(sd)

    @torch.no_grad()
    def slide_prob(self, x):
        p = torch.sigmoid(self.net(x.float())).cpu().numpy()
        return pool(p, self.cfg['agg'], self.cfg['k'])


def time_tilemlp(conf, names, out_dir, now, drop_page_cache, warm_up):
    """time_infer.py --mode tilemlp: patching, the low-res pass of every tile, the tile classifier and pooling."""
    from experiments.sasha.lr_tokens import timed_meanmax
    from utils.inference_utils import Helper
    helper = Helper(conf, None)
    b = Bundle(conf.mlp_bundle, conf.device)
    warm_up(helper.feature_extractor, conf.device, (conf.batch_size_lr,))
    for slide in names:
        path = os.path.join(out_dir, slide + '.json')
        if os.path.exists(path):
            continue
        drop_page_cache(os.path.join(conf.source, slide + '.tif'))
        t0 = now()
        helper.create_patches(slide, conf.patch_level_low_res, conf.step_size, conf.patch_size)
        t1 = now()
        if b.cfg['feat'] == 'cls':
            feature, coords, _ = helper.extract_lr_features(slide, 0, conf.slide_ext, conf.patch_level_low_res, conf.patch_level_high_res)
            x = torch.from_numpy(feature).to(conf.device)
        else:
            x, coords = timed_meanmax(helper, slide, conf)
        t2 = now()
        prob = b.slide_prob(x)
        t3 = now()
        times = dict(patching=t1 - t0, lr_features=t2 - t1, classify=t3 - t2, total_wall=t3 - t0)
        json.dump(dict(load1=os.getloadavg()[0], slide=slide, n_tiles=len(coords), n_visited=0, prob_tumour=prob, threshold=b.cfg['thr'],
                       kind=b.cfg['kind'], times=times), open(path, 'w'), indent=1)
        print(f"{slide}: {len(coords)} tiles, p {prob:.3f}, {times['total_wall']:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', choices=('linear', 'mlp', 'juiced'), required=True)
    ap.add_argument('--tokens', default=None, help='juiced: directory written by lr_tokens.py')
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--bs', type=int, default=2048)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    train(ap.parse_args())


if __name__ == '__main__':
    main()
