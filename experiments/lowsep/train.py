# Trains one screen head (level, context on/off, seed) on windows of the training slides' token
# maps, about 32 x 32 units each: half the windows are centred on a crop holding tumour, half on
# a random tissue crop; random 90-degree rotations and flips. Class-weighted cross-entropy over
# labelled tissue units. With --unit_drop, random units of each window are zeroed as if a
# selective screen had not read them, and only read units are scored. Every few hundred steps the whole val slides are scored and the
# checkpoint with the best val macro-recall over the four buckets is kept.

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from experiments.lowsep import paths as P
from experiments.lowsep.maps import IGNORE, SlideMaps, to_device
from experiments.lowsep.model import ScreenHead

WINDOW_UNITS = 32


def names_of(split, level):
    s = P.split_of()
    return sorted(n for n in s if s[n] == split and os.path.exists(P.label_npz(n)) and os.path.exists(f'{P.tok_dir(level)}/{n}.npy'))


class Windows(IterableDataset):
    def __init__(self, names, level, seed):
        self.names, self.level, self.seed = names, level, seed
        self.wc = max(1, WINDOW_UNITS // P.units_per_crop(level))

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, info.id if info else 0, int(time.time())])
        maps = [SlideMaps(n, self.level) for n in self.names]
        tumour = [(i, rc) for i, m in enumerate(maps) for rc in m.tumour_crops()]
        tissue = [(i, rc) for i, m in enumerate(maps) for rc in m.rc]
        while True:
            pool = tumour if rng.random() < 0.5 else tissue
            i, (r, c) = pool[rng.integers(len(pool))]
            r0 = r - self.wc // 2 + rng.integers(-self.wc // 4, self.wc // 4 + 1)
            c0 = c - self.wc // 2 + rng.integers(-self.wc // 4, self.wc // 4 + 1)
            tk, tg = maps[i].window(r0, c0, self.wc, self.wc)
            k = int(rng.integers(4))
            tk, tg = np.rot90(tk, k, (1, 2)), np.rot90(tg, k)
            if rng.random() < 0.5:
                tk, tg = tk[:, :, ::-1], tg[:, ::-1]
            yield torch.from_numpy(tk.copy()), torch.from_numpy(tg.copy())


def class_weights(names, level):
    counts = np.zeros(4)
    for n in names:
        t = SlideMaps(n, level).target
        counts += np.bincount(t[t >= 0], minlength=4)
    w = 1.0 / np.maximum(counts, 1)
    return torch.tensor(w / w.sum() * 4, dtype=torch.float32), counts


def drop_units(tk, keep, k):
    """Zero the token blocks of the units not kept: tk (B, C, H*k, W*k), keep (B, H, W) bool."""
    m = keep.repeat_interleave(k, 1).repeat_interleave(k, 2)[:, None]
    return tk * m.to(tk.dtype)


@torch.no_grad()
def predict(head, maps, device, keep=None):
    """Per-unit softmax over the whole slide, cropped to the slide's unit grid. With keep (hu, wu)
    bool, the tokens of every other unit are zeroed first (only those units were read)."""
    tk, _ = maps.full()
    tk = to_device(tk, device)[None]
    if keep is not None:
        hu, wu = keep.shape
        k = tk.shape[-2] // maps.target.shape[0]
        kp = torch.zeros(1, maps.target.shape[0], maps.target.shape[1], dtype=torch.bool, device=device)
        kp[0, :hu, :wu] = torch.as_tensor(keep, device=device)
        tk = drop_units(tk, kp, k)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits = head(tk)[0].float()
    hu, wu = maps.tissue.shape
    return torch.softmax(logits, 0)[:, :hu, :wu].cpu().numpy()


def macro_recall(head, val, device, drop=0.0):
    """Val macro-recall over the four buckets; with drop > 0, a fixed random share of units is
    zeroed (as unread) and only the remaining units are scored."""
    y, p = [], []
    rng = np.random.default_rng(0)
    for m in val:
        hu, wu = m.tissue.shape
        kept = rng.random((hu, wu)) >= drop if drop > 0 else None
        prob = predict(head, m, device, kept)
        t = m.target[:hu, :wu]
        keep = (t >= 0) if kept is None else (t >= 0) & kept
        y.append(t[keep]); p.append(prob.argmax(0)[keep])
    y, p = np.concatenate(y), np.concatenate(p)
    rec = [float((p[y == c] == c).mean()) for c in range(4) if (y == c).any()]
    return float(np.mean(rec)), rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--level', type=int, required=True)
    ap.add_argument('--ctx', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--eval_every', type=int, default=300)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--d', type=int, default=256, help='head width')
    ap.add_argument('--drop', type=float, default=0.1)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--name', default='', help='tag suffix for sweep runs')
    ap.add_argument('--unit_drop', type=float, default=0.0,
                    help='max share of units zeroed per training window (drawn uniformly in [0, max]); val zeroes half')
    a = ap.parse_args()
    tag = f'L{a.level}{P.backbone_tag()}_{"ctx" if a.ctx else "noctx"}{a.name}_s{a.seed}'
    out = P.run_dir(tag)
    os.makedirs(out, exist_ok=True)
    json.dump(vars(a), open(f'{out}/args.json', 'w'), indent=1)
    torch.manual_seed(a.seed)
    train, val = names_of('train', a.level), names_of('val', a.level)
    w, counts = class_weights(train, a.level)
    print(f'{tag}: {len(train)} train / {len(val)} val slides, train unit counts {counts.astype(int).tolist()}', flush=True)
    head = ScreenHead(P.tokens_per_unit(a.level), ctx=bool(a.ctx), d=a.d, p_drop=a.drop).to(a.device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=a.steps, pct_start=0.05)
    loader = iter(DataLoader(Windows(train, a.level, a.seed), batch_size=a.batch, num_workers=a.workers, persistent_workers=True))
    val_maps = [SlideMaps(n, a.level) for n in val]
    w = w.to(a.device)
    best, log, t0 = -1, open(f'{out}/train.jsonl', 'w'), time.time()
    for step in range(1, a.steps + 1):
        head.train()
        tk, tg = next(loader)
        tk, tg = tk.to(a.device).float(), tg.to(a.device)
        if a.unit_drop > 0:   # units left unread by a selective screen: zero their tokens, score only the read ones
            frac = torch.rand(tg.shape[0], 1, 1, device=a.device) * a.unit_drop
            keep = torch.rand(tg.shape, device=a.device) >= frac
            tk = drop_units(tk, keep, tk.shape[-1] // tg.shape[-1])
            tg = torch.where(keep, tg, torch.full_like(tg, IGNORE))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = head(tk)
        loss = F.cross_entropy(logits.float(), tg, weight=w, ignore_index=IGNORE)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % a.eval_every == 0 or step == a.steps:
            head.eval()
            mr, rec = macro_recall(head, val_maps, a.device, 0.5 if a.unit_drop > 0 else 0.0)
            row = {'step': step, 'loss': loss.item(), 'val_macro_recall': mr, 'val_recall': rec, 'sec': time.time() - t0}
            log.write(json.dumps(row) + '\n'); log.flush()
            print(f'{tag} step {step} loss {loss.item():.3f} val macro-recall {mr:.3f} {[round(r, 3) for r in rec]} {row["sec"]:.0f}s', flush=True)
            if mr > best:
                best = mr
                torch.save({'model': head.state_dict(), 'step': step, 'val_macro_recall': mr}, f'{out}/best.pt')
    print(f'{tag} done, best val macro-recall {best:.3f}', flush=True)


if __name__ == '__main__':
    main()
