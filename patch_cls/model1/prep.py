# Builds the flat sub-patch arrays Model-1 trains on.
#
# Reads the per-slide ViT-S feature dumps (N,16,384) plus the per-sub-patch tumour
# area fractions, applies the label policy (positive >= 0.5, negative == 0 and not
# from a non-exhaustively annotated tumour slide, boundary dropped) and writes flat
# .npy arrays to a cache dir so the model scripts never touch the .pt files again.
#
# Val is written unfiltered (every sub-patch plus its sub_frac) so both the clean and
# the all-sub-patch validation CE can be derived from one array.

import argparse
import json
import os
import pickle
import time

import numpy as np
import torch

HOME = os.path.expanduser("~")
FEAT_TRAIN = f"{HOME}/seg_outputs/hr_train_feats"
FEAT_TEST = f"{HOME}/seg_outputs/hr_test_feats"
LABELS = f"{HOME}/seg_outputs/patch_cls/labels.pkl"
SPLIT = f"{HOME}/SASHA-Segmentation/dataset_csv/camelyon16/splits/split_4.json"
CACHE = f"{HOME}/seg_outputs/patch_cls/model1/cache"

POS_THR = 0.5


def load_split():
    with open(SPLIT) as f:
        s = json.load(f)
    return s["train_names"], s["val_names"], s["test_names"]


def load_labels():
    with open(LABELS, "rb") as f:
        lab = pickle.load(f)
    meta = lab.pop("_meta")
    return lab, set(meta["non_exhaustive_tumor_train_slides"])


def _np(a):
    return a.numpy() if isinstance(a, torch.Tensor) else np.asarray(a)


def load_feat(slide, root):
    d = torch.load(f"{root}/{slide}.pt", weights_only=False)
    return _np(d["feat"]), _np(d["coords"])


TILE = 2048


def neighbour_mean(tile_mean, coords, radius=1):
    """Mean tile-feature over the (2r+1)^2 tile block around each tile.

    Coords are level-0 pixel positions on a TILE-pixel grid with a per-slide offset, so
    they are snapped to integer cells before the lookup rather than assumed aligned.
    """
    g = np.rint(coords.astype(np.float64) / TILE).astype(np.int64)
    cell = {}
    for i, (gx, gy) in enumerate(g):
        cell.setdefault((gx, gy), []).append(i)
    out = np.empty_like(tile_mean)
    for i, (gx, gy) in enumerate(g):
        idx = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                idx.extend(cell.get((gx + dx, gy + dy), ()))
        out[i] = tile_mean[idx].mean(axis=0)
    return out


def build(names, labels, nonexh, root, filtered):
    """filtered=True -> apply the training label policy; False -> keep everything."""
    Xs, Cs, Ss, Ns, Ms, Ks, ys, fr, sid, missing = [], [], [], [], [], [], [], [], [], []
    for si, name in enumerate(sorted(names)):
        p = f"{root}/{name}.pt"
        if not os.path.exists(p):
            missing.append(name)
            continue
        feat, coords = load_feat(name, root)
        sub = labels[name]["sub_frac"].astype(np.float32)
        n = min(feat.shape[0], sub.shape[0])
        if feat.shape[0] != sub.shape[0]:
            print(f"  !! {name}: feat {feat.shape[0]} vs label {sub.shape[0]}, truncating to {n}")
        feat, sub, coords = feat[:n], sub[:n], coords[:n]
        f32 = feat.astype(np.float32)
        tmean = f32.mean(axis=1)                                  # (n,384) per-tile mean
        ctx = np.repeat(tmean[:, None, :], 16, axis=1).astype(np.float16)
        nb = np.repeat(neighbour_mean(tmean, coords, 1)[:, None, :], 16, axis=1).astype(np.float16)
        mb = np.repeat(neighbour_mean(tmean, coords, 2)[:, None, :], 16, axis=1).astype(np.float16)
        kb = np.repeat(neighbour_mean(tmean, coords, 4)[:, None, :], 16, axis=1).astype(np.float16)
        # slide mean over every sub-patch, computed before any label filtering so it is
        # exactly what inference can compute on an unlabelled slide
        smean = f32.reshape(-1, 384).mean(axis=0).astype(np.float16)
        f = feat.reshape(-1, 384)
        c = ctx.reshape(-1, 384)
        nbf = nb.reshape(-1, 384)
        mbf = mb.reshape(-1, 384)
        kbf = kb.reshape(-1, 384)
        s = sub.reshape(-1)
        if filtered:
            pos = s >= POS_THR
            neg = (s == 0.0) & (name not in nonexh)
            keep = pos | neg
            f, c, nbf, mbf, kbf = f[keep], c[keep], nbf[keep], mbf[keep], kbf[keep]
            ys.append(pos[keep].astype(np.uint8))
            fr.append(s[keep])
        else:
            ys.append((s >= POS_THR).astype(np.uint8))
            fr.append(s)
        Xs.append(f)
        Cs.append(c)
        Ns.append(nbf)
        Ms.append(mbf)
        Ks.append(kbf)
        Ss.append(np.repeat(smean[None, :], f.shape[0], axis=0))
        sid.append(np.full(f.shape[0], si, dtype=np.int16))
    out = dict(
        X=np.concatenate(Xs), C=np.concatenate(Cs), S=np.concatenate(Ss),
        N=np.concatenate(Ns), M=np.concatenate(Ms), K=np.concatenate(Ks),
        y=np.concatenate(ys), frac=np.concatenate(fr), slide=np.concatenate(sid),
        slide_names=np.array(sorted(names)),
    )
    return out, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="trainval", choices=["trainval", "test"])
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    labels, nonexh = load_labels()
    tr, va, te = load_split()

    if a.which == "trainval":
        t0 = time.time()
        print(f"non-exhaustive slides: {len(nonexh)}")
        print(f"val slides that are non-exhaustive: {sorted(set(va) & nonexh)}")
        for tag, names, filt in (("train", tr, True), ("val", va, False)):
            d, missing = build(names, labels, nonexh, FEAT_TRAIN, filt)
            print(f"[{tag}] slides={len(names)} missing={len(missing)} {missing}")
            print(f"[{tag}] n={len(d['y'])} pos={int(d['y'].sum())} "
                  f"rate={d['y'].mean():.4%} X={d['X'].shape} {d['X'].dtype}")
            for k, v in d.items():
                np.save(f"{CACHE}/{tag}_{k}.npy", v)
        print(f"done in {time.time()-t0:.0f}s")
    else:
        os.makedirs(f"{CACHE}/test", exist_ok=True)
        for name in sorted(te):
            feat, _ = load_feat(name, FEAT_TEST)
            np.save(f"{CACHE}/test/{name}.npy", feat)
        print(f"cached {len(te)} test slides")


if __name__ == "__main__":
    main()
