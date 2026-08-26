# Builds the flat per-tile arrays Model-2 trains and is scored on.
#
# Six caches get written, three "row sources" for training and three evaluation sets:
#
#   tr_det   train slides, deterministic SASHA-0.2 rollout final state   (variant b)
#   tr_aug   train slides, the 2 stochastic rollouts, stacked            (variant b+ extra)
#   tr_agg   train slides, true phase-1 aggregates from the h5           (variant a)
#   va_det   val slides, deterministic rollout state    <- the inference distribution
#   va_agg   val slides, true aggregates                <- the transfer-check distribution
#   te_det   test slides, deterministic rollout state   <- read once, at the very end
#
# The context pyramid of every cache is pooled over that cache's OWN feature matrix, so a
# rollout-state row never sees an aggregate neighbour and vice versa.
#
# Aggregate rows carry visited=1 for every tile: a visited tile's rollout state IS its true
# phase-1 aggregate (verified: cosine 1.0 on visited indices), so a matrix of pure
# aggregates is exactly the state of a hypothetical rollout that visited everything.
#
# Train caches apply the label policy (tile_frac >= 0.25 positive, == 0 negative and not on
# a non-exhaustively annotated slide, the strip between dropped). Eval caches keep every
# tile plus its tile_frac so any threshold or mask can be derived downstream.

import argparse
import json
import os
import pickle
import time

import h5py
import numpy as np

import context as X

HOME = os.path.expanduser("~")
STATES = f"{HOME}/seg_outputs/patch_cls/rollout_states.pkl"
LABELS = f"{HOME}/seg_outputs/patch_cls/labels.pkl"
SPLIT = f"{HOME}/SASHA-Segmentation/dataset_csv/camelyon16/splits/split_4.json"
AGG_H5 = "/data2/tej/SASHA/hafed13julyintmdtfeat/patch_feats_pretrain_medical_ssl.h5"
LR_H5 = "/data2/tej/SASHA/camelyon_lr_features/patch_feats_pretrain_medical_ssl.h5"
CACHE = f"{HOME}/seg_outputs/patch_cls/model2/cache"

POS_THR = 0.25
KEYS = ("X", "N", "M", "K", "V", "frac", "slide", "ne")


def load_split():
    with open(SPLIT) as f:
        s = json.load(f)
    return sorted(s["train_names"]), sorted(s["val_names"]), sorted(s["test_names"])


def build(names, feat_of, coords_of, labels, nonexh, filtered, reps=1):
    """feat_of(slide, rep) -> ((N,384), visited idx). reps>1 stacks several draws as rows."""
    acc = {k: [] for k in KEYS}
    smeans = []
    for si, name in enumerate(names):
        coords = coords_of(name)
        frac = labels[name]["tile_frac"].astype(np.float32)
        ne = name in nonexh
        for rep in range(reps):
            feat, visited = feat_of(name, rep)
            means, smean, aux = X.pyramid(feat, coords, visited)
            keep = slice(None)
            if filtered:
                keep = (frac >= POS_THR) | ((frac == 0.0) & (not ne))
            acc["X"].append(np.asarray(feat, dtype=np.float32)[keep].astype(np.float16))
            for key, mm in zip(("N", "M", "K"), means):
                acc[key].append(mm[keep].astype(np.float16))
            acc["V"].append(aux[keep].astype(np.float16))
            acc["frac"].append(frac[keep])
            n_kept = acc["X"][-1].shape[0]
            # one slide-mean row per (slide, rep): two stochastic rollouts of the same slide
            # have different slide means, and a row's s block has to come from its own
            # rollout like every other context block does
            acc["slide"].append(np.full(n_kept, si * reps + rep, dtype=np.int32))
            acc["ne"].append(np.full(n_kept, ne, dtype=bool))
            smeans.append(smean.astype(np.float16))
    out = {k: np.concatenate(v) for k, v in acc.items()}
    out["Sm"] = np.stack(smeans)
    out["slide_names"] = np.array(names)
    out["reps"] = np.array(reps)
    return out


def report(tag, d):
    y = (d["frac"] >= POS_THR)
    neg = (d["frac"] == 0.0) & ~d["ne"]
    print(f"[{tag}] rows={len(y):,} pos={int(y.sum()):,} neg={int(neg.sum()):,} "
          f"boundary={int(((d['frac'] > 0) & (d['frac'] < POS_THR)).sum()):,} "
          f"ne_zero={int(((d['frac'] == 0) & d['ne']).sum()):,} "
          f"visited_frac={float(d['V'][:, 0].astype(np.float32).mean()):.4f} "
          f"X={d['X'].shape}", flush=True)


def save(tag, d):
    os.makedirs(CACHE, exist_ok=True)
    for k, v in d.items():
        np.save(f"{CACHE}/{tag}_{k}.npy", v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="trainval", choices=["trainval", "test", "all"])
    a = ap.parse_args()
    t0 = time.time()

    with open(STATES, "rb") as f:
        st = pickle.load(f)["slides"]
    with open(LABELS, "rb") as f:
        labels = pickle.load(f)
    nonexh = set(labels["_meta"]["non_exhaustive_tumor_train_slides"])
    tr, va, te = load_split()

    lr = h5py.File(LR_H5, "r")
    agg = h5py.File(AGG_H5, "r")
    coords_of = lambda s: lr[s]["coords"][:]
    det_of = lambda s, rep: (st[s]["state"], st[s]["visited"])
    aug_of = lambda s, rep: (st[s]["aug"][rep]["state"], st[s]["aug"][rep]["visited"])

    def agg_of(s, rep):
        f = agg[s]["feat"][:]
        return f, np.arange(len(f))

    jobs = []
    if a.which in ("trainval", "all"):
        jobs += [("tr_det", tr, det_of, True, 1), ("tr_aug", tr, aug_of, True, 2),
                 ("tr_agg", tr, agg_of, True, 1),
                 ("va_det", va, det_of, False, 1), ("va_agg", va, agg_of, False, 1)]
    if a.which in ("test", "all"):
        jobs += [("te_det", te, det_of, False, 1)]

    for tag, names, fn, filt, reps in jobs:
        d = build(names, fn, coords_of, labels, nonexh, filt, reps)
        report(tag, d)
        save(tag, d)
    lr.close()
    agg.close()
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
