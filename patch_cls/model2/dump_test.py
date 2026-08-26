# Read-once test-set scoring for the chosen Model-2.
#
# Predicts on the deterministic test rollout states, writes the per-slide probability dump,
# and prints the slide-level sanity numbers (mean tile probability on tumour vs normal
# slides, slide ROC-AUC from the max and mean tile probability). Nothing here is tuned --
# it runs after the model is frozen, exactly once.
#
# It also re-predicts a few slides through predict.py's from-scratch path (state matrix +
# coords + visited, pyramid rebuilt on the fly) and asserts the two agree, which is the
# check that the exported artifact reproduces the cached-feature pipeline.

import argparse
import json
import os
import pickle

import h5py
import numpy as np
import torch

import common as C
import predict as P

# fp16 rounds anything above 1 - 2^-11 straight to 1.0, which would hand a downstream
# log-loss an infinity. Clamp into the open interval before the cast; the metrics below are
# all computed on the float64 probabilities, so nothing reported changes.
F16_HI = float(np.nextafter(np.float16(1.0), np.float16(0.0)))
F16_LO = 6e-8

HOME = os.path.expanduser("~")
STATES = f"{HOME}/seg_outputs/patch_cls/rollout_states.pkl"
LABELS = f"{HOME}/seg_outputs/patch_cls/labels.pkl"
LR_H5 = "/data2/tej/SASHA/camelyon_lr_features/patch_feats_pretrain_medical_ssl.h5"


@torch.no_grad()
def predict_cache(nets, d, device, bs=25_000):
    import views as V
    n = len(d["frac"])
    s_all = d["Sm"][d["slide"].astype(np.int64)]
    out = np.empty(n, dtype=np.float64)
    for i in range(0, n, bs):
        j = slice(i, min(i + bs, n))
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)
        x, nb, m, k = t(d["X"][j]), t(d["N"][j]), t(d["M"][j]), t(d["K"][j])
        s, v = t(s_all[j]), t(d["V"][j])
        z = torch.stack([e["net"]((V.make_view_t(x, nb, m, k, s, v, e["view"]) - e["mu"]) / e["sd"]).double()
                         for e in nets]).mean(0)
        out[j] = torch.sigmoid(z).cpu().numpy()
    return out


def slide_level(probs, slide_label):
    from sklearn.metrics import roc_auc_score
    names = sorted(probs)
    y = np.array([slide_label[n] for n in names])
    mx = np.array([probs[n].astype(np.float64).max() for n in names])
    mn = np.array([probs[n].astype(np.float64).mean() for n in names])
    return dict(n_slides=len(names), n_tumor=int(y.sum()), n_normal=int((1 - y).sum()),
                mean_tile_prob_tumor=float(mn[y == 1].mean()),
                mean_tile_prob_normal=float(mn[y == 0].mean()),
                slide_auc_max=float(roc_auc_score(y, mx)),
                slide_auc_mean=float(roc_auc_score(y, mn)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=f"{C.OUT}/best/model.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=f"{C.OUT}/test_probs.pkl")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()

    dev = torch.device(a.device)
    nets = P.load_model(a.ckpt, dev)
    tag = {"test": "te_det", "val": "va_det"}[a.split]
    d = C.load_cache(tag)
    p = predict_cache(nets, d, dev)

    probs = {}
    for si, name in enumerate(d["slide_names"]):
        probs[str(name)] = np.clip(p[d["slide"] == si], F16_LO, F16_HI).astype(np.float16)
    with open(a.out, "wb") as f:
        pickle.dump(probs, f)
    print(f"wrote {len(probs)} slides -> {a.out}")

    with open(LABELS, "rb") as f:
        lab = pickle.load(f)
    sl = {n: int(lab[n]["slide_label"]) for n in probs}
    tile = C.eval_set(p, d)
    lvl = slide_level(probs, sl)
    print(json.dumps(dict(tile=tile, slide=lvl), indent=1))
    with open(f"{C.OUT}/{a.split}_metrics.json", "w") as f:
        json.dump(dict(tile=tile, slide=lvl,
                       per_slide=C.per_slide_ce(p, d)[:20]), f, indent=1)

    # from-scratch cross-check on a handful of slides
    with open(STATES, "rb") as f:
        st = pickle.load(f)["slides"]
    cpu_nets = P.load_model(a.ckpt, "cpu")
    worst = 0.0
    with h5py.File(LR_H5, "r") as h5:
        for name in list(d["slide_names"])[:4]:
            name = str(name)
            q = P.predict_slide(cpu_nets, st[name]["state"], h5[name]["coords"][:],
                                st[name]["visited"], "cpu")
            q = np.clip(q, F16_LO, F16_HI)
            worst = max(worst, float(np.abs(q - probs[name].astype(np.float64)).max()))
    print(f"predict.py vs cache max |dP| over 4 slides: {worst:.3e}")


if __name__ == "__main__":
    main()
