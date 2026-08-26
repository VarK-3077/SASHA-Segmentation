# Logistic-regression floor for Model-1.
#
# Plain sklearn on the cached arrays. The class_weight='balanced' variants are also
# reported with the prior shift undone (logit - log(n_neg/n_pos)), because balanced
# training deliberately moves the operating prior and the selection metric is an
# unweighted cross-entropy on the natural prior.

import argparse
import json
import os
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

import common as C

PRED = f"{C.OUT}/preds"
CKPT = f"{C.OUT}/ckpt"


def run(cfg, tr, va):
    t0 = time.time()
    view = cfg.get("view", "raw")
    Xtr = C.make_view(tr["X"], tr["C"], tr["S"], tr["N"], tr["M"], tr["K"], view)
    mu, sd = C.standardizer(Xtr)
    Xtr = (Xtr - mu) / sd
    ytr = tr["y"].astype(np.float64)

    if cfg.get("sub", 1.0) < 1.0:
        rng = np.random.default_rng(0)
        keep = (ytr > 0.5) | (rng.random(len(ytr)) < cfg["sub"])
        Xtr, ytr = Xtr[keep], ytr[keep]

    lr = LogisticRegression(C=cfg.get("C", 1.0), max_iter=cfg.get("max_iter", 200),
                            class_weight=cfg.get("class_weight"), solver="lbfgs", n_jobs=-1)
    lr.fit(Xtr, ytr)
    del Xtr

    Xva = (C.make_view(va["X"], va["C"], va["S"], va["N"], va["M"], va["K"], view) - mu) / sd
    z = Xva @ lr.coef_[0] + lr.intercept_[0]
    del Xva

    shift = 0.0
    if cfg.get("class_weight") == "balanced":
        shift = -float(np.log((ytr < 0.5).sum() / (ytr > 0.5).sum()))
    if cfg.get("sub", 1.0) < 1.0:
        shift += float(np.log(cfg["sub"]))

    dt = time.time() - t0
    out = []
    for tag, sh in ([("", 0.0)] if shift == 0.0 else [("_uncorr", 0.0), ("", shift)]):
        p = 1.0 / (1.0 + np.exp(-(z + sh)))
        met = C.eval_val(p, va)
        name = cfg["name"] + tag
        C.log_result(name, "logreg", cfg, met, dt, extra=dict(shift=sh))
        np.save(f"{PRED}/{name}_val.npy", p.astype(np.float32))
        out.append(name)
    np.savez(f"{CKPT}/{cfg['name']}.npz", coef=lr.coef_[0], intercept=lr.intercept_[0],
             mu=mu, sd=sd, shift=shift, view=view)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    a = ap.parse_args()
    os.makedirs(PRED, exist_ok=True)
    os.makedirs(CKPT, exist_ok=True)
    tr = C.load_split_arrays("train")
    va = C.load_split_arrays("val")
    for cfg in json.load(open(a.sweep)):
        print(f"\n=== {cfg}", flush=True)
        run(cfg, tr, va)


if __name__ == "__main__":
    main()
