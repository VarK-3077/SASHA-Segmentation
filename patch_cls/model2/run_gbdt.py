# LightGBM / XGBoost baselines for Model-2, on the same feature views the MLPs use.
#
# Kept mainly as a check that the Model-1 finding still holds here (GBDTs win on ranking
# and lose badly on calibrated cross-entropy). The view matrices are materialised in
# float32 on the host, which is the reason this is a separate script from the GPU sweeps.

import argparse
import json
import os
import time

import numpy as np

import common as C
import views as V

SRC = ("tr_det", "tr_aug", "tr_agg")


def materialise(d, mode, rows=None):
    s = d["Sm"][d["slide"].astype(np.int64)]
    x = V.make_view_np(d["X"], d["N"], d["M"], d["K"], s, d["V"], mode)
    return x if rows is None else x[rows]


def train_rows(tags, mode):
    xs, ys = [], []
    for t in tags:
        d = C.load_cache(t if t.startswith("tr_") else "tr_" + t)
        xs.append(materialise(d, mode))
        ys.append(C.labels_of(d["frac"]))
    return np.concatenate(xs), np.concatenate(ys)


def fit_lgb(xtr, ytr, xva, yva, cfg):
    import lightgbm as lgb
    p = dict(objective="binary", metric="binary_logloss", learning_rate=cfg.get("lr", 0.05),
             num_leaves=cfg.get("num_leaves", 63), min_data_in_leaf=cfg.get("min_leaf", 50),
             feature_fraction=cfg.get("ff", 0.3), bagging_fraction=cfg.get("bf", 0.8),
             bagging_freq=1, lambda_l2=cfg.get("l2", 1.0), verbose=-1,
             num_threads=cfg.get("threads", 16), seed=cfg.get("seed", 0))
    ds = lgb.Dataset(xtr, ytr)
    dv = lgb.Dataset(xva, yva, reference=ds)
    bst = lgb.train(p, ds, num_boost_round=cfg.get("rounds", 2000), valid_sets=[dv],
                    callbacks=[lgb.early_stopping(cfg.get("patience", 100), verbose=False),
                               lgb.log_evaluation(200)])
    return bst, bst.best_iteration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    a = ap.parse_args()
    os.makedirs(C.PRED, exist_ok=True)
    cfgs = json.load(open(a.sweep))
    evals = {t: C.load_cache(t) for t in ("va_det", "va_agg")}
    for cfg in cfgs:
        t0 = time.time()
        mode = cfg.get("view", "L2:x+n+m+k")
        print(f"\n=== {cfg['name']} :: {json.dumps(cfg)}", flush=True)
        xtr, ytr = train_rows(cfg["rows"], mode)
        xva = {t: materialise(d, mode) for t, d in evals.items()}
        sel = evals["va_det"]
        m = C.clean_mask(sel["frac"], sel["ne"])
        bst, best_it = fit_lgb(xtr, ytr, xva["va_det"][m], C.labels_of(sel["frac"])[m], cfg)
        p = {t: bst.predict(xva[t], num_iteration=best_it).astype(np.float64) for t in xva}
        met = C.eval_set(p["va_det"], evals["va_det"])
        met.update(C.eval_set(p["va_agg"], evals["va_agg"], prefix="agg_"))
        met["best_epoch"] = int(best_it)
        met["n_rows"] = int(len(ytr))
        C.log_result(cfg["name"], "lgb", cfg, met, time.time() - t0)
        for t, v in p.items():
            np.save(f"{C.PRED}/{cfg['name']}_{t}.npy", v.astype(np.float32))


if __name__ == "__main__":
    main()
