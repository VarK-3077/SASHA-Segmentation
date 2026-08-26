# LightGBM / XGBoost runs for Model-1.
#
# Trees on 384-d ViT features are a genuinely different inductive bias from the linear
# and MLP families, which is what makes them worth keeping for the ensemble even if they
# lose on their own. Early stopping is on the validation logloss restricted to the clean
# label policy, matching the selection metric.

import argparse
import json
import os
import time

import numpy as np

import common as C

PRED = f"{C.OUT}/preds"
CKPT = f"{C.OUT}/ckpt"


def make_val(va, view):
    Xva = C.make_view(va["X"], va["C"], va["S"], va["N"], va["M"], va["K"], view)
    m = C.clean_mask(va["frac"])
    y_all = (va["frac"] >= 0.5).astype(np.float64)
    return Xva, m, y_all


def run_lgb(cfg, tr, va):
    import lightgbm as lgb
    t0 = time.time()
    view = cfg.get("view", "raw")
    Xtr = C.make_view(tr["X"], tr["C"], tr["S"], tr["N"], tr["M"], tr["K"], view)
    ytr = tr["y"].astype(np.float32)
    if cfg.get("sub", 1.0) < 1.0:
        rng = np.random.default_rng(0)
        keep = (ytr > 0.5) | (rng.random(len(ytr)) < cfg["sub"])
        Xtr, ytr = Xtr[keep], ytr[keep]
    Xva, m, y_all = make_val(va, view)

    params = dict(objective="binary", metric="binary_logloss", learning_rate=cfg.get("lr", 0.05),
                  num_leaves=cfg.get("num_leaves", 63), min_data_in_leaf=cfg.get("min_leaf", 200),
                  feature_fraction=cfg.get("ff", 0.5), bagging_fraction=cfg.get("bf", 0.8),
                  bagging_freq=1, lambda_l2=cfg.get("l2", 1.0), max_bin=cfg.get("max_bin", 63),
                  num_threads=cfg.get("threads", 32), verbosity=-1, force_row_wise=True)
    if cfg.get("scale_pos_weight"):
        spw = cfg["scale_pos_weight"]
        params["scale_pos_weight"] = float((ytr < 0.5).sum() / (ytr > 0.5).sum()) if spw == "auto" else float(spw)

    dtr = lgb.Dataset(Xtr, label=ytr)
    dva = lgb.Dataset(Xva[m], label=y_all[m], reference=dtr)
    bst = lgb.train(params, dtr, num_boost_round=cfg.get("rounds", 3000), valid_sets=[dva],
                    callbacks=[lgb.early_stopping(cfg.get("patience", 50), verbose=False),
                               lgb.log_evaluation(100)])
    del Xtr, dtr, dva

    shift = 0.0
    if params.get("scale_pos_weight"):
        shift -= float(np.log(params["scale_pos_weight"]))
    if cfg.get("sub", 1.0) < 1.0:
        shift += float(np.log(cfg["sub"]))

    z = bst.predict(Xva, raw_score=True)
    dt = time.time() - t0
    for tag, sh in ([("", 0.0)] if shift == 0.0 else [("_uncorr", 0.0), ("", shift)]):
        p = 1.0 / (1.0 + np.exp(-(z + sh)))
        met = C.eval_val(p, va)
        met["best_iter"] = bst.best_iteration
        C.log_result(cfg["name"] + tag, "lightgbm", cfg, met, dt, extra=dict(shift=sh))
        np.save(f"{PRED}/{cfg['name']}{tag}_val.npy", p.astype(np.float32))
    bst.save_model(f"{CKPT}/{cfg['name']}.lgb", num_iteration=bst.best_iteration)
    json.dump(dict(view=view, shift=shift), open(f"{CKPT}/{cfg['name']}.lgb.json", "w"))


def run_xgb(cfg, tr, va):
    import xgboost as xgb
    t0 = time.time()
    view = cfg.get("view", "raw")
    Xtr = C.make_view(tr["X"], tr["C"], tr["S"], tr["N"], tr["M"], tr["K"], view)
    ytr = tr["y"].astype(np.float32)
    if cfg.get("sub", 1.0) < 1.0:
        rng = np.random.default_rng(0)
        keep = (ytr > 0.5) | (rng.random(len(ytr)) < cfg["sub"])
        Xtr, ytr = Xtr[keep], ytr[keep]
    Xva, m, y_all = make_val(va, view)

    spw = cfg.get("scale_pos_weight")
    if spw == "auto":
        spw = float((ytr < 0.5).sum() / (ytr > 0.5).sum())
    p = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
             device=cfg.get("device", "cuda:1"), max_depth=cfg.get("max_depth", 8),
             eta=cfg.get("lr", 0.05), subsample=cfg.get("subsample", 0.8),
             colsample_bytree=cfg.get("ff", 0.5), min_child_weight=cfg.get("min_child", 20),
             reg_lambda=cfg.get("l2", 1.0), max_bin=cfg.get("max_bin", 64))
    if spw:
        p["scale_pos_weight"] = float(spw)
    dtr = xgb.QuantileDMatrix(Xtr, label=ytr)
    dva = xgb.QuantileDMatrix(Xva[m], label=y_all[m], ref=dtr)
    bst = xgb.train(p, dtr, num_boost_round=cfg.get("rounds", 3000), evals=[(dva, "val")],
                    early_stopping_rounds=cfg.get("patience", 50), verbose_eval=100)
    del Xtr, dtr, dva

    shift = 0.0
    if spw:
        shift -= float(np.log(spw))
    if cfg.get("sub", 1.0) < 1.0:
        shift += float(np.log(cfg["sub"]))
    z = bst.inplace_predict(Xva, iteration_range=(0, bst.best_iteration + 1),
                            predict_type="margin")
    dt = time.time() - t0
    for tag, sh in ([("", 0.0)] if shift == 0.0 else [("_uncorr", 0.0), ("", shift)]):
        pr = 1.0 / (1.0 + np.exp(-(z + sh)))
        met = C.eval_val(pr, va)
        met["best_iter"] = int(bst.best_iteration)
        C.log_result(cfg["name"] + tag, "xgboost", cfg, met, dt, extra=dict(shift=sh))
        np.save(f"{PRED}/{cfg['name']}{tag}_val.npy", pr.astype(np.float32))
    bst.save_model(f"{CKPT}/{cfg['name']}.ubj")
    json.dump(dict(view=view, shift=shift, best_iter=int(bst.best_iteration)),
              open(f"{CKPT}/{cfg['name']}.ubj.json", "w"))


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
        (run_xgb if cfg.get("lib") == "xgb" else run_lgb)(cfg, tr, va)


if __name__ == "__main__":
    main()
