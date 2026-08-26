# Shared plumbing for the Model-1 sub-patch classifier experiments.
#
# Loads the cached flat arrays, builds the various feature views (raw / L2-normalised /
# with tile context), computes the validation metrics everything is ranked by, and
# appends results to a single JSONL leaderboard that the markdown table is rendered from.

import json
import os
import time

import numpy as np

import views as V

HOME = os.path.expanduser("~")
CACHE = f"{HOME}/seg_outputs/patch_cls/model1/cache"
OUT = f"{HOME}/seg_outputs/patch_cls/model1"
LB_JSONL = f"{OUT}/leaderboard.jsonl"

EPS = 1e-7


def load_split_arrays(tag):
    d = {}
    for k in ("X", "C", "S", "N", "M", "K", "y", "frac", "slide"):
        d[k] = np.load(f"{CACHE}/{tag}_{k}.npy")
    d["slide_names"] = np.load(f"{CACHE}/{tag}_slide_names.npy")
    return d


def clean_mask(frac):
    """Sub-patches the clean label policy keeps: unambiguous positive or pure negative."""
    return (frac >= 0.5) | (frac == 0.0)


make_view = V.make_view_np
parse_view = V.parse_view
view_dim = V.view_dim
_blocks = V.blocks


def standardizer(Xtr):
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def bce(p, y):
    p = np.clip(p.astype(np.float64), EPS, 1 - EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def roc_auc(p, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def pr_auc(p, y):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, p))


def eval_val(p_all, val):
    """p_all: probability for EVERY val sub-patch, in cache order."""
    y_all = (val["frac"] >= 0.5).astype(np.float64)
    m = clean_mask(val["frac"])
    p_c, y_c = p_all[m], y_all[m]
    return dict(
        ce_clean=bce(p_c, y_c),
        ce_all=bce(p_all, y_all),
        roc_auc=roc_auc(p_c, y_c),
        pr_auc=pr_auc(p_c, y_c),
        n_clean=int(m.sum()),
        n_all=int(len(y_all)),
        pos_rate_clean=float(y_c.mean()),
    )


def per_slide_ce(p_all, val):
    """CE per val slide under the clean policy, sorted worst first."""
    y_all = (val["frac"] >= 0.5).astype(np.float64)
    m = clean_mask(val["frac"])
    rows = []
    for si, name in enumerate(val["slide_names"]):
        sel = m & (val["slide"] == si)
        if sel.sum() == 0:
            continue
        rows.append(dict(slide=str(name), n=int(sel.sum()), pos=int(y_all[sel].sum()),
                         ce=bce(p_all[sel], y_all[sel]),
                         mean_p=float(p_all[sel].mean())))
    return sorted(rows, key=lambda r: -r["ce"])


def platt(p_val, y_val, p_target):
    """Fit a 1-d logistic on val logits, apply to p_target. Reuses val - report as such."""
    from sklearn.linear_model import LogisticRegression
    z = np.log(np.clip(p_val, EPS, 1 - EPS) / (1 - np.clip(p_val, EPS, 1 - EPS)))
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(z.reshape(-1, 1), y_val)
    zt = np.log(np.clip(p_target, EPS, 1 - EPS) / (1 - np.clip(p_target, EPS, 1 - EPS)))
    return lr.predict_proba(zt.reshape(-1, 1))[:, 1], float(lr.coef_[0][0]), float(lr.intercept_[0])


def log_result(name, family, config, metrics, train_s, extra=None):
    os.makedirs(OUT, exist_ok=True)
    row = dict(name=name, family=family, config=config, train_s=round(train_s, 1),
               ts=time.strftime("%Y-%m-%d %H:%M:%S"), **metrics)
    if extra:
        row["extra"] = extra
    with open(LB_JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[LB] {name}  ce_clean={metrics['ce_clean']:.5f}  ce_all={metrics['ce_all']:.5f}  "
          f"roc={metrics['roc_auc']:.4f}  pr={metrics['pr_auc']:.4f}  {train_s:.0f}s")
    return row
