# Shared plumbing for the Model-2 tile classifier experiments.
#
# Loads the cached flat arrays, holds the metric definitions everything is ranked by, and
# appends every run to a single JSONL leaderboard the markdown table is rendered from.
#
# The one selection metric is `ce_state`: unweighted cross-entropy over the validation
# tiles the clean label policy keeps, computed on DETERMINISTIC ROLLOUT-STATE inputs,
# because that is what the deployed pipeline actually hands the model. `ce_agg` is the same
# thing on true-aggregate inputs and is reported only, never selected on.

import json
import os
import time

import numpy as np

import views as V

HOME = os.path.expanduser("~")
CACHE = f"{HOME}/seg_outputs/patch_cls/model2/cache"
OUT = f"{HOME}/seg_outputs/patch_cls/model2"
LB_JSONL = f"{OUT}/leaderboard.jsonl"
PRED = f"{OUT}/preds"
CKPT = f"{OUT}/ckpt"

EPS = 1e-7
POS_THR = 0.25

make_view = V.make_view_np
parse_view = V.parse_view
view_dim = V.view_dim


def load_cache(tag):
    d = {}
    for k in ("X", "N", "M", "K", "V", "Sm", "frac", "slide", "ne"):
        d[k] = np.load(f"{CACHE}/{tag}_{k}.npy")
    d["slide_names"] = np.load(f"{CACHE}/{tag}_slide_names.npy")
    return d


def clean_mask(frac, ne, thr=POS_THR):
    """Tiles the clean label policy keeps: clear positive, or pure negative we trust."""
    return (frac >= thr) | ((frac == 0.0) & ~ne)


def labels_of(frac, thr=POS_THR):
    return (frac >= thr).astype(np.float64)


def bce(p, y):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def eval_set(p_all, d, thr=POS_THR, prefix=""):
    """p_all: probability for EVERY tile of the cache, in cache order."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = labels_of(d["frac"], thr)
    m = clean_mask(d["frac"], d["ne"], thr)
    pc, yc = p_all[m], y[m]
    return {prefix + "ce": bce(pc, yc), prefix + "ce_all": bce(p_all, y),
            prefix + "roc_auc": float(roc_auc_score(yc, pc)),
            prefix + "pr_auc": float(average_precision_score(yc, pc)),
            prefix + "n_clean": int(m.sum()), prefix + "pos_rate": float(yc.mean())}


def per_slide_ce(p_all, d, thr=POS_THR):
    """Per-slide CE plus that slide's share of the pooled CE, worst share first.

    The share matters more than the rate: a handful of tumour slides can own most of the
    total loss while every normal slide contributes almost nothing.
    """
    y = labels_of(d["frac"], thr)
    m = clean_mask(d["frac"], d["ne"], thr)
    n_tot = int(m.sum())
    rows = []
    for si, name in enumerate(d["slide_names"]):
        sel = m & (d["slide"] == si)
        if sel.sum() == 0:
            continue
        ce = bce(p_all[sel], y[sel])
        rows.append(dict(slide=str(name), n=int(sel.sum()), pos=int(y[sel].sum()),
                         ce=ce, ce_share=ce * int(sel.sum()) / n_tot,
                         mean_p=float(p_all[sel].mean()), max_p=float(p_all[sel].max())))
    return sorted(rows, key=lambda r: -r["ce_share"])


def log_result(name, family, config, metrics, train_s, extra=None):
    os.makedirs(OUT, exist_ok=True)
    row = dict(name=name, family=family, config=config, train_s=round(train_s, 1),
               ts=time.strftime("%Y-%m-%d %H:%M:%S"), **metrics)
    if extra:
        row["extra"] = extra
    with open(LB_JSONL, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[LB] {name}  ce_state={metrics['ce']:.5f}  ce_agg={metrics.get('agg_ce', float('nan')):.5f}  "
          f"roc={metrics['roc_auc']:.4f}  pr={metrics['pr_auc']:.4f}  {train_s:.0f}s", flush=True)
    return row


def read_lb():
    if not os.path.exists(LB_JSONL):
        return []
    with open(LB_JSONL) as f:
        return [json.loads(l) for l in f if l.strip()]
