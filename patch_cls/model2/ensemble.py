# Seed / config ensembling for Model-2, straight off the saved validation predictions.
#
# Members are combined by averaging logits, which is what worked for Model-1. Selecting
# the member list ON the validation set is optimistic, so anything chosen that way is
# suffixed _VALFIT and treated as a diagnostic; the honest candidate is the seed-ensemble
# of a config picked by its multi-seed mean CE.

import argparse
import json
import os
import re

import numpy as np

import common as C

EPS = 1e-7
TAGS = ("va_det", "va_agg")


def logit(p):
    p = np.clip(p.astype(np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def load(name, tag):
    return np.load(f"{C.PRED}/{name}_{tag}.npy").astype(np.float64)


def combine(names, tag):
    return 1.0 / (1.0 + np.exp(-np.mean([logit(load(n, tag)) for n in names], axis=0)))


def score(names, label, evals, family="ensemble", note=""):
    p = {t: combine(names, t) for t in TAGS}
    met = C.eval_set(p["va_det"], evals["va_det"])
    met.update(C.eval_set(p["va_agg"], evals["va_agg"], prefix="agg_"))
    met["best_epoch"] = 0
    met["n_rows"] = 0
    C.log_result(label, family, f"{len(names)} members: {note or 'logit-mean'}", met, 0.0,
                 extra=dict(members=list(names)))
    for t, v in p.items():
        np.save(f"{C.PRED}/{label}_{t}.npy", v.astype(np.float32))
    return met


def seed_groups(rows):
    """Group leaderboard rows by config name with the trailing _s<k> stripped."""
    g = {}
    for r in rows:
        b = re.sub(r"_s\d+$", "", r["name"])
        if b != r["name"]:
            g.setdefault(b, []).append(r)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-groups", action="store_true", help="ensemble every _s<k> family")
    ap.add_argument("--members", nargs="*", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--top", type=int, default=0, help="also build top-k VALFIT ensembles")
    a = ap.parse_args()
    evals = {t: C.load_cache(t) for t in TAGS}
    rows = [r for r in C.read_lb() if r["family"] in ("mlp", "lgb")]

    if a.members:
        score(a.members, a.label or "ens_custom", evals)
    if a.seed_groups:
        summary = []
        for base, grp in sorted(seed_groups(rows).items()):
            names = [r["name"] for r in grp if os.path.exists(f"{C.PRED}/{r['name']}_va_det.npy")]
            if len(names) < 2:
                continue
            ces = np.array([r["ce"] for r in grp])
            agg = np.array([r["agg_ce"] for r in grp])
            met = score(sorted(names), f"ens_{base}", evals, note=f"seed-mean of {base}")
            summary.append(dict(base=base, n=len(names), mean_ce=float(ces.mean()),
                                sd_ce=float(ces.std(ddof=1)) if len(ces) > 1 else 0.0,
                                min_ce=float(ces.min()), max_ce=float(ces.max()),
                                mean_agg_ce=float(agg.mean()), ens_ce=met["ce"],
                                ens_agg_ce=met["agg_ce"], ens_roc=met["roc_auc"],
                                ens_pr=met["pr_auc"], view=grp[0]["config"].get("view"),
                                rows=grp[0]["config"].get("rows")))
        summary.sort(key=lambda r: r["mean_ce"])
        with open(f"{C.OUT}/seed_summary.json", "w") as f:
            json.dump(summary, f, indent=1)
        for r in summary:
            print(f"{r['base']:>18s}  mean={r['mean_ce']:.5f} sd={r['sd_ce']:.5f} "
                  f"ens={r['ens_ce']:.5f}  agg={r['mean_agg_ce']:.5f}")
    if a.top:
        rows.sort(key=lambda r: r["ce"])
        for k in range(2, a.top + 1):
            score([r["name"] for r in rows[:k]], f"ens_top{k}_VALFIT", evals,
                  note="top-k selected ON val: optimistic")


if __name__ == "__main__":
    main()
