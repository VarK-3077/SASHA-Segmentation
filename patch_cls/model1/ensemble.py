# Ensembling and calibration on top of the saved per-model validation probabilities.
#
# Averaging happens in logit space, which is the right space for a metric that is itself
# a log-loss: it multiplies odds rather than probabilities and does not get dragged around
# by one member's near-zero output. Greedy forward selection is included but it picks on
# the same val split it is scored on, so its number is optimistic by construction and is
# labelled as such.

import argparse
import glob
import itertools
import json
import os
import re

import numpy as np

import common as C

PRED = f"{C.OUT}/preds"


def load_preds(names):
    return {n: np.load(f"{PRED}/{n}_val.npy").astype(np.float64) for n in names}


def logit(p):
    p = np.clip(p, C.EPS, 1 - C.EPS)
    return np.log(p / (1 - p))


def avg(preds, names, w=None):
    z = np.stack([logit(preds[n]) for n in names])
    w = np.ones(len(names)) if w is None else np.asarray(w, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-(w[:, None] * z).sum(0) / w.sum()))


def greedy(preds, cand, va, rounds=8):
    """Caruana-style forward selection with replacement, scored on val CE."""
    m, y = C.clean_mask(va["frac"]), (va["frac"] >= 0.5).astype(np.float64)
    chosen, best_ce = [], 1e9
    for _ in range(rounds):
        sc = [(C.bce(avg(preds, chosen + [n])[m], y[m]), n) for n in cand]
        ce, n = min(sc)
        if ce >= best_ce - 1e-6:
            break
        chosen.append(n)
        best_ce = ce
    return chosen, best_ce


SEED_RE = re.compile(r"^(.*)_s\d+$")


def seed_groups(names):
    g = {}
    for n in names:
        mt = SEED_RE.match(n)
        if mt:
            g.setdefault(mt.group(1), []).append(n)
    return {k: sorted(v) for k, v in g.items() if len(v) > 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--members", nargs="*", default=None)
    a = ap.parse_args()
    va = C.load_split_arrays("val")
    rows = [json.loads(l) for l in open(C.LB_JSONL)]
    seen = {}
    for r in rows:
        seen[r["name"]] = r
    avail = {os.path.basename(f)[:-8] for f in glob.glob(f"{PRED}/*_val.npy")}
    rank = sorted([r for r in seen.values()
                   if r["name"] in avail and not r["name"].startswith(("smoke", "ens_", "platt_"))
                   and r["name"] != "BUNDLE"],
                  key=lambda r: r["ce_clean"])

    combos = {}
    # 1. average the seed replicates of each finalist config
    groups = seed_groups([r["name"] for r in rank])
    for g, mem in groups.items():
        ces = [seen[n]["ce_clean"] for n in mem]
        print(f"{g}: {len(mem)} seeds  CE mean={np.mean(ces):.5f} sd={np.std(ces):.5f} "
              f"min={min(ces):.5f} max={max(ces):.5f}")
        combos[f"ens_{g}"] = mem
    # 2. all finalist seed models pooled
    if groups:
        combos["ens_allseeds"] = sorted(sum(groups.values(), []))
    # 3. plain top-k of individual models
    cand = a.members or [r["name"] for r in rank[:a.top]]
    for k in range(2, min(len(cand), 6) + 1):
        combos[f"ens_top{k}"] = cand[:k]

    preds = load_preds(sorted(set(sum(combos.values(), [])) | set(cand)))
    print("\ncandidates:", cand)

    for name, mem in combos.items():
        p = avg(preds, mem)
        met = C.eval_val(p, va)
        C.log_result(name, "ensemble", dict(members=mem, method="logit-mean"), met, 0.0)
        np.save(f"{PRED}/{name}_val.npy", p.astype(np.float32))

    g, gce = greedy(preds, list(preds), va)
    p = avg(preds, g)
    met = C.eval_val(p, va)
    C.log_result("ens_greedy_VALFIT", "ensemble",
                 dict(members=g, method="greedy-fwd (selected ON val: optimistic)"), met, 0.0)
    np.save(f"{PRED}/ens_greedy_VALFIT_val.npy", p.astype(np.float32))

    # Platt on the strongest non-greedy ensemble, for the before/after calibration note
    best = min((n for n in combos if not n.startswith("ens_greedy")),
               key=lambda n: C.bce(avg(preds, combos[n])[C.clean_mask(va["frac"])],
                                   (va["frac"] >= 0.5).astype(np.float64)[C.clean_mask(va["frac"])]))
    pb = avg(preds, combos[best])
    m, y = C.clean_mask(va["frac"]), (va["frac"] >= 0.5).astype(np.float64)
    pc, A, B = C.platt(pb[m], y[m], pb)
    met = C.eval_val(pc, va)
    C.log_result(f"platt_{best}_VALFIT", "calibration",
                 dict(members=combos[best], method=f"Platt on val (A={A:.3f}, B={B:.3f}); val-reuse"),
                 met, 0.0)
    print(f"\nPlatt on {best}: CE {C.bce(pb[m], y[m]):.5f} -> {C.bce(pc[m], y[m]):.5f} "
          f"(A={A:.3f} B={B:.3f})")


if __name__ == "__main__":
    main()
