# Renders the JSONL result log into the markdown leaderboard, with the label-policy
# counts and the seed-replicate summary that the raw table cannot show.

import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

import common as C

SEED_RE = re.compile(r"^(.*)_s\d+$")

HEAD = """# Model-1 leaderboard - per-sub-patch cancer classifier

Binary classifier over single 512px sub-patches, from precomputed 384-d ViT-S features.
Selection metric is **val CE (clean)**: unweighted cross-entropy over validation
sub-patches that the clean label policy keeps (sub_frac >= 0.5 positive, sub_frac == 0
negative). **val CE (all)** additionally includes the 0 < sub_frac < 0.5 boundary
sub-patches, labelled by the >= 0.5 rule; it is reported, never selected on.

## Data and label policy

| | slides with features | sub-patches | positives | negatives | boundary dropped | pos rate |
|---|---|---|---|---|---|---|
| train (clean policy) | 229 of 242 | 2,493,685 | 61,408 | 2,432,277 | 13,808 | 2.463% |
| val (clean policy) | 25 of 27 | 241,361 | 8,583 | 232,778 | 1,791 | 3.556% |
| val (all sub-patches) | 25 of 27 | 243,152 | 8,583 | 234,569 | - | 3.530% |

13 train and 2 val normal slides have no feature dump and are skipped. All 20
non-exhaustively annotated tumour slides fall in train, and 265,403 of their
sub_frac == 0 sub-patches are dropped as untrustworthy negatives; their positives
are kept. No non-exhaustive slide is in val, so the val policy loses nothing.

Feature views are named by which 384-d blocks are concatenated: `x` the sub-patch
feature, `c` its 2048px parent tile mean, `n`/`m`/`k` the mean tile feature over the
3x3 / 5x5 / 9x9 tile neighbourhood, `s` the slide mean, `xc` = x - c, and an `L2:`
prefix L2-normalises every block source first. All of these are computable at inference
time from the slide alone.

Rows suffixed `_uncorr` are the same fit *without* the prior-shift correction that
undoes class re-balancing; both are shown so the cost of re-balancing on an unweighted
CE is visible. Rows suffixed `_VALFIT` are selected or fitted on the validation set
itself and are therefore optimistic - they are diagnostics, not candidates.
"""


def short_cfg(row):
    c = dict(row.get("config", {}))
    c.pop("name", None)
    if row["family"] in ("ensemble", "calibration"):
        mem = c.get("members", [])
        return f"{len(mem)} members: {c.get('method','')}"
    keep = ("view", "hidden", "dropout", "lr", "wd", "loss", "gamma", "neg_keep",
            "label_smooth", "in_noise", "mixup", "in_drop", "pos_weight", "class_weight",
            "sub", "C", "num_leaves", "max_depth", "scale_pos_weight", "lib", "seed")
    skip_default = {"wd": 1e-4, "dropout": 0.0, "label_smooth": 0.0, "seed": 0}
    parts = []
    for k in keep:
        if k not in c or c[k] in (None, False):
            continue
        if skip_default.get(k) == c[k]:
            continue
        parts.append(f"{k}={c[k]}")
    return ", ".join(parts) if parts else "-"


def main():
    rows = [json.loads(l) for l in open(C.LB_JSONL)]
    seen = {}
    for r in rows:                       # later run of the same name wins
        if not r["name"].startswith("smoke"):
            seen[r["name"]] = r
    uniq = sorted(seen.values(), key=lambda r: r["ce_clean"])

    out = [HEAD, "", "## Seed replicates", "",
           "Six seeds of each finalist config, plus an early one, to separate real",
           "differences from run-to-run noise. `ens_*` is the logit-mean over those seeds.", "",
           "| config | view | seeds | val CE mean | sd | min | max | seed-ensemble CE |",
           "|---|---|---|---|---|---|---|---|"]
    groups = defaultdict(list)
    for r in uniq:
        mt = SEED_RE.match(r["name"])
        if mt:
            groups[mt.group(1)].append(r)
    for g, rs in sorted(groups.items(), key=lambda kv: np.mean([r["ce_clean"] for r in kv[1]])):
        if len(rs) < 2:
            continue
        ce = [r["ce_clean"] for r in rs]
        ens = seen.get(f"ens_{g}")
        out.append(f"| `{g}` | `{rs[0]['config'].get('view','?')}` | {len(rs)} | "
                   f"**{np.mean(ce):.5f}** | {np.std(ce):.5f} | {min(ce):.5f} | {max(ce):.5f} | "
                   f"{ens['ce_clean']:.5f}" + (f" (`ens_{g}`)" if ens else "-") + " |")

    out += ["", "## All runs", "",
            "| # | model | family | config | val CE (clean) | val CE (all) | ROC-AUC | PR-AUC | train s |",
            "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(uniq, 1):
        out.append(f"| {i} | `{r['name']}` | {r['family']} | {short_cfg(r)} | "
                   f"**{r['ce_clean']:.5f}** | {r['ce_all']:.5f} | {r['roc_auc']:.4f} | "
                   f"{r['pr_auc']:.4f} | {r['train_s']:.0f} |")
    out.append("")
    txt = "\n".join(out) + "\n"
    path = sys.argv[1] if len(sys.argv) > 1 else f"{C.OUT}/leaderboard.md"
    open(path, "w").write(txt)
    print(f"wrote {path} ({len(uniq)} rows, {len(groups)} seed groups)")


if __name__ == "__main__":
    main()
