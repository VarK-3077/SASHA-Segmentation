# Renders the Model-2 markdown leaderboard from the JSONL log plus the seed summary.
#
# Three tables: the two-way transfer matrix that answers the (a)-vs-(b) training
# distribution question, the seed-replicate table, and every run ever logged.

import json
import os
import re

import numpy as np

import common as C

HDR = """# Model-2 leaderboard - per-tile cancer classifier on SASHA rollout states

Binary classifier over 2048px level-0 CAMELYON16 tiles. At inference the input is the
tile's 384-d vector in the **final state of a SASHA-0.2 deterministic rollout**: a true
phase-1 aggregate for the ~20% of tiles the RL agent visited, a TSU-updated pseudo-feature
for cosine-similar tiles, and the untouched low-res feature everywhere else.

**Selection metric `val CE (state)`** - unweighted cross-entropy over the validation tiles
the clean label policy keeps, on deterministic rollout-state inputs. That is the deployed
input distribution, so it is the only metric anything is chosen by.
**`val CE (agg)`** is the same model scored on true phase-1 aggregate inputs; it is
reported so the transfer in both directions is visible, and never selected on.
`val CE (all)` adds back the 0 < tile_frac < 0.25 boundary tiles under the >= 0.25 rule.

## Data and label policy

Positive `tile_frac >= 0.25`, negative `tile_frac == 0` excluding tiles of the 20
non-exhaustively annotated tumour slides, the strip in between dropped from training.

| | slides | tiles | positives | negatives | boundary dropped | untrusted zeros dropped | pos rate |
|---|---|---|---|---|---|---|---|
| train | 242 | 188,110 | 4,608 | 165,350 | 2,196 | 15,956 | 2.711% |
| val | 27 | 17,927 | 644 | 17,057 | 226 | 0 | 3.638% |
| test | 129 | 108,791 | 4,558 | 103,004 | 1,229 | 0 | 4.238% |

All 20 non-exhaustive slides fall in train, so val and test lose nothing. Each rollout of
a train slide yields 169,958 clean rows, so the two stochastic rollouts add 339,916 more
and the widest training pool used here is 679,832 rows.

## Training row sources

| tag | rows | what the tile vector is | what the context pyramid pools over |
|---|---|---|---|
| `det` | 169,958 | deterministic rollout final state | the same state matrix |
| `aug` | 339,916 | the 2 stochastic rollouts' final states | each rollout's own state matrix |
| `agg` | 169,958 | true phase-1 aggregate from the h5 | the aggregate matrix |

A visited tile's rollout state is bit-identical to its true aggregate (verified: cosine
1.0 on every visited index), so an all-aggregate matrix is exactly the state a rollout
that visited everything would leave. That is why `agg` rows carry `visited = 1` for every
tile, and why the `v` block is degenerate when training on `agg` alone.

## Feature views

Blocks concatenated into a tile's input vector: `x` the tile vector, `n`/`m`/`k` the mean
tile vector over the 3x3 / 5x5 / 9x9 tile neighbourhood on the 2048px grid, `s` the slide
mean, `xn`/`xm`/`xk`/`ns`/`mk`/`ks` differences of those, `v` a 4-d aux block
`[visited, visited-fraction in 3x3, in 5x5, in 9x9]`, `q` a 5-d block of the raw block
norms `[||x||,||n||,||m||,||k||,||s||]` that an `L2:` view would otherwise discard.
`L2:` L2-normalises `x,n,m,k,s`
before blocks are formed. Every block is computable at inference from the state matrix,
the coords and the visited list. Neighbourhood means are always pooled over the same
feature matrix the tile vector came from.

Rows suffixed `_VALFIT` are selected or fitted on validation itself and are diagnostics,
not candidates.
"""

MATRIX_DOC = {
    "E0_det_nctx": ("(b) det only, NO spatial context", "ablation"),
    "E1_det": ("(b) rollout states, deterministic only", "1"),
    "E2_detaug": ("(b+) rollout states + 2 stochastic rollouts", "2"),
    "E3_agg": ("(a) true aggregates", "3"),
    "E4_ft": ("(c) pretrain on (a), fine-tune on (b+)", "4"),
    "E5_union": ("(d) union of (a) and (b+)", "5"),
    "E6v_det": ("(e) (b) + visited block", "6"),
    "E6v_detaug": ("(e) (b+) + visited block", "6"),
    "E6v_union": ("(e) (d) + visited block", "6"),
    "Z3_det": ("(b) rollout states, deterministic only", "1"),
    "Z1_detaug": ("(b+) rollout states + 2 stochastic rollouts", "2"),
    "Z5_agg": ("(a) true aggregates", "3"),
    "Z6_ft": ("(c) pretrain on (a), fine-tune on (b+)", "4"),
    "Z0_union": ("(d) union of (a) and (b+)", "5"),
    "Z7_union_v": ("(e) (d) + visited block", "6"),
}
MATRIX_DOC.update({
    "W6_lin_det": ("(b) rollout states, deterministic only", "1"),
    "W5_lin_detaug": ("(b+) rollout states + 2 stochastic rollouts", "2"),
    "W7_lin_agg": ("(a) true aggregates", "3"),
    "W0_lin_union": ("(d) union of (a) and (b+)", "5"),
    "W3_lin_union_v": ("(e) (d) + visited block", "6"),
})
E_ORDER = ["E1_det", "E2_detaug", "E3_agg", "E4_ft", "E5_union",
           "E6v_det", "E6v_detaug", "E6v_union", "E0_det_nctx"]
Z_ORDER = ["Z3_det", "Z1_detaug", "Z5_agg", "Z6_ft", "Z0_union", "Z7_union_v"]
W_ORDER = ["W6_lin_det", "W5_lin_detaug", "W7_lin_agg", "W0_lin_union", "W3_lin_union_v"]
ABL_DOC = {0.1: "loose", 0.25: "the training policy", 0.5: "strict", 0.75: "very strict"}


def fam(name):
    return re.sub(r"_s\d+$", "", name)


def group(rows):
    g = {}
    for r in rows:
        g.setdefault(fam(r["name"]), []).append(r)
    return g


def transfer_table(groups, ens, order, title, note):
    L = [f"### {title}", "", note, "",
         "| # | training rows | seeds | val CE on **rollout states** (mean +- sd) | "
         "val CE on **true aggregates** | seed-ens state CE | ROC-AUC | PR-AUC |",
         "|---|---|---|---|---|---|---|---|"]
    for k in order:
        if k not in groups:
            continue
        g = groups[k]
        ce = np.array([r["ce"] for r in g])
        ag = np.array([r["agg_ce"] for r in g])
        ro = np.array([r["roc_auc"] for r in g])
        pr = np.array([r["pr_auc"] for r in g])
        e = ens.get(f"ens_{k}")
        desc, tag = MATRIX_DOC.get(k, (k, ""))
        sd = ce.std(ddof=1) if len(ce) > 1 else 0.0
        asd = ag.std(ddof=1) if len(ag) > 1 else 0.0
        L.append(f"| {tag} | {desc} | {len(ce)} | **{ce.mean():.5f}** +- {sd:.5f} | "
                 f"{ag.mean():.5f} +- {asd:.5f} | "
                 f"{e['ce']:.5f} | {ro.mean():.4f} | {pr.mean():.4f} |"
                 if e else
                 f"| {tag} | {desc} | {len(ce)} | **{ce.mean():.5f}** +- {sd:.5f} | "
                 f"{ag.mean():.5f} +- {asd:.5f} | - | {ro.mean():.4f} | {pr.mean():.4f} |")
    return L


def seed_table(groups, ens):
    L = ["## Seed replicates", "",
         "Every config with more than one seed, ranked by multi-seed mean state CE - the",
         "statistic the final model was chosen by. `seed-ens` is the logit-mean over that",
         "config's seeds. Configs with only 2 seeds sit high in this table on a very noisy",
         "estimate; the chosen model comes from the 6-seed block.", "",
         "| config | rows | view | hidden | seeds | val CE (state) mean | sd | min | seed-ens | val CE (agg) mean |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    out = []
    for k, g in groups.items():
        if len(g) < 2:
            continue
        ce = np.array([r["ce"] for r in g])
        ag = np.array([r["agg_ce"] for r in g])
        out.append((ce.mean(), k, g[0]["config"], len(g), ce, ag))
    out.sort()
    for mean, k, cfg, n, ce, ag in out[:45]:
        cfg = cfg if isinstance(cfg, dict) else {}
        e = ens.get(f"ens_{k}")
        hid = cfg.get("hidden", "?")
        hid = "logistic" if hid == [] else str(hid)
        L.append(f"| `{k}` | `{'+'.join(cfg.get('rows', [])) or '?'}` | `{cfg.get('view', '?')}` | "
                 f"{hid} | {n} | **{mean:.5f}** | {ce.std(ddof=1):.5f} | {ce.min():.5f} | "
                 f"{e['ce']:.5f} | {ag.mean():.5f} |" if e else
                 f"| `{k}` | `{'+'.join(cfg.get('rows', [])) or '?'}` | `{cfg.get('view', '?')}` | "
                 f"{hid} | {n} | **{mean:.5f}** | {ce.std(ddof=1):.5f} | {ce.min():.5f} | - | "
                 f"{ag.mean():.5f} |")
    return L


CHOSEN = "ens_W0_lin_union"


def chosen_section(ens, groups):
    r = ens.get(CHOSEN)
    if not r:
        return []
    g = groups.get(CHOSEN[4:], [])
    ce = np.array([x["ce"] for x in g]) if g else np.array([np.nan])
    cfg = g[0]["config"] if g else {}
    L = ["## Chosen model", "",
         f"`{CHOSEN}` - the logit-mean of {len(g)} seeds of `{CHOSEN[4:]}`, the config with the",
         "best multi-seed mean state CE. Logistic regression on the multi-scale context",
         "pyramid, trained on the union of true aggregates and all three rollout states.", "",
         f"    view      {cfg.get('view')}   ({1536} d: x, 3x3, 5x5, 9x9, each L2-normalised)",
         f"    rows      {'+'.join(cfg.get('rows', []))}   ({r.get('n_rows') or 679832:,} rows before the label policy)",
         f"    optimiser AdamW lr {cfg.get('lr')} wd {cfg.get('wd')}, batch {cfg.get('bs')}, "
         f"{cfg.get('epochs')} epochs cosine, bf16",
         "    output bias initialised at the training base rate",
         "", "| | val CE (state) | val CE (agg) | ROC-AUC | PR-AUC |", "|---|---|---|---|---|",
         f"| single seed (mean of {len(g)}) | {ce.mean():.5f} +- {ce.std(ddof=1):.5f} | - | - | - |",
         f"| **seed ensemble** | **{r['ce']:.5f}** | {r['agg_ce']:.5f} | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} |",
         ""]
    for tag, path in (("val", "val_metrics.json"), ("test", "test_metrics.json")):
        p = f"{C.OUT}/{path}"
        if not os.path.exists(p):
            continue
        m = json.load(open(p))
        t, s = m["tile"], m["slide"]
        if tag == "val":
            L += ["Slide-level sanity on val (diagnostic):"]
        else:
            L += ["", "**Test set, read once, nothing tuned on it:**"]
        L += ["",
              f"- tiles: CE {t['ce']:.5f}, ROC-AUC {t['roc_auc']:.4f}, PR-AUC {t['pr_auc']:.4f} "
              f"over {t['n_clean']:,} clean tiles ({t['pos_rate']:.2%} positive)",
              f"- slides: mean tile probability {s['mean_tile_prob_tumor']:.4f} on the "
              f"{s['n_tumor']} tumour slides vs {s['mean_tile_prob_normal']:.4f} on the "
              f"{s['n_normal']} normal ones",
              f"- slide ROC-AUC {s['slide_auc_max']:.4f} from the max tile probability, "
              f"{s['slide_auc_mean']:.4f} from the mean", ""]
    return L


def ablation_table(groups):
    L = ["## Positive-threshold ablation", "",
         "The winning config retrained under other positive thresholds. Each threshold is a",
         "different label policy - it changes which tiles are positive, which are dropped as",
         "boundary, and therefore what the CE is even measuring - so these numbers are",
         "**reported, not compared**, and nothing was selected across them. Thresholds below",
         "0.25 are absent on purpose: the cached training rows already had the 0-0.25 strip",
         "removed, so a looser policy could only be faked by relabelling rather than by",
         "actually training on the tiles it would admit.", "",
         "| tile_frac >= | | train rows | val clean tiles | val pos rate | val CE (state) mean +- sd | val CE (agg) | ROC-AUC |",
         "|---|---|---|---|---|---|---|---|"]
    rows = []
    for k, g in groups.items():
        if not k.startswith(("AB", "W0_lin_union")):
            continue
        thr = g[0].get("pos_thr", 0.25)
        if thr < 0.25:
            continue
        ce = np.array([r["ce"] for r in g])
        ag = np.array([r["agg_ce"] for r in g])
        ro = np.array([r["roc_auc"] for r in g])
        rows.append((thr, len(ce), ce, ag, ro, g[0]["n_rows"], g[0]["n_clean"], g[0]["pos_rate"]))
    for thr, n, ce, ag, ro, nr, nc, pr_ in sorted(rows):
        sd = ce.std(ddof=1) if n > 1 else 0.0
        L.append(f"| **{thr}** | {ABL_DOC.get(thr, '')} | {nr:,} | {nc:,} | {pr_:.2%} | "
                 f"**{ce.mean():.5f}** +- {sd:.5f} | {ag.mean():.5f} | {ro.mean():.4f} |")
    return L


def all_table(rows):
    L = ["## All runs", "",
         "Sorted by the selection metric. The very top of this table is not the ranking any",
         "decision was made on: single seeds and 2-seed ensembles float up on luck, which is",
         "exactly why the finalists were re-run at 6 seeds and chosen on the mean. The chosen",
         f"model `{CHOSEN}` sits mid-pack here and first among 6-seed candidates.", "",
         "| # | model | family | rows | view | val CE (state) | val CE (agg) | val CE (all) | ROC-AUC | PR-AUC | train s |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    by_name = {r["name"]: r for r in rows}
    for i, r in enumerate(sorted(rows, key=lambda x: x["ce"]), 1):
        cfg = r["config"] if isinstance(r["config"], dict) else {}
        if not cfg and r.get("extra", {}).get("members"):
            m = by_name.get(r["extra"]["members"][0], {})
            cfg = m.get("config", {}) if isinstance(m.get("config"), dict) else {}
        fam_ = r["family"]
        if fam_ == "mlp":
            fam_ = "logistic" if cfg.get("hidden") == [] else "mlp"
        rr = "+".join(cfg.get("rows", [])) or "-"
        L.append(f"| {i} | `{r['name']}` | {fam_} | `{rr}` | `{cfg.get('view', '-')}` | "
                 f"**{r['ce']:.5f}** | {r['agg_ce']:.5f} | {r['ce_all']:.5f} | "
                 f"{r['roc_auc']:.4f} | {r['pr_auc']:.4f} | {r['train_s']:.0f} |")
    return L


def caveats():
    L = ["## Caveats", ""]
    p = f"{C.OUT}/val_metrics.json"
    if os.path.exists(p):
        ps = json.load(open(p))["per_slide"]
        tot = sum(r["ce_share"] for r in ps)
        pos = [r for r in ps if r["pos"] > 0]
        cum, top = 0.0, []
        for r in ps[:3]:
            cum += r["ce_share"]
            top.append(f"`{r['slide']}` {100 * r['ce_share'] / tot:.0f}%")
        L += [f"- **The selection metric rests on a handful of slides.** Only {len(pos)} of the 27",
              "  validation slides contain a single positive tile, and the pooled validation CE is",
              f"  dominated by {', '.join(top)} - {100 * cum / tot:.0f}% of the total between them.",
              "  Differences below roughly 0.0005 CE are one slide changing its mind, and the",
              "  seed-to-seed sd of the finalists (~0.0007) is the same size as the gap between the",
              "  top few configs. The 6-seed mean, not a single run, is what the choice rested on."]
    L += [
        "- **The validation number carries selection optimism.** Every run early-stopped on",
        "  val, and the winner was picked from roughly 300 configs scored on the same 27",
        "  slides, so 0.00634 is a best-of-many and is biased low as an estimate of what this",
        "  model does on unseen slides. The test number is the unbiased one, and it was",
        "  computed once, after the model was frozen.",
        "- **One defect was found and fixed late.** The two stochastic-rollout augmentation",
        "  rows shared a single slide-mean vector per slide instead of one per rollout, so a",
        "  second-rollout row's `s` block came from the first rollout's state matrix. Only",
        "  views containing `s` (directly or via `ns`/`ms`/`ks`) were affected; the caches were",
        "  rebuilt and all 26 such configs re-run. Two seeds of the chosen config were re-run",
        "  on the rebuilt caches and reproduced their logged CE to 8 decimal places, which is",
        "  the evidence that nothing outside the `s` views moved.",
        "- **Val CE is optimistic relative to test.** The chosen model scores 0.00634 on val and",
        "  0.01649 on test, a 2.6x gap that is mostly composition: test has 49 tumour slides",
        "  against val's 9, and a 4.24% positive rate against val's 3.64%.",
        "- **Aggregate-input CE is a diagnostic, not a second objective.** For an all-aggregate",
        "  input matrix every tile is by construction what a visited tile's state would be, so",
        "  those runs are scored with `visited = 1` everywhere. A model that leans on the `v`",
        "  block therefore sees an input pattern it never met in training, which is part of why",
        "  the aggregate column is noisier than the state column.",
        "- **The GBDT rows had a small unfair advantage and still lost.** LightGBM early-stopped",
        "  on the clean validation set itself, so its numbers are mildly optimistic; its CE is",
        "  still 2-3x the MLP's at comparable ROC-AUC, reproducing the Model-1 finding that",
        "  GBDT probabilities are badly over-confident.",
        "- **The probability dumps are float16 and clamped.** fp16 rounds anything above",
        "  1 - 2^-11 to exactly 1.0, which 2.5% of test tiles hit; the stored values are",
        "  clamped into (5.96e-8, 0.999512) so a downstream log-loss cannot see an infinity.",
        "  Every metric in this file is computed on the float64 probabilities, before the cast.",
        "- **`p_tumor` was deliberately not used.** The rollout dump carries SASHA's own",
        "  slide-level tumour probability, which would sharply cut the CE by letting the model",
        "  zero out normal slides wholesale. It is outside the stated input contract and would",
        "  measure the slide classifier rather than tile localisation, so no run uses it.",
        "",
        "## Paths (server)", "",
        "```",
        f"{C.OUT}/leaderboard.md          this file",
        f"{C.OUT}/leaderboard.jsonl       one json row per run",
        f"{C.OUT}/best/model.pt           the 6 ensemble members + their mu/sd",
        f"{C.OUT}/best/predict.py         self-contained inference",
        f"{C.OUT}/best/model_card.json    input contract and val metrics",
        f"{C.OUT}/test_probs.pkl          " + "{slide: (N,) float16} on test rollout states",
        f"{C.OUT}/val_probs.pkl           the same on val",
        f"{C.OUT}/cache/                  the flat per-tile arrays every run reads",
        "~/SASHA-Segmentation/patch_cls/model2/   the scripts",
        "```"]
    return L


def main():
    rows = C.read_lb()
    seen = {}
    for r in rows:
        seen[r["name"]] = r                       # last write wins
    rows = list(seen.values())
    # CHK_* are deliberate re-runs of an already-logged config, used to prove a cache
    # rebuild changed nothing; they are duplicates and must not enter any ranking
    rows = [r for r in rows if not r["name"].startswith("CHK_")]
    trained = [r for r in rows if r["family"] in ("mlp", "lgb")]
    ens = {r["name"]: r for r in rows if r["family"] == "ensemble"}
    groups = group(trained)
    # a run under a non-default positive threshold measures a different quantity, so it is
    # kept out of every ranking table and shown only in the ablation section
    std = lambda rs: [r for r in rs if r.get("pos_thr", 0.25) == 0.25]
    groups_std = {k: v for k, v in ((k, std(v)) for k, v in groups.items()) if v}

    L = [HDR]
    L += chosen_section(ens, groups_std)
    L += ["## The training-distribution question: two-way transfer", "",
         "Each row is one training distribution. Both columns score the *same* trained",
         "model; only the validation input representation changes. Selection never looks at",
         "the aggregate column.", ""]
    L += transfer_table(
        groups_std, ens, W_ORDER, "In the winning model family (6 seeds)",
        "`L2:x+n+m+k`, logistic regression, lr 3e-2 wd 3e-2, prior-initialised output bias "
        "- the configuration the sweeps ended on and the one the final model uses. Variant "
        "(c) has no entry here: fine-tuning a pretrained model is an MLP manoeuvre and it "
        "already lost in the MLP table below.") + [""]
    L += transfer_table(
        groups_std, ens, Z_ORDER, "At the tuned MLP architecture (6 seeds)",
        "`L2:x+n+m+k`, 2048-512 MLP, dropout 0.1, unweighted BCE.") + [""]
    L += transfer_table(
        groups_std, ens, E_ORDER, "At the starting architecture (3 seeds)",
        "`L2:x+n+m+k`, 1024-512 MLP, dropout 0.3, unweighted BCE - Model-1's winner, "
        "carried over unchanged so no distribution got an architecture advantage in the "
        "first pass.") + [""]
    L += ablation_table(groups) + [""]
    L += seed_table(groups_std, ens) + [""]
    L += all_table(std(rows)) + [""]
    L += caveats() + [""]
    txt = "\n".join(L)
    with open(f"{C.OUT}/leaderboard.md", "w") as f:
        f.write(txt)
    print(f"wrote {C.OUT}/leaderboard.md  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
