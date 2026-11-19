# rlpolicy: an RL screening policy for the CAMELYON16 cascade

Goal: replace the supervised cascade's decisions (which units to read at L3, which to segment at L2, which
to paint as core) with a policy trained by reinforcement learning that is faster and at least as accurate,
keeping the rest of the cascade (L6 pass, L2 token-head segmenter, border tracing).

Runs on vivi (and prince) next to `experiments/rlcascade`, `experiments/rlprobe`, `experiments/rlscreen`
and `experiments/lowsep`, which it imports; those live on the server tree, not in this branch.

## What is here

| file | what it does |
|---|---|
| `where.py` | paths (override the output root with `RLPOLICY_ROOT`) |
| `tok3_data.py`, `tok3_model.py`, `tok3_train.py`, `tok3_predict.py`, `tok3_eval.py` | token-level L3 head: one tumour logit per L3 token of a unit, trained on the cached L3 tokens with dense labels, five fold heads plus an all-train head, out-of-fold predictions for every train slide |
| `tok6.py` | the same idea at L6 with context (kept for the record: not safe as a veto) |
| `simtab.py`, `sim.py`, `check_sim.py` | the fast cascade simulator: per-component tables of the real segmenter's dense map, border tracing, cleaned Dice, FROC records and per-unit rewards for any routing, in milliseconds per slide; reproduces the supervised rule's exact scores |
| `rl3.py` | **the RL agent**: PPO on the screen / zoom / paint decisions, small MLP policies whose state is the out-of-fold output of the frozen readers; cross-fitted |
| `policy.py`, `vsel.py`, `vsel6.py`, `search.py` | supervised reference: the consensus (veto) policy and its one-number fit; the staged policy and its cross-entropy search (overfits, kept for the record) |
| `qpol.py` | value-based variant (reward regression); unstable, kept for the record |
| `exact.py`, `evalx.py` | exact-scorer tables and paired bootstrap for saved policies; logs every test read |
| `run_staged.py` | real timed run of a policy in the rule's own harness |
| `diag.py` | where the rule spends its units; slide-level separability of the L3 scores |

## Order of a full run

1. `tok3_data` → `tok3_train --fold k` for k in 0..4 and -1 → `tok3_predict` → `tok3_eval`
2. `simtab --split train`, `simtab --split test`, `check_sim` (must reproduce the rule)
3. `rl3 --tok3 TAG --out NAME --lams ...` (cross-fitted PPO; prints one cross-fitted row per lambda)
4. `evalx --split train --runs NAME@lamX` (exact scorer, cross-fitted) and, once, `evalx --split test`
5. `run_staged --policy NAME@lamX` on an idle GPU for wall time

Selection is on cross-fitted train only: the fastest setting whose cross-fitted Dice and FROC are not below
the supervised rule's. The test set is read once per selected policy (`TEST_READS.txt`).
