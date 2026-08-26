# Packages the chosen ensemble members into a self-contained bundle for predict.py,
# then re-scores the validation set through that bundle as a check that the artifact
# reproduces the leaderboard number rather than trusting the training-time value.

import argparse
import json
import os
import shutil

import numpy as np
import torch

import common as C
import predict as P

BEST = f"{C.OUT}/best"
CKPT = f"{C.OUT}/ckpt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--calib", nargs=2, type=float, default=None,
                    help="optional A B applied to the averaged logit")
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    os.makedirs(BEST, exist_ok=True)
    man = dict(members=[], calib=a.calib, note="logit-mean ensemble; see leaderboard.md")
    for i, n in enumerate(a.members):
        src = f"{CKPT}/{n}.pt"
        dst = f"member_{i}.pt"
        shutil.copy(src, f"{BEST}/{dst}")
        ck = torch.load(src, map_location="cpu", weights_only=False)
        man["members"].append(dict(file=dst, name=n, view=ck["view"], shift=ck["shift"],
                                   cfg=ck["cfg"], weight=1.0))
    json.dump(man, open(f"{BEST}/manifest.json", "w"), indent=1)
    for f in ("predict.py", "views.py"):
        shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), f), f"{BEST}/{f}")
    print(f"wrote bundle -> {BEST}")

    va = C.load_split_arrays("val")
    m = P.Model1.from_dir(BEST, a.device)
    z = m.logit(va["X"].astype(np.float32), va["C"].astype(np.float32),
                va["S"].astype(np.float32), va["N"].astype(np.float32),
                va["M"].astype(np.float32), va["K"].astype(np.float32))
    p = 1.0 / (1.0 + np.exp(-z))
    met = C.eval_val(p, va)
    print("bundle re-scored on val:", json.dumps(met, indent=1))
    np.save(f"{C.OUT}/preds/BUNDLE_val.npy", p.astype(np.float32))
    json.dump(met, open(f"{BEST}/val_metrics.json", "w"), indent=1)

    rows = C.per_slide_ce(p, va)
    json.dump(rows, open(f"{C.OUT}/val_per_slide_best.json", "w"), indent=1)
    print("\nworst val slides by CE:")
    for r in rows[:5]:
        print(f"  {r['slide']:>12}  n={r['n']:>6}  pos={r['pos']:>5}  "
              f"CE={r['ce']:.5f}  mean_p={r['mean_p']:.4f}")
    print("\nbest val slides by CE:")
    for r in rows[-3:]:
        print(f"  {r['slide']:>12}  n={r['n']:>6}  pos={r['pos']:>5}  "
              f"CE={r['ce']:.5f}  mean_p={r['mean_p']:.4f}")


if __name__ == "__main__":
    main()
