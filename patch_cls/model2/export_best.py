# Packages the chosen Model-2 into a standalone directory that predict.py can run from.
#
# Writes one model.pt holding every ensemble member (weights + view string + the per-dim
# standardisation mu/sd measured on that member's training rows), copies the two modules
# predict.py imports, and drops a card with the metrics and the input contract.

import argparse
import json
import os
import shutil

import numpy as np
import torch

import common as C

BEST = f"{C.OUT}/best"
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--label", required=True, help="leaderboard row name of the ensemble")
    a = ap.parse_args()
    os.makedirs(BEST, exist_ok=True)

    members = []
    for n in a.members:
        ck = torch.load(f"{C.CKPT}/{n}.pt", map_location="cpu", weights_only=False)
        members.append(dict(name=n, state=ck["state"], mu=ck["mu"], sd=ck["sd"],
                            view=ck["view"], cfg=ck["cfg"]))
    torch.save(dict(members=members, label=a.label), f"{BEST}/model.pt")
    np.savez(f"{BEST}/standardization.npz",
             **{f"{m['name']}_mu": m["mu"].numpy() for m in members},
             **{f"{m['name']}_sd": m["sd"].numpy() for m in members})

    for f in ("predict.py", "views.py", "context.py"):
        shutil.copy(f"{HERE}/{f}", f"{BEST}/{f}")

    row = next((r for r in reversed(C.read_lb()) if r["name"] == a.label), None)
    # the card advertises `row`'s validation metrics, so the exported weights had better be
    # the same members that produced them
    logged = (row or {}).get("extra", {}).get("members")
    if logged is not None and sorted(logged) != sorted(a.members):
        raise SystemExit(f"--members {sorted(a.members)} != the members logged for "
                         f"{a.label}: {sorted(logged)}")
    card = dict(label=a.label, members=a.members, view=members[0]["view"],
                cfg=members[0]["cfg"], val_metrics=row,
                input_contract="state (N,384) + coords (N,2) level-0 px on a 2048 grid + "
                               "visited tile-index list -> (N,) P(tile_frac >= 0.25)",
                standardization="per-dim mean/sd of the assembled view over that member's "
                                "training rows; stored per member in model.pt as mu/sd and "
                                "mirrored in standardization.npz")
    with open(f"{BEST}/model_card.json", "w") as f:
        json.dump(card, f, indent=1)
    print(f"exported {len(members)} member(s) -> {BEST}")


if __name__ == "__main__":
    main()
