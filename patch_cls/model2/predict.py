# Self-contained inference for the exported Model-2 tile classifier.
#
# Input contract, exactly what a SASHA-0.2 rollout leaves behind for one slide:
#   state   (N,384) float  the final rollout state matrix (fp16 or fp32)
#   coords  (N,2)   int    level-0 pixel coords of each 2048px tile, same row order
#   visited list/array of tile indices the agent visited during the rollout
# Output:
#   (N,) float64 P(tile is cancer), tumour fraction >= 0.25 under the training label policy.
#
# Everything the model needs beyond the state matrix -- the 3x3 / 5x5 / 9x9 neighbourhood
# means, the slide mean, the visited aux block -- is recomputed here from those three
# inputs, pooled over the state matrix itself. The standardisation constants (per-dim mean
# and sd of the assembled view, measured on the training rows) travel inside the checkpoint
# as `mu` and `sd`, and are also written out as standardization.npz next to it.
#
#   python predict.py --ckpt best/model.pt --states rollout_states.pkl --out probs.pkl

import argparse
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import context as CX
import views as V


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    members = ck["members"] if "members" in ck else [ck]
    nets = []
    for m in members:
        cfg = m["cfg"]
        net = V.MLP(V.view_dim(m["view"]), cfg.get("hidden", [1024, 512]),
                    cfg.get("dropout", 0.3), cfg.get("bn", True)).to(device)
        net.load_state_dict(m["state"])
        net.eval()
        nets.append(dict(net=net, view=m["view"], mu=m["mu"].to(device).float(),
                         sd=m["sd"].to(device).float()))
    return nets


@torch.no_grad()
def predict_slide(nets, state, coords, visited, device="cpu"):
    f = np.asarray(state, dtype=np.float32)
    means, smean, aux = CX.pyramid(f, coords, visited)
    s = np.repeat(smean[None, :], len(f), axis=0)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)
    x, n, m, k, sm, v = t(f), t(means[0]), t(means[1]), t(means[2]), t(s), t(aux)
    zs = []
    for e in nets:
        view = V.make_view_t(x, n, m, k, sm, v, e["view"])
        zs.append(e["net"]((view - e["mu"]) / e["sd"]).double())
    z = torch.stack(zs).mean(0)                      # logit-mean over ensemble members
    return torch.sigmoid(z).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pt"))
    ap.add_argument("--states", required=True, help="rollout_states.pkl")
    ap.add_argument("--coords_h5", default="/data2/tej/SASHA/camelyon_lr_features/patch_feats_pretrain_medical_ssl.h5")
    ap.add_argument("--split", default=None, help="only slides with this split tag")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import h5py
    nets = load_model(a.ckpt, a.device)
    with open(a.states, "rb") as fh:
        slides = pickle.load(fh)["slides"]
    out = {}
    with h5py.File(a.coords_h5, "r") as h5:
        for name in sorted(slides):
            e = slides[name]
            if a.split and e.get("split") != a.split:
                continue
            p = predict_slide(nets, e["state"], h5[name]["coords"][:], e["visited"], a.device)
            # fp16 rounds p > 1 - 2^-11 to exactly 1.0, which is an infinity to any
            # downstream log-loss; keep the stored value inside the open interval
            hi = float(np.nextafter(np.float16(1.0), np.float16(0.0)))
            out[name] = np.clip(p, 6e-8, hi).astype(np.float16)
    with open(a.out, "wb") as fh:
        pickle.dump(out, fh)
    print(f"wrote {len(out)} slides -> {a.out}")


if __name__ == "__main__":
    main()
