# Inference artifact for Model-1: 384-d ViT-S sub-patch features -> calibrated P(cancer).
#
# Load a bundle directory written by export_best.py and call predict_slide() with a
# slide's (N,16,384) feature tensor. The chosen model conditions on context that is
# derived from the slide itself -- the mean of the parent tile's 16 sub-features, the
# mean over the whole slide, and optionally the mean over the 3x3 tile neighbourhood --
# so the natural inference unit is one slide, not one loose feature vector. Everything
# needed (per-dimension standardisation constants, the prior-shift offset, the view
# spec and the ensemble weights) lives inside the bundle.
#
# Usage:
#     from predict import Model1
#     m = Model1.from_dir("/path/to/model1/best")
#     p = m.predict_slide(feat, coords)          # feat (N,16,384) -> p (N,16)
#     p = m.predict(X, tile_mean=..., slide_mean=..., nbr_mean=...)   # (M,384) -> (M,)

import json
import os

import numpy as np
import torch

import views as V


class Model1:
    def __init__(self, members, calib=None, device="cpu"):
        self.members = members
        self.calib = calib          # (A, B) applied to the averaged logit, or None
        self.device = torch.device(device)
        for m in members:
            m["model"].to(self.device).eval()

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_dir(cls, path, device="cpu"):
        man = json.load(open(os.path.join(path, "manifest.json")))
        members = []
        for e in man["members"]:
            ck = torch.load(os.path.join(path, e["file"]), map_location="cpu", weights_only=False)
            net = V.MLP(V.view_dim(ck["view"]), ck["cfg"].get("hidden", [512, 256]),
                        ck["cfg"].get("dropout", 0.0), ck["cfg"].get("bn", True))
            net.load_state_dict(ck["state"])
            members.append(dict(model=net, mu=ck["mu"], sd=ck["sd"], shift=float(ck["shift"]),
                                view=ck["view"], weight=float(e.get("weight", 1.0))))
        return cls(members, man.get("calib"), device)

    @property
    def needs_neighbours(self):
        return any("n" in V.parse_view(m["view"])[1] or
                   any(t in ("xn", "nc", "ns") for t in V.parse_view(m["view"])[1])
                   for m in self.members)

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def logit(self, X, tile_mean, slide_mean, nbr_mean, nbr2_mean, nbr4_mean, bs=200_000):
        """All args (M,384). Returns the ensemble logit (M,)."""
        t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32)
        X, Cm, S, Nb, Mb, Kb = (t(a) for a in (X, tile_mean, slide_mean, nbr_mean, nbr2_mean, nbr4_mean))
        out = torch.zeros(X.shape[0], dtype=torch.float64)
        wsum = sum(m["weight"] for m in self.members)
        for i in range(0, X.shape[0], bs):
            sl = slice(i, i + bs)
            x, c, s, nb, mb, kb = (a[sl].to(self.device) for a in (X, Cm, S, Nb, Mb, Kb))
            acc = torch.zeros(x.shape[0], dtype=torch.float64, device=self.device)
            for m in self.members:
                v = V.make_view_t(x, c, s, nb, mb, kb, m["view"])
                v = (v - m["mu"].to(self.device)) / m["sd"].to(self.device)
                acc += (m["model"](v).double() + m["shift"]) * m["weight"]
            out[sl] = (acc / wsum).cpu()
        if self.calib:
            out = self.calib[0] * out + self.calib[1]
        return out.numpy()

    def predict(self, X, tile_mean=None, slide_mean=None, nbr_mean=None, nbr2_mean=None,
                nbr4_mean=None):
        """(M,384) -> (M,) probabilities.

        With no context supplied, X is assumed to be exactly one slide's sub-patches in
        tile-major order (16 consecutive rows per 2048px tile), which is the layout of the
        cached feature dumps; tile and slide means are then derived from X itself. The
        neighbourhood mean falls back to the tile mean, so pass nbr_mean (or use
        predict_slide with coords) if the bundle's view uses the `n` block.
        """
        X = np.asarray(X, dtype=np.float32)
        if tile_mean is None:
            assert X.shape[0] % 16 == 0, "supply tile_mean, or pass whole tiles (M %% 16 == 0)"
            tile_mean = np.repeat(X.reshape(-1, 16, 384).mean(1), 16, axis=0)
        if slide_mean is None:
            slide_mean = np.repeat(X.mean(0, keepdims=True), X.shape[0], axis=0)
        if nbr_mean is None:
            nbr_mean = tile_mean
        if nbr2_mean is None:
            nbr2_mean = nbr_mean
        if nbr4_mean is None:
            nbr4_mean = nbr2_mean
        z = self.logit(X, tile_mean, slide_mean, nbr_mean, nbr2_mean, nbr4_mean)
        return 1.0 / (1.0 + np.exp(-z))

    def predict_slide(self, feat, coords=None, tile_px=2048, radius=1):
        """feat (N,16,384) for one slide -> (N,16) probabilities."""
        f = np.asarray(feat, dtype=np.float32)
        n = f.shape[0]
        tmean = f.mean(axis=1)
        X = f.reshape(-1, 384)
        Cm = np.repeat(tmean, 16, axis=0)
        S = np.repeat(X.mean(0, keepdims=True), n * 16, axis=0)
        if coords is not None:
            co = np.asarray(coords)
            nmean = neighbour_mean(tmean, co, tile_px, radius)
            mmean = neighbour_mean(tmean, co, tile_px, radius * 2)
            kmean = neighbour_mean(tmean, co, tile_px, radius * 4)
        else:
            nmean = mmean = kmean = tmean
        Nb, Mb, Kb = (np.repeat(a, 16, axis=0) for a in (nmean, mmean, kmean))
        return (1.0 / (1.0 + np.exp(-self.logit(X, Cm, S, Nb, Mb, Kb)))).reshape(n, 16)


def neighbour_mean(tile_mean, coords, tile_px=2048, radius=1):
    """Mean tile feature over the (2r+1)^2 tile block around each tile."""
    g = np.rint(coords.astype(np.float64) / tile_px).astype(np.int64)
    cell = {}
    for i, (gx, gy) in enumerate(g):
        cell.setdefault((gx, gy), []).append(i)
    out = np.empty_like(tile_mean)
    for i, (gx, gy) in enumerate(g):
        idx = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                idx.extend(cell.get((gx + dx, gy + dy), ()))
        out[i] = tile_mean[idx].mean(axis=0)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="score one slide .pt feature dump")
    ap.add_argument("bundle")
    ap.add_argument("slide_pt")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    d = torch.load(a.slide_pt, weights_only=False)
    f = d["feat"].numpy() if isinstance(d["feat"], torch.Tensor) else np.asarray(d["feat"])
    c = d["coords"].numpy() if isinstance(d["coords"], torch.Tensor) else np.asarray(d["coords"])
    p = Model1.from_dir(a.bundle, a.device).predict_slide(f, c)
    print(f"{p.shape} mean={p.mean():.5f} max={p.max():.5f} frac>0.5={(p > 0.5).mean():.5f}")
