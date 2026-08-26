# The feature-view grammar and the MLP definition, shared by training and inference.
#
# A view names which blocks get concatenated into the input vector. The blocks are built
# from three 384-d vectors: x, the sub-patch feature; c, the mean over the 16 sub-patches
# of its parent tile; s, the mean over every sub-patch of its slide. Both c and s are
# computable at inference time from the tile/slide alone, so nothing here needs labels.
#
#   x  c  s      the vectors themselves
#   xc xs cs     the pairwise differences (x-c, x-s, c-s)
#   "L2:" prefix L2-normalises x, c and s before any block is formed
#
# e.g. "L2:x+c+s" -> 1152-d, "x+xc" -> 768-d.

import numpy as np
import torch
import torch.nn as nn

EPS = 1e-7
ALIAS = {"raw": "x", "l2": "L2:x", "ctx": "x+c", "l2ctx": "L2:x+c",
         "diff": "x+xc", "ctxd": "x+c+xc"}
BLOCKS = ("x", "c", "s", "n", "m", "k", "xc", "xs", "cs", "xn", "xm", "xk",
          "nc", "ns", "mn", "ms", "km", "ks")


def parse_view(mode):
    mode = ALIAS.get(mode, mode)
    l2 = mode.startswith("L2:")
    toks = (mode[3:] if l2 else mode).split("+")
    for t in toks:
        if t not in BLOCKS:
            raise ValueError(f"bad view block {t!r} in {mode!r}")
    return l2, toks


def view_dim(mode):
    return 384 * len(parse_view(mode)[1])


def blocks(x, c, s, nb, mb, kb, toks, norm):
    if norm:
        x, c, s, nb, mb, kb = (norm(a) for a in (x, c, s, nb, mb, kb))
    d = {"x": x, "c": c, "s": s, "n": nb, "m": mb, "k": kb,
         "xc": x - c, "xs": x - s, "cs": c - s, "xn": x - nb, "xm": x - mb, "xk": x - kb,
         "nc": nb - c, "ns": nb - s, "mn": mb - nb, "ms": mb - s, "km": kb - mb, "ks": kb - s}
    return [d[t] for t in toks]


def make_view_np(X, Cm, S, Nb, Mb, Kb, mode):
    l2, toks = parse_view(mode)
    f = (lambda a: a / (np.linalg.norm(a, axis=1, keepdims=True) + EPS)) if l2 else None
    a32 = lambda z: z.astype(np.float32)
    parts = blocks(a32(X), a32(Cm), a32(S), a32(Nb), a32(Mb), a32(Kb), toks, f)
    return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)


def make_view_t(x, c, s, nb, mb, kb, mode):
    l2, toks = parse_view(mode)
    f = (lambda a: a / (a.norm(dim=1, keepdim=True) + EPS)) if l2 else None
    parts = blocks(x, c, s, nb, mb, kb, toks, f)
    return parts[0] if len(parts) == 1 else torch.cat(parts, 1)


class MLP(nn.Module):
    def __init__(self, d_in, hidden, dropout, bn):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers.append(nn.Linear(d, h))
            if bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def tile_and_slide_means(feat):
    """feat: (N,16,384) -> per-sub-patch tile mean and slide mean, both (N*16,384)."""
    f = feat.astype(np.float32) if isinstance(feat, np.ndarray) else feat.float()
    n = f.shape[0]
    if isinstance(f, np.ndarray):
        c = np.repeat(f.mean(axis=1, keepdims=True), 16, axis=1).reshape(-1, 384)
        s = np.repeat(f.reshape(-1, 384).mean(axis=0, keepdims=True), n * 16, axis=0)
    else:
        c = f.mean(dim=1, keepdim=True).expand(-1, 16, -1).reshape(-1, 384)
        s = f.reshape(-1, 384).mean(dim=0, keepdim=True).expand(n * 16, -1)
    return f.reshape(-1, 384), c, s
