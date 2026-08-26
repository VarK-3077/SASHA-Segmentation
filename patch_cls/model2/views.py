# Feature-view grammar and the MLP, shared by Model-2 training and inference.
#
# A view names which blocks get concatenated into a tile's input vector. Blocks are built
# from five 384-d vectors: x, the tile's own feature; n/m/k, the mean tile feature over the
# 3x3 / 5x5 / 9x9 tile neighbourhood on the 2048px grid; s, the slide mean. Plus one 4-d
# aux block v = [visited, visited-fraction in 3x3, in 5x5, in 9x9].
#
# Every block is computable at inference time from the rollout state matrix, the coords and
# the visited list alone. Crucially the neighbourhood pool is taken over the SAME feature
# matrix the tile vector comes from -- rollout-state neighbours for a rollout-state model,
# aggregate neighbours for an aggregate model -- because that is all inference ever has.
#
#   x n m k s      the vectors themselves
#   xn xm xk xs    x minus a context vector
#   nm mk ks ns    context-scale differences
#   v              the 4-d visited aux block
#   "L2:" prefix   L2-normalises x,n,m,k,s before any block is formed
#
# e.g. "L2:x+n+m+k" -> 1536-d, "x+n+m+k+v" -> 1540-d.

import numpy as np
import torch
import torch.nn as nn

EPS = 1e-7
D = 384
VDIM = 4
VEC = ("x", "n", "m", "k", "s")
DIFF = {"xn": ("x", "n"), "xm": ("x", "m"), "xk": ("x", "k"), "xs": ("x", "s"),
        "nm": ("n", "m"), "mk": ("m", "k"), "ks": ("k", "s"), "ns": ("n", "s"),
        "ms": ("m", "s"), "nk": ("n", "k")}
QDIM = 5
BLOCKS = set(VEC) | set(DIFF) | {"v", "q"}
ALIAS = {"raw": "x", "l2": "L2:x", "ctx": "x+n+m+k", "l2ctx": "L2:x+n+m+k"}


def parse_view(mode):
    mode = ALIAS.get(mode, mode)
    l2 = mode.startswith("L2:")
    toks = (mode[3:] if l2 else mode).split("+")
    for t in toks:
        if t not in BLOCKS:
            raise ValueError(f"bad view block {t!r} in {mode!r}")
    return l2, toks


def view_dim(mode):
    toks = parse_view(mode)[1]
    sz = {"v": VDIM, "q": QDIM}
    return sum(sz.get(t, D) for t in toks)


def blocks(x, n, m, k, s, v, toks, norm, qb=None):
    # q is the raw block norms, which an L2: view would otherwise throw away -- for a
    # rollout state that magnitude is informative (a TSU pseudo-feature and an untouched
    # low-res feature do not live at the same scale as a true aggregate).
    if "q" in toks and qb is None:
        qb = norms(x, n, m, k, s)
    if norm:
        x, n, m, k, s = (norm(a) for a in (x, n, m, k, s))
    base = {"x": x, "n": n, "m": m, "k": k, "s": s}
    out = []
    for t in toks:
        if t == "v":
            out.append(v)
        elif t == "q":
            out.append(qb)
        elif t in base:
            out.append(base[t])
        else:
            a, b = DIFF[t]
            out.append(base[a] - base[b])
    return out


def norms(x, n, m, k, s):
    if isinstance(x, np.ndarray):
        return np.stack([np.linalg.norm(a, axis=1) for a in (x, n, m, k, s)], axis=1)
    return torch.stack([a.norm(dim=1) for a in (x, n, m, k, s)], dim=1)


def make_view_np(x, n, m, k, s, v, mode):
    l2, toks = parse_view(mode)
    f = (lambda a: a / (np.linalg.norm(a, axis=1, keepdims=True) + EPS)) if l2 else None
    a32 = lambda z: np.asarray(z, dtype=np.float32)
    parts = blocks(a32(x), a32(n), a32(m), a32(k), a32(s), a32(v), toks, f)
    return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)


def make_view_t(x, n, m, k, s, v, mode):
    l2, toks = parse_view(mode)
    f = (lambda a: a / (a.norm(dim=1, keepdim=True) + EPS)) if l2 else None
    parts = blocks(x, n, m, k, s, v, toks, f)
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
