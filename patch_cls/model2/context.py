# Multi-scale spatial context for 2048px tiles.
#
# Tiles of a slide sit on a regular 2048px level-0 grid, so the mean feature over a
# (2r+1)x(2r+1) tile neighbourhood is a box filter. Rasterising the slide's tiles into a
# dense grid and taking an integral image makes every radius O(1) per tile instead of the
# O(r^2) dictionary lookup the Model-1 code used, which matters here because r goes to 4.
#
# Cells with no tile contribute nothing to either numerator or denominator, so the result
# is the mean over the tiles that actually exist in the window - never a zero-padded mean.

import numpy as np

TILE = 2048


def grid_cells(coords):
    """Level-0 pixel coords on a TILE grid -> zero-based integer cells and grid shape."""
    g = np.rint(np.asarray(coords, dtype=np.float64) / TILE).astype(np.int64)
    g -= g.min(axis=0, keepdims=True)
    return g, (int(g[:, 0].max()) + 1, int(g[:, 1].max()) + 1)


def _integral(a):
    """Zero-padded 2-d integral image over the first two axes."""
    out = np.zeros((a.shape[0] + 1, a.shape[1] + 1) + a.shape[2:], dtype=np.float64)
    out[1:, 1:] = a.cumsum(axis=0).cumsum(axis=1)
    return out


def _boxsum(ii, r0, r1, c0, c1):
    return ii[r1, c1] - ii[r0, c1] - ii[r1, c0] + ii[r0, c0]


def box_means(values, g, shape, radii):
    """Mean of `values` rows over each tile's (2r+1)^2 neighbourhood, for every r in radii.

    values: (N,d) or (N,); g: (N,2) integer cells; shape: (H,W). Returns a list of (N,d).
    """
    v = np.asarray(values, dtype=np.float64)
    flat = v.ndim == 1
    if flat:
        v = v[:, None]
    h, w = shape
    num = np.zeros((h, w, v.shape[1]))
    den = np.zeros((h, w))
    np.add.at(num, (g[:, 0], g[:, 1]), v)
    np.add.at(den, (g[:, 0], g[:, 1]), 1.0)
    inum, iden = _integral(num), _integral(den)
    out = []
    for r in radii:
        r0 = np.clip(g[:, 0] - r, 0, h)
        r1 = np.clip(g[:, 0] + r + 1, 0, h)
        c0 = np.clip(g[:, 1] - r, 0, w)
        c1 = np.clip(g[:, 1] + r + 1, 0, w)
        s = _boxsum(inum, r0, r1, c0, c1)
        d = _boxsum(iden, r0, r1, c0, c1)[:, None]
        m = s / np.maximum(d, 1.0)
        out.append(m[:, 0] if flat else m)
    return out


def pyramid(feat, coords, visited_idx, radii=(1, 2, 4)):
    """Everything a tile's context needs: the 3 neighbourhood means, slide mean, aux block.

    Returns (means[list of (N,384)], slide_mean (384,), aux (N,4) = [visited, vfrac@radii]).
    """
    f = np.asarray(feat, dtype=np.float32)
    g, shape = grid_cells(coords)
    means = box_means(f, g, shape, radii)
    vis = np.zeros(len(f), dtype=np.float32)
    if len(visited_idx):
        vis[np.asarray(visited_idx, dtype=np.int64)] = 1.0
    vfr = box_means(vis, g, shape, radii)
    aux = np.stack([vis] + [a.astype(np.float32) for a in vfr], axis=1)
    return [m.astype(np.float32) for m in means], f.mean(axis=0), aux.astype(np.float32)
