# Assembles a slide's screen tokens into one dense map and its unit labels into the matching
# grid: the token map has 16 tokens per crop side and the unit grid m units per crop side, so a
# unit is a block of 16/m tokens. Windows of either are cut on the crop grid, which keeps the
# two aligned. Tokens are memory-mapped; only the crops inside a window are read.

import numpy as np
import torch

from experiments.lowsep import paths as P

IGNORE = -1


class SlideMaps:
    def __init__(self, name, level):
        self.name, self.level = name, level
        self.m = P.units_per_crop(level)
        z = np.load(P.label_npz(name))
        self.tissue, self.frac, self.bucket, self.trust = z['tissue'], z['frac'], z['bucket'].astype(np.int64), bool(z['trust'])
        hu, wu = self.tissue.shape
        self.hc, self.wc = -(-hu // self.m), -(-wu // self.m)
        self.rc = np.load(f'{P.tok_dir(level)}/{name}_rc.npy')
        self.tok = np.load(f'{P.tok_dir(level)}/{name}.npy', mmap_mode='r')
        self.index = np.full((self.hc, self.wc), -1, np.int64)
        self.index[self.rc[:, 0], self.rc[:, 1]] = np.arange(len(self.rc))
        pad = ((0, self.hc * self.m - hu), (0, self.wc * self.m - wu))
        target = np.where(self.tissue, self.bucket, IGNORE)
        if not self.trust:
            target = np.where(self.bucket == 0, IGNORE, target)
        self.target = np.pad(target, pad, constant_values=IGNORE)
        self.frac_p = np.pad(self.frac, pad)
        self.tissue_p = np.pad(self.tissue, pad)
        self.bucket_p = np.pad(self.bucket, pad)

    def tumour_crops(self):
        has = (self.bucket_p > 0).reshape(self.hc, self.m, self.wc, self.m).any(axis=(1, 3))
        return np.argwhere(has)

    def window(self, r0, c0, h, w):
        """Token map (256, h*16, w*16) float16 and unit targets (h*m, w*m) for crops [r0, r0+h) x [c0, c0+w)."""
        tk = np.zeros((P.D_PCA, h * P.TOK, w * P.TOK), np.float16)
        idx = self.index[max(r0, 0):r0 + h, max(c0, 0):c0 + w]
        rr, cc = np.nonzero(idx >= 0)
        if len(rr):
            rows = idx[rr, cc]
            order = np.argsort(rows)
            blocks = self.tok[rows[order]]
            for b, r, c in zip(blocks, rr[order] + max(r0, 0) - r0, cc[order] + max(c0, 0) - c0):
                tk[:, r * P.TOK:(r + 1) * P.TOK, c * P.TOK:(c + 1) * P.TOK] = b
        tg = np.full((h * self.m, w * self.m), IGNORE, np.int64)
        u0, v0 = r0 * self.m, c0 * self.m
        ua, va = max(u0, 0), max(v0, 0)
        ub, vb = min(u0 + h * self.m, self.target.shape[0]), min(v0 + w * self.m, self.target.shape[1])
        if ub > ua and vb > va:
            tg[ua - u0:ub - u0, va - v0:vb - v0] = self.target[ua:ub, va:vb]
        return tk, tg

    def full(self):
        return self.window(0, 0, self.hc, self.wc)


def to_device(tk, device):
    return torch.from_numpy(tk).to(device, non_blocking=True).float()
