# The screen head: from a slide's level-L token map to one 4-way bucket logit per routing unit.
# Tokens are projected to 256-d; with context on, PFA-ScanNet's PFE block (GlobalConv: the
# separable 15x1 / 1x15 branches summed, as in the token head) mixes neighbouring tokens. Each
# unit's block of tokens is pooled two ways, mean and max (max keeps a few tumour tokens from
# being diluted, the small-lesion study's main lever), then a second GlobalConv at unit scale
# adds wider context, and a 1x1 layer gives the four logits. No window-mean vector: training
# sees windows and evaluation whole slides, so a mean would shift between the two.
# With context off the same head reduces to per-unit token pooling + MLP: the floor.

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalConv(nn.Module):
    def __init__(self, c, k):
        super().__init__()
        p = k // 2
        self.l1, self.l2 = nn.Conv2d(c, c, (k, 1), padding=(p, 0)), nn.Conv2d(c, c, (1, k), padding=(0, p))
        self.r1, self.r2 = nn.Conv2d(c, c, (1, k), padding=(0, p)), nn.Conv2d(c, c, (k, 1), padding=(p, 0))

    def forward(self, x):
        return self.l2(self.l1(x)) + self.r2(self.r1(x))


class ScreenHead(nn.Module):
    def __init__(self, tok_per_unit, ctx=True, d_in=256, d=256, p_drop=0.1):
        super().__init__()
        self.k, self.ctx = tok_per_unit, ctx
        self.proj = nn.Conv2d(d_in, d, 1)
        self.tok_ctx = GlobalConv(d, 15) if ctx else None
        self.mix = nn.Conv2d(2 * d, d, 1)
        self.unit_ctx = GlobalConv(d, 7) if ctx else None
        self.hidden = nn.Conv2d(d, d, 1)
        self.drop = nn.Dropout2d(p_drop)
        self.cls = nn.Conv2d(d, 4, 1)

    def forward(self, tok):
        """tok (B, 256, H*k, W*k) -> logits (B, 4, H, W)."""
        h = F.gelu(self.proj(tok))
        if self.ctx:
            h = F.gelu(h + self.tok_ctx(h))
        u = torch.cat([F.avg_pool2d(h, self.k), F.max_pool2d(h, self.k)], 1)
        u = F.gelu(self.mix(u))
        if self.ctx:
            u = F.gelu(u + self.unit_ctx(u))
        u = F.gelu(self.hidden(u))
        return self.cls(self.drop(u))
