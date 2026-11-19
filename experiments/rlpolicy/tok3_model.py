# The token-level L3 head: one tumour logit per level-3 token of a unit (a 16 x 16 map) and two unit logits
# (holds tumour, is core) read from pooled token features and the token logits' own statistics. It sees one
# unit's tokens only, so it costs what the supervised unit head costs and can run on any subset of units.

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokHead(nn.Module):
    def __init__(self, d_in=256, d=128, ctx=3, p_drop=0.2):
        super().__init__()
        self.proj = nn.Conv2d(d_in, d, 1)
        self.ctx = nn.Conv2d(d, d, ctx, padding=ctx // 2) if ctx > 1 else None
        self.hidden = nn.Conv2d(d, d, 1)
        self.drop = nn.Dropout2d(p_drop)
        self.tok = nn.Conv2d(d, 1, 1)
        self.unit = nn.Sequential(nn.Linear(2 * d + 3, d), nn.GELU(), nn.Dropout(p_drop), nn.Linear(d, 2))

    def forward(self, x):
        """x (B, 256, 16, 16) -> token logits (B, 16, 16), unit logits (B, 2)."""
        h = F.gelu(self.proj(x))
        if self.ctx is not None:
            h = F.gelu(h + self.ctx(h))
        h = F.gelu(self.hidden(h))
        t = self.tok(self.drop(h))[:, 0]
        flat = t.flatten(1)
        stats = torch.stack([flat.amax(1), flat.topk(4, 1)[0].mean(1), torch.sigmoid(flat).mean(1)], 1)
        u = self.unit(torch.cat([h.mean((2, 3)), h.amax((2, 3)), stats], 1))
        return t, u
