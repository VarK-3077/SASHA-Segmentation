# The screening policy: a light fully convolutional net over the unit grid. A 1x1 layer squeezes
# each unit's planes to 64 channels, three dilated 3x3 residual blocks (reach 7 units) mix in the
# neighbourhood, the slide-level scalars (log lambda, round, compute spent) are added to every
# unit, and three heads read the result: per-unit logits over none / screen / zoom, and from the
# tissue-pooled features a stop logit and the value.

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.rlscreen.env import N_PLANES, N_SCALARS


class Block(nn.Module):
    def __init__(self, c, dil):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=dil, dilation=dil)
        self.norm = nn.GroupNorm(8, c)

    def forward(self, x):
        return x + F.gelu(self.norm(self.conv(x)))


class ScreenPolicy(nn.Module):
    def __init__(self, state='full', c=64):
        super().__init__()
        self.inp = nn.Conv2d(N_PLANES[state], c, 1)
        self.scal = nn.Linear(N_SCALARS, c)
        self.blocks = nn.Sequential(Block(c, 1), Block(c, 2), Block(c, 4))
        self.unit = nn.Conv2d(c, 3, 1)
        self.unit_value = nn.Conv2d(c, 1, 1)   # each unit's own future return, for per-unit credit
        self.slide = nn.Sequential(nn.Linear(2 * c + N_SCALARS, c), nn.GELU())
        self.stop = nn.Linear(c, 1)
        self.value = nn.Linear(c, 1)

    def forward(self, planes, scal, tissue, unit_values=False):
        """planes (B, C, H, W), scal (B, 3), tissue (B, H, W) bool -> unit logits (B, H, W, 3), stop logit (B,),
        value (B,), and with unit_values the per-unit values (B, H, W)."""
        h = F.gelu(self.inp(planes) + self.scal(scal)[:, :, None, None])
        h = self.blocks(h)
        logits = self.unit(h).permute(0, 2, 3, 1)
        m = tissue[:, None].float()
        n = m.sum((2, 3)).clamp(min=1)
        mean = (h * m).sum((2, 3)) / n
        mx = (h - 1e4 * (1 - m)).amax((2, 3))
        g = self.slide(torch.cat([mean, mx, scal], -1))
        if unit_values:
            return logits, self.stop(g)[:, 0], self.value(g)[:, 0], self.unit_value(h)[:, 0]
        return logits, self.stop(g)[:, 0], self.value(g)[:, 0]


def distributions(logits, stop_logit, allowed):
    """Masked per-unit categorical and the stop Bernoulli."""
    logits = logits.masked_fill(~allowed, -1e9)
    return torch.distributions.Categorical(logits=logits), torch.distributions.Bernoulli(logits=stop_logit)
