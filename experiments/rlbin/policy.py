# The actor-critic over a whole slide. Every tile gets a hidden vector from its embedding,
# low-res score summary and current status; a pooled slide vector feeds back into each tile's
# zoom/core logits and produces the stop logit and the value. Works on (N, ·) for acting and
# on (T, N, ·) for the PPO update.

import torch
import torch.nn as nn

from experiments.rlbin.env import N_DYN, N_EXTRA, N_GLOB, UNASSIGNED, ZOOMED


def action_mask(status, cap):
    """status (..., N) -> bool (..., 2N+1): zoom allowed, core allowed, stop always."""
    free = status == UNASSIGNED
    zooms = (status == ZOOMED).sum(-1, keepdim=True)
    zoom_ok = free & (zooms < cap)
    stop = torch.ones_like(free[..., :1])
    return torch.cat([zoom_ok, free, stop], -1)


class Policy(nn.Module):
    def __init__(self, h=256, e=128):
        super().__init__()
        self.proj = nn.Linear(384, e)
        self.patch = nn.Sequential(nn.Linear(e + N_EXTRA + N_DYN + N_GLOB, h), nn.GELU(), nn.Linear(h, h), nn.GELU())
        self.glob = nn.Sequential(nn.Linear(2 * h + N_GLOB, e), nn.GELU())
        self.act = nn.Sequential(nn.Linear(h + e, h), nn.GELU(), nn.Linear(h, 2))
        self.stop = nn.Linear(e, 1)
        self.value = nn.Linear(e, 1)

    def forward(self, static, dyn, glob):
        lead = static.shape[:-1]
        emb = self.proj(static[..., :384])
        g_b = glob.unsqueeze(-2).expand(*lead, N_GLOB)
        h = self.patch(torch.cat([emb, static[..., 384:], dyn, g_b], -1))
        g = self.glob(torch.cat([h.mean(-2), h.amax(-2), glob], -1))
        g_b = g.unsqueeze(-2).expand(*lead, g.shape[-1])
        a = self.act(torch.cat([h, g_b], -1))
        logits = torch.cat([a[..., 0], a[..., 1], self.stop(g)], -1)
        return logits, self.value(g).squeeze(-1)
