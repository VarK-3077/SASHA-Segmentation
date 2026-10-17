# The actor-critic over a whole slide. Every tile gets a hidden vector from its embedding,
# low-res score summary and current status; a pooled slide vector feeds back into each tile's
# action logits (zoom, core, and normal in mode 'lr') and produces the stop logit and the
# value. Works on (N, ·) for acting and on (T, N, ·) for the PPO update.

import torch
import torch.nn as nn

from experiments.rlbin.env import N_EXTRA, N_GLOB, UNASSIGNED, ZOOMED


def action_mask(status, cap, n_act=2):
    """status (..., N) -> bool (..., n_act*N+1): zoom allowed, core allowed, [normal allowed], stop."""
    free = status == UNASSIGNED
    zooms = (status == ZOOMED).sum(-1, keepdim=True)
    parts = [free & (zooms < cap), free] + ([free] if n_act == 3 else [])
    return torch.cat(parts + [torch.ones_like(free[..., :1])], -1)


class Policy(nn.Module):
    def __init__(self, d_emb=384, n_act=2, n_dyn=5, h=256, e=128):
        super().__init__()
        self.d_emb, self.n_act = d_emb, n_act
        self.proj = nn.Linear(d_emb, e)
        self.patch = nn.Sequential(nn.Linear(e + N_EXTRA + n_dyn + N_GLOB, h), nn.GELU(), nn.Linear(h, h), nn.GELU())
        self.glob = nn.Sequential(nn.Linear(2 * h + N_GLOB, e), nn.GELU())
        self.act = nn.Sequential(nn.Linear(h + e, h), nn.GELU(), nn.Linear(h, n_act))
        self.stop = nn.Linear(e, 1)
        self.value = nn.Linear(e, 1)

    def forward(self, static, dyn, glob):
        lead = static.shape[:-1]
        emb = self.proj(static[..., :self.d_emb])
        g_b = glob.unsqueeze(-2).expand(*lead, N_GLOB)
        h = self.patch(torch.cat([emb, static[..., self.d_emb:], dyn, g_b], -1))
        g = self.glob(torch.cat([h.mean(-2), h.amax(-2), glob], -1))
        g_b = g.unsqueeze(-2).expand(*lead, g.shape[-1])
        a = self.act(torch.cat([h, g_b], -1))
        logits = torch.cat([a[..., k] for k in range(self.n_act)] + [self.stop(g)], -1)
        return logits, self.value(g).squeeze(-1)
