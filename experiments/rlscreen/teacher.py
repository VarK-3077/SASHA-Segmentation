# The hand-written screening policy: the supervised baseline the agent is compared with, and the
# teacher it is cloned from. Generic rules, one probability threshold tau on p(tumour) from the
# supervised heads (0.5 by default, no dataset tuning). Coarse design: round 0 screens at level 3
# every unit the global view calls tumour, round 1 zooms the screened units level 3 calls tumour.
# Full design: round 0 zooms every unit level 3 calls tumour. With follow on, every later round
# also zooms the unzoomed neighbours of zoomed units whose tumour touches the shared border, so a
# lesion found at its edge is traced. A round with nothing to do stops the episode.

import torch

from experiments.rlscreen.env import NONE, SCREEN, ZOOM


def neighbours_to_follow(ep, outcome):
    """Units next to a zoomed unit whose border strip on that side holds tumour."""
    z = ep.zoomed
    e = outcome[..., 1:] > 0          # N S W E strips of each unit
    want = torch.zeros_like(z)
    want[:-1] |= (z & e[..., 0])[1:]  # tumour on the north edge of the unit below
    want[1:] |= (z & e[..., 1])[:-1]  # south edge of the unit above
    want[:, :-1] |= (z & e[..., 2])[:, 1:]
    want[:, 1:] |= (z & e[..., 3])[:, :-1]
    return want & ep.t.tissue & ~ep.zoomed


def act(ep, tau=0.5, follow=True, outcome=None):
    t = ep.t
    out = t.outcome if outcome is None else outcome
    dev = t.tissue.device
    p_g = torch.from_numpy(1 - t.p_g[..., 0]).to(dev)
    p_3 = torch.from_numpy(1 - t.p3[..., 0]).to(dev)
    a = torch.full((t.hu, t.wu), NONE, dtype=torch.long, device=dev)
    free = t.tissue & ~ep.zoomed
    if t.design == 'coarse':
        if ep.round == 0:
            a[free & (p_g >= tau)] = SCREEN
        else:
            a[free & ep.screened & (p_3 >= tau)] = ZOOM
    elif ep.round == 0:
        a[free & (p_3 >= tau)] = ZOOM
    if follow and ep.round > 0:
        a[neighbours_to_follow(ep, out)] = ZOOM
    stop = bool((a == NONE).all())
    return stop, a
