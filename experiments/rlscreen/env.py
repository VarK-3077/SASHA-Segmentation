# One slide as an episode of batched rounds. Every round the agent first decides whether to stop;
# if not, each tissue unit gets one of none / screen (read at level 3, revealing its level-3 view;
# coarse design only) / zoom (segment it at level 2, revealing the outcome). A round that picks
# nothing also ends the episode, as does round R_MAX. With the ideal segmenter a zoomed unit
# contributes exactly its ground-truth tumour, so quality J is a recall: of tumour area ('area'),
# of non-ITC lesions weighted equally ('lesion'), or their mean ('mix'); a slide without tumour
# has J = 1 throughout. Reward per round: S^J(t-1) * (J(t) - J(t-1) - lam * cost), with cost the
# round's A100 seconds over the seconds of segmenting the whole slide (costs.py).

import numpy as np
import torch

from experiments.rlscreen import costs as C
from experiments.rlscreen.paths import R_MAX

NONE, SCREEN, ZOOM = 0, 1, 2


class SlideTable:
    """Static per-slide tensors on the device, for one design."""

    def __init__(self, name, z, design, g_level, backbone, device):
        self.name, self.design, self.g_level, self.backbone, self.device = name, design, g_level, backbone, device
        self.tissue = torch.from_numpy(z['tissue']).to(device)
        self.hu, self.wu = self.tissue.shape
        self.n_units = int(z['tissue'].sum())
        self.frac = z['frac'].astype(np.float64)
        self.tumour_px = float(self.frac[z['tissue'] | (self.frac > 0)].sum())
        gl = 3 if design == 'full' else g_level
        t = lambda k: torch.from_numpy(z[k].astype(np.float32)).to(device)
        self.g = torch.cat([t(f'f{gl}_mean'), t(f'f{gl}_max'), t(f'p{gl}')], -1)           # (hu, wu, 516)
        self.l3 = torch.cat([t('f3_mean'), t('f3_max'), t('p3_unit' if design == 'coarse' else 'p3')], -1)
        self.p_g = z[f'p{gl}'].astype(np.float32)
        self.p3 = z['p3_unit' if design == 'coarse' else 'p3'].astype(np.float32)
        self.outcome = torch.cat([t('frac')[..., None], t('edges')], -1)                   # (hu, wu, 5): ideal segmenter
        self.edges = z['edges'].astype(np.float32)
        self.bucket = z['bucket']
        # lesion bookkeeping for the 'lesion' quality: per unit, which non-ITC lesions it holds and how much
        keep = ~z['lesion_itc'][z['lu_lesion']] if len(z['lu_lesion']) else np.zeros(0, bool)
        self.lu_unit, self.lu_lesion, self.lu_px = z['lu_unit'][keep], z['lu_lesion'][keep], z['lu_px'][keep]
        self.lesions = np.unique(self.lu_lesion)
        self.lesion_px = z['lesion_px'].astype(np.float64)
        self.has_tumour = self.tumour_px > 0


class Episode:
    def __init__(self, tab, lam, metric='mix', S=4.0):
        self.t, self.lam, self.metric, self.S = tab, lam, metric, S
        self.screened = torch.zeros_like(tab.tissue)
        self.zoomed = torch.zeros_like(tab.tissue)
        if tab.design == 'full':
            self.screened = tab.tissue.clone()   # the whole level-3 screen is the global pass
        self.round, self.spent, self.done = 0, 0.0, False
        self.n_screen = self.n_zoom = 0
        self.J = self.quality()

    def quality(self):
        tab = self.t
        if not tab.has_tumour:
            return 1.0
        zm = self.zoomed.cpu().numpy()
        area = float(tab.frac[zm].sum() / tab.tumour_px)
        if self.metric == 'area':
            return area
        if len(tab.lesions):
            hit = zm.reshape(-1)[tab.lu_unit]
            covered = np.bincount(tab.lu_lesion[hit], weights=tab.lu_px[hit], minlength=len(tab.lesion_px))
            lesion = float(np.mean(covered[tab.lesions] / tab.lesion_px[tab.lesions]))
        else:
            lesion = area
        return lesion if self.metric == 'lesion' else 0.5 * (area + lesion)

    def allowed(self):
        """(hu, wu, 3) bool action mask."""
        free = self.t.tissue & ~self.zoomed
        can_screen = free & ~self.screened if self.t.design == 'coarse' else torch.zeros_like(free)
        return torch.stack([torch.ones_like(free), can_screen, free], -1)

    def step(self, stop, actions):
        """stop: bool; actions: (hu, wu) long tensor. Returns the round reward."""
        if stop or self.round >= R_MAX:
            self.done = True
            return 0.0
        sc = (actions == SCREEN) & self.t.tissue & ~self.screened & ~self.zoomed
        zm = (actions == ZOOM) & self.t.tissue & ~self.zoomed
        ns, nz = int(sc.sum()), int(zm.sum())
        if ns + nz == 0:
            self.done = True
            return 0.0
        self.screened |= sc
        self.zoomed |= zm
        self.n_screen += ns; self.n_zoom += nz
        cost = C.cost_fraction(self.t.backbone, ns, nz, self.t.n_units)
        self.spent += cost
        self.round += 1
        J0, self.J = self.J, self.quality()
        if self.round >= R_MAX:
            self.done = True
        return float(self.S ** J0 * (self.J - J0 - self.lam * cost))

    def seconds(self):
        g = 3 if self.t.design == 'full' else self.t.g_level
        return C.episode_seconds(self.t.backbone, g, 0 if self.t.design == 'full' else self.n_screen, self.n_zoom, self.round)


N_PLANES = {'full': 516 + 516 + 1 + 5 + 1 + 1,   # pooled tokens + head probabilities of both views
            'light': 4 + 4 + 1 + 5 + 1 + 1}       # head probabilities only: the less intense state
N_SCALARS = 3


def snapshot(ep):
    return dict(screened=ep.screened.clone(), zoomed=ep.zoomed.clone(), round=ep.round, spent=ep.spent, lam=ep.lam)


def observation(t, s, state='full', outcome=None):
    """Per-unit input planes (C, hu, wu) and the slide-level scalars, from a table t and a state
    snapshot s. `outcome` is the zoom result per unit (tumour fraction, N S W E border-strip
    fractions); default the ideal segmenter's."""
    sc = s['screened'].float()[..., None]
    zm = s['zoomed'].float()[..., None]
    out = t.outcome if outcome is None else outcome
    g, l3 = (t.g, t.l3) if state == 'full' else (t.g[..., -4:], t.l3[..., -4:])
    planes = torch.cat([g, l3 * sc, sc, out * zm, zm, t.tissue.float()[..., None]], -1)
    scal = torch.tensor([np.log(s['lam']), s['round'] / R_MAX, s['spent']], device=t.device, dtype=torch.float32)
    return planes.permute(2, 0, 1), scal
