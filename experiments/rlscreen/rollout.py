# Loading slide tables and playing episodes with any of the three actors: the learned policy
# (sampled or greedy), the hand-written teacher, or the oracle that zooms exactly the units holding
# tumour. Each played round is kept as a transition (state snapshot, actions, stop, log-probs,
# value, reward) for training, and every finished episode is summarised for evaluation.

import os

import numpy as np
import torch

from experiments.lowsep import paths as LP
from experiments.rlscreen import paths as P
from experiments.rlscreen import teacher as T
from experiments.rlscreen.env import NONE, ZOOM, Episode, SlideTable, observation, snapshot
from experiments.rlscreen.policy import distributions


def load_tables(split, design, g_level, backbone, device, trusted_only=False):
    d = P.units_dir(backbone)
    out = []
    for f in sorted(os.listdir(d)):
        if not f.endswith('.npz'):
            continue
        z = np.load(os.path.join(d, f))
        if str(z['split']) != split or (trusted_only and not bool(z['trust'])):
            continue
        out.append(SlideTable(f[:-4], z, design, g_level, backbone, device))
    return out


def policy_step(policy, tab, ep, state, greedy):
    s = snapshot(ep)
    planes, scal = observation(tab, s, state)
    allowed = ep.allowed()
    with torch.no_grad():
        logits, stop_logit, value = policy(planes[None], scal[None], tab.tissue[None])
        unit_d, stop_d = distributions(logits[0], stop_logit[0], allowed)
        if greedy:
            stop = bool(stop_logit[0] > 0)
            a = logits[0].masked_fill(~allowed, -1e9).argmax(-1)
        else:
            stop = bool(stop_d.sample())
            a = unit_d.sample()
        a = torch.where(tab.tissue, a, torch.full_like(a, NONE))
        lp_units = unit_d.log_prob(a)
        lp_stop = stop_d.log_prob(torch.tensor(float(stop), device=a.device))
    return s, allowed, stop, a, lp_units, lp_stop, float(value[0])


def play(tab, lam, actor, metric='mix', S=4.0, policy=None, state='full', greedy=False, tau=0.5, follow=True):
    """actor: 'policy', 'teacher' or 'oracle'. Returns (summary, transitions)."""
    ep = Episode(tab, lam, metric, S)
    trans = []
    while not ep.done:
        if actor == 'policy':
            s, allowed, stop, a, lp_u, lp_s, v = policy_step(policy, tab, ep, state, greedy)
        else:
            s, allowed, v, lp_u, lp_s = snapshot(ep), ep.allowed(), 0.0, None, None
            if actor == 'teacher':
                stop, a = T.act(ep, tau=tau, follow=follow)
            else:
                a = torch.full((tab.hu, tab.wu), NONE, dtype=torch.long, device=tab.tissue.device)
                if ep.round == 0:
                    a[torch.from_numpy(tab.frac > 0).to(a.device) & tab.tissue] = ZOOM
                stop = bool((a == NONE).all())
        r = ep.step(stop, a)
        trans.append(dict(s=s, allowed=allowed, stop=stop, a=a, lp_u=lp_u, lp_s=lp_s, v=v, r=r))
    return summary(ep), trans


def summary(ep):
    t = ep.t
    m = ep.metric
    ep.metric = 'area'; area = ep.quality()
    ep.metric = 'lesion'; lesion = ep.quality()
    ep.metric = m
    zm = ep.zoomed.cpu().numpy()
    found = None
    if len(t.lesions):
        hit = zm.reshape(-1)[t.lu_unit]
        found = float(np.mean([hit[t.lu_lesion == l].any() for l in t.lesions]))
    return dict(slide=t.name, tumour=bool(t.has_tumour), n_units=t.n_units, area_recall=area, lesion_recall=lesion,
                lesions_found=found, ideal_dice=(2 * area / (1 + area)) if t.has_tumour else 1.0,
                zoom_frac=ep.n_zoom / max(t.n_units, 1), screen_frac=ep.n_screen / max(t.n_units, 1),
                rounds=ep.round, seconds=ep.seconds(), spent=ep.spent)
