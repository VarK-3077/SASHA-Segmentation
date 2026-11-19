# The staged screening policy. Stage 1 is a strict pass: screen the units the L6 head scores >= t6a at L3,
# paint the ones the supervised head is sure are core, segment the ones the token-level head scores >= t3a or
# the supervised head scores >= p3a (at most k1 of them, best first) and trace borders. What the segmenter
# returns is the evidence: if any segmented unit reaches `ev` (or anything was painted) the slide holds tumour
# and stage 2 runs the looser pass (t6b, t3b, p3b); if not, the slide ends there. One vector theta holds the
# nine numbers, plus `veto`: a unit the token-level head scores below it is never segmented or painted, and
# `veto6` / `r6`, the same idea one level up with a second L6 head (it can veto an L3 read, or ask for one alone). `key` names the token-level score the policy reads; key=None leaves the token head out, so the
# same code plays the supervised rule (RULE: stage 1 only) and every staged variant.

from collections import namedtuple

import numpy as np

from experiments.rlpolicy import sim as S

Theta = namedtuple('Theta', 't6a t3a p3a tc ev t6b t3b p3b k1 veto veto6 r6', defaults=(0.0, 0.0, 2.0))
OFF = 2.0                                                   # a threshold no score reaches
RULE = Theta(0.6, OFF, 0.5, 0.99, OFF, 0.6, OFF, 0.5, 1e9, 0.0, 0.0, OFF)  # never escalates: the plain cascade
LO = Theta(0.30, 0.01, 0.05, 0.80, 0.40, 0.02, 0.002, 0.02, 5, 0.0, 0.0, 0.0)
HI = Theta(0.995, 1.01, 1.01, 1.01, 1.01, 0.995, 1.01, 1.01, 400, 0.99, 0.99, 2.0)


def wanted(s, t, tt, tp, veto=0.0):
    return ((t >= tt) | (s.p3 >= tp)) & (t >= veto)


def screen(s, t6, th):
    """Units read at L3: the L6 head passes them and the second L6 head does not veto them, or the second head alone is sure."""
    if th.veto6 <= 0 and th.r6 > 1:
        return s.tissue & (s.pg >= t6)
    g = getattr(s, s.g_key)
    return s.tissue & (((s.pg >= t6) & (g >= th.veto6)) | (g >= th.r6))


def top_k(mask, score, k):
    if mask.sum() <= k:
        return mask
    idx = np.flatnonzero(mask.ravel())
    out = np.zeros(mask.size, bool)
    out[idx[np.argsort(-score.ravel()[idx], kind='stable')[:int(k)]]] = True
    return out.reshape(mask.shape)


def play(s, th, key='t_top4'):
    t = getattr(s, key) if key else np.zeros_like(s.p3)
    t3b, p3b, t6b = min(th.t3a, th.t3b), min(th.p3a, th.p3b), min(th.t6a, th.t6b)
    scr = screen(s, th.t6a, th)
    paint = scr & (s.p_core >= th.tc) & wanted(s, t, t3b, p3b, th.veto)
    zoom = top_k(scr & wanted(s, t, th.t3a, th.p3a, th.veto) & ~paint, np.maximum(t / th.t3a, s.p3 / th.p3a), th.k1)
    z1, _ = S.trace(s, zoom, paint)
    if not (paint.any() or (s.unit_max[z1] >= th.ev).any()):
        out = S.finish(s, scr, zoom, paint)
        out['stage'], out['stopped'] = 1, th.ev <= 1.0
        return out
    scr2 = screen(s, t6b, th)
    paint2 = scr2 & (s.p_core >= th.tc) & wanted(s, t, t3b, p3b, th.veto)
    out = S.finish(s, scr2, (scr2 & wanted(s, t, t3b, p3b, th.veto) & ~paint2) | z1, paint2, rounds=1)
    out['stage'] = 2
    return out


def clip(v):
    return Theta(*np.clip(np.asarray(v, np.float64), LO, HI))
