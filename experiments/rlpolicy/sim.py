# The fast cascade simulator: plays a routing (which units are screened at L3, segmented at L2 or painted) on
# a slide's tables (simtab.py) and returns what the exact scorer would give, without building a map. Border
# tracing is the cascade's own: a segmented unit whose paint reaches a border strip pulls in the neighbour
# across it, a painted unit pulls in its four neighbours, repeated until nothing is added. A component of the
# dense map restricted to the segmented units counts as one component (a cut that splits it is ignored); a
# group of painted units is one candidate at 1.0. check_sim.py measures the gap to the exact scorer.

import os

import numpy as np
from scipy import ndimage as nd

from experiments.rlpolicy import where as W
from experiments.rlprobe.cost import seconds
from utils.segmentation_metrics import froc as froc_of

SEED = 0.9
ROUND_S = 0.025          # est. cost of one extra batched round (rlcascade env2)
N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


class Slide:
    def __init__(self, name, tok3=None):
        z = np.load(f'{W.ROOT}/sim/{name}.npz')
        self.name = name
        for k in z.files:
            setattr(self, k, z[k])
        self.tumour, self.G, self.n_lab = bool(self.tumour), int(self.G), int(self.n_lab)
        self.hu, self.wu = self.tissue.shape
        self.n = int(self.tissue.sum())
        self.pos, self.edges = self.seg_frac > 0, self.seg_edges > 0
        self.lesions = np.setdiff1d(np.arange(1, self.n_lab + 1), self.itcs)
        if tok3 is not None:
            self.add_tok3(tok3)

    def add_tok3(self, tag, prefix='t'):
        """A token-level L3 head's outputs as unit grids (attributes PREFIX_unit, _core, _top4, _top16, _max, _mean,
        _mix, _edges): unit probability, core probability, mean of the top 4 / 16 tokens, maximum and mean token
        probability, and the mean probability on each border strip. prefix 't' is the tumour head; a second head
        (e.g. one trained on the segmenter's output) can sit beside it under another prefix."""
        p = np.load(f'{W.head_dir(tag)}/pred/{self.name}.npz')
        rc = p['rc']
        ok = (rc[:, 0] < self.hu) & (rc[:, 1] < self.wu)
        r, c = rc[ok, 0], rc[ok, 1]
        tp = p['tokp'][ok].astype(np.float32) / 255
        flat = np.sort(tp.reshape(len(tp), -1), 1)[:, ::-1]

        def grid(v):
            a = np.zeros((self.hu, self.wu) + v.shape[1:], np.float32)
            a[r, c] = v
            return a
        sig = lambda x: 1 / (1 + np.exp(-x))
        vals = dict(unit=grid(sig(p['unit'][ok, 0])), core=grid(sig(p['unit'][ok, 1])), top4=grid(flat[:, :4].mean(1)), mean=grid(flat.mean(1)),
                    max=grid(flat[:, 0]), top16=grid(flat[:, :16].mean(1)),
                    edges=grid(np.stack([tp[:, 0].mean(-1), tp[:, -1].mean(-1), tp[:, :, 0].mean(-1), tp[:, :, -1].mean(-1)], -1)))
        vals['mix'] = 0.5 * (vals['unit'] + vals['top4'])
        for k, v in vals.items():
            setattr(self, f'{prefix}_{k}', v)

    def add_tok6(self, tag, key='g_max'):
        """The second L6 head's per-unit score as a grid; `key` picks the unit maximum or mean of its 2 x 2 tokens."""
        p = np.load(f'{W.ROOT}/tok6/{tag}/pred/{self.name}.npz')
        self.g_max, self.g_mean, self.g_key = p['g_max'].astype(np.float32), p['g_mean'].astype(np.float32), key


def shift(m, dr, dc):
    out = np.zeros_like(m)
    h, w = m.shape
    out[max(dr, 0):h + min(dr, 0), max(dc, 0):w + min(dc, 0)] = m[max(-dr, 0):h + min(-dr, 0), max(-dc, 0):w + min(-dc, 0)]
    return out


def follow_from(s, posz, paint, done):
    e = s.edges
    want = shift(posz & e[..., 0], -1, 0) | shift(posz & e[..., 1], 1, 0) | shift(posz & e[..., 2], 0, -1) | shift(posz & e[..., 3], 0, 1)
    for dr, dc in N4:
        want |= shift(paint, dr, dc)
    return want & s.tissue & ~done


def trace(s, zoom, paint):
    """Border tracing from a first set of segmented and painted units -> every unit segmented in the end."""
    zoom = zoom & ~paint
    done = zoom | paint
    add = follow_from(s, zoom & s.pos, paint, done)
    rounds = 0
    while add.any():
        zoom, done, rounds = zoom | add, done | add, rounds + 1
        add = follow_from(s, add & s.pos, np.zeros_like(add), done)
    return zoom, rounds


def outcome(s, zoom, paint):
    """Scores of the map with `zoom` segmented and `paint` painted: cleaned Dice (tumour slides), the raw map's
    FROC record (fp probabilities, best probability per lesion, lesion count) and its FP component count."""
    sel = np.flatnonzero(zoom.reshape(-1)[s.cu_unit])
    comp = s.cu_comp[sel]
    first = sel[np.r_[0, np.flatnonzero(np.diff(comp)) + 1]] if len(sel) else sel       # rows are sorted: strongest unit first
    mx, lab = s.cu_max[first], s.cu_lab[first]
    if paint.any():
        grp, k = nd.label(paint)
        fi = nd.minimum(np.arange(paint.size).reshape(paint.shape), grp, np.arange(1, k + 1)).astype(np.int64)
        mx, lab = np.r_[mx, np.ones(k, np.float32)], np.r_[lab, s.paint_lab.reshape(-1)[fi]]
    tp = np.zeros(s.n_lab, np.float32)
    if s.tumour:
        hit = (lab > 0) & ~np.isin(lab, s.itcs)
        np.maximum.at(tp, lab[hit] - 1, mx[hit])
        tp = tp[s.lesions - 1]
    fp = mx[lab == 0] if s.tumour else mx
    out = dict(froc=(fp.tolist(), tp, len(s.lesions)), n_fp=int(len(fp)), found=int((tp > 0).sum()))
    out['fp_paint'] = [float(v) for v, l in zip(mx[len(first):], lab[len(first):]) if l == 0 or not s.tumour]      # painted groups off any lesion
    if s.tumour:
        kept = np.zeros(int(s.cu_comp.max()) + 1 if len(s.cu_comp) else 0, bool)
        kept[s.cu_comp[first[s.cu_max[first] >= SEED]]] = True
        rows = sel[kept[comp]] if len(sel) else sel
        out['parts'] = (int(s.cu_tp[rows].sum()), int(s.cu_n[rows].sum()), int(s.gt_tis[paint].sum()), int(s.n_tis[paint].sum()))   # zoom tp, zoom pixels, paint tp, paint pixels
        inter, pred = s.cu_tp[rows].sum() + s.gt_tis[paint].sum(), s.cu_n[rows].sum() + s.n_tis[paint].sum()
        out['dice'] = float(2 * inter / (pred + s.G)) if pred + s.G > 0 else 1.0
    return out


def unit_rewards(s, out, w_les, w_fp):
    """Per-unit quality reward of a played route (finish's output), the RL cascade's own credit: each unit's share
    of the slide's cleaned Dice (Dice minus Dice without the unit), + w_les for a lesion whose best candidate lies
    in it, - w_fp times the score of each false-positive candidate in it. Painted groups count as one candidate."""
    zoom, paint = out['zoom'], out['paint']
    hw = s.tissue.size
    q = np.zeros(hw)
    sel = np.flatnonzero(zoom.reshape(-1)[s.cu_unit])
    comp = s.cu_comp[sel]
    first = sel[np.r_[0, np.flatnonzero(np.diff(comp)) + 1]] if len(sel) else sel
    mx, lab, unit = s.cu_max[first].astype(np.float64), s.cu_lab[first], s.cu_unit[first]
    if paint.any():
        grp, k = nd.label(paint)
        fi = nd.minimum(np.arange(paint.size).reshape(paint.shape), grp, np.arange(1, k + 1)).astype(np.int64)
        mx, lab, unit = np.r_[mx, np.ones(k)], np.r_[lab, s.paint_lab.reshape(-1)[fi]], np.r_[unit, fi]
    fp = lab == 0
    np.add.at(q, unit[fp], -w_fp * mx[fp])
    if s.tumour:
        ok = np.flatnonzero((lab > 0) & ~np.isin(lab, s.itcs))
        if len(ok):
            order = ok[np.lexsort((-mx[ok], lab[ok]))]
            best = order[np.r_[0, np.flatnonzero(np.diff(lab[order])) + 1]]
            np.add.at(q, unit[best], w_les)
        iu, pu = np.zeros(hw), np.zeros(hw)
        if len(sel):
            kept = np.zeros(int(s.cu_comp.max()) + 1, bool)
            kept[s.cu_comp[first[s.cu_max[first] >= SEED]]] = True
            rows = sel[kept[comp]]
            iu, pu = np.bincount(s.cu_unit[rows], s.cu_tp[rows], hw).astype(np.float64), np.bincount(s.cu_unit[rows], s.cu_n[rows], hw).astype(np.float64)
        pf = paint.reshape(-1)
        iu[pf] += s.gt_tis.reshape(-1)[pf]
        pu[pf] += s.n_tis.reshape(-1)[pf]
        i, p = iu.sum(), pu.sum()
        d = 2 * i / (p + s.G) if p + s.G > 0 else 0.0
        q += d - 2 * (i - iu) / np.maximum(p - pu + s.G, 1e-9)
    return q.reshape(s.tissue.shape)


def summarise(recs):
    """recs: outcomes with 'sec' -> mean seconds, Dice over tumour slides, FROC."""
    d = [r['dice'] for r in recs if 'dice' in r]
    return dict(sec=float(np.mean([r['sec'] for r in recs])), dice=float(np.mean(d)) if d else float('nan'), froc=float(froc_of([r['froc'] for r in recs])[0]),
                l3=float(np.mean([r['n_l3'] for r in recs])), l2=float(np.mean([r['n_l2'] for r in recs])), paint=float(np.mean([r['n_paint'] for r in recs])),
                fp=float(np.mean([r['n_fp'] for r in recs])), found=int(sum(r['found'] for r in recs)))


def finish(s, screened, zoom, paint, rounds=0):
    """Traces borders from the policy's zoom / paint sets and scores the result; rounds = extra probing rounds paid."""
    paint = paint & s.tissue
    z, _ = trace(s, zoom & s.tissue, paint)
    out = outcome(s, z, paint)
    out.update(n_l3=int(screened.sum()), n_l2=int(z.sum()), n_paint=int(paint.sum()), zoom=z, paint=paint, screened=screened)
    out['sec'] = seconds(out['n_l3'], out['n_l2']) + ROUND_S * rounds
    return out


def rule(s, tau=0.6, tz=0.5, tc=0.99):
    """The supervised cascade r4_trace."""
    screened = s.tissue & (s.pg >= tau)
    routed = screened & (s.p3 >= tz)
    paint = routed & (s.p_core >= tc)
    return finish(s, screened, routed & ~paint, paint)


def load(split, tok3=None, names=None, tok6=None):
    from experiments.rlpolicy.simtab import names_of
    out = [Slide(n, tok3) for n in (names or names_of(split))]
    if tok6:
        for s in out:
            s.add_tok6(tok6)
    return out
