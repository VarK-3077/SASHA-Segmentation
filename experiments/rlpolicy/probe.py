# Probe before committing: can the segmenter's own answer on a few units decide the rest of a clump?
# The supervised rule's (or the consensus policy's) zoom candidates are grouped into connected clumps on the unit
# grid. In each clump the M best-scored units are segmented first; the rest of the clump is segmented only if a
# probe returned something (the segmenter painted any pixel there), or for units scoring above a rescue
# threshold. Painted core units and border tracing are as in the cascade. Simulator, train slides (out-of-fold
# tables), so the few knobs (M, the score that ranks, the rescue threshold) can be chosen there.
# Also prints how the empty zooms are distributed over clumps (the premise).
# Usage: python -m experiments.rlpolicy.probe [--split train] [--seg3 s2]

import argparse

import numpy as np
from scipy import ndimage as nd

from experiments.rlpolicy import sim as S
from experiments.rlprobe.cost import seconds

S8 = np.ones((3, 3), bool)
VETO = 0.03845


def candidates(s, veto):
    screen = s.tissue & (s.pg >= 0.6)
    want = screen & (s.p3 >= 0.5) & ((s.t_top16 >= veto) if veto else True)
    paint = want & (s.p_core >= 0.99)
    return screen, want & ~paint, paint


def play(s, veto, key, m, rescue, conn=S8):
    screen, cand, paint = candidates(s, veto)
    score = getattr(s, key)
    grp, k = nd.label(cand, structure=conn)
    zoom = np.zeros_like(cand)
    probes = 0
    if k:
        flat, sc, pos = grp.ravel(), score.ravel(), s.pos.ravel()
        order = np.lexsort((-sc, flat))                      # by clump, best score first
        order = order[flat[order] > 0]
        g = flat[order]
        first = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
        rank = np.arange(len(order)) - np.repeat(first, np.diff(np.r_[first, len(order)]))
        probe = order[rank < m]
        probes = len(probe)
        hit = np.zeros(k + 1, bool)
        np.logical_or.at(hit, flat[probe], pos[probe])
        take = np.zeros(cand.size, bool)
        take[probe] = True
        rest = order[rank >= m]
        take[rest] = hit[flat[rest]] | (sc[rest] >= rescue)
        zoom = take.reshape(cand.shape)
    out = S.finish(s, screen, zoom, paint, rounds=1 if (k and probes < int(cand.sum())) else 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train')
    ap.add_argument('--seg3', default=None)
    a = ap.parse_args()
    sl = S.load(a.split, 't1')
    if a.seg3:
        for s in sl:
            s.add_tok3(a.seg3, 's')
    rule = [S.rule(s) for s in sl]
    tum = [i for i, s in enumerate(sl) if s.tumour]
    fmt = lambda m: f"{m['sec']:.3f} s  Dice {m['dice']:.4f}  FROC {m['froc']:.4f}  L2 {m['l2']:.0f}  FP {m['fp']:.2f}"
    lost = lambda recs: sum(max(0, rule[i]['found'] - recs[i]['found']) for i in tum)
    print('rule            ', fmt(S.summarise(rule)), flush=True)
    # the premise: how the candidates and the empty ones fall into clumps
    for name, veto in (('rule', 0.0), ('consensus', VETO)):
        n_c = n_g = n_empty_g = n_empty_u = n_single = 0
        for s in sl:
            _, cand, _ = candidates(s, veto)
            grp, k = nd.label(cand, structure=S8)
            if not k:
                continue
            any_pos = nd.maximum(s.pos, grp, np.arange(1, k + 1)).astype(bool)
            size = nd.sum(cand, grp, np.arange(1, k + 1))
            n_c += int(cand.sum()); n_g += k; n_empty_g += int((~any_pos).sum()); n_empty_u += int(size[~any_pos].sum()); n_single += int((size == 1).sum())
        n = len(sl)
        print(f'{name}: candidates {n_c / n:.1f}/slide in {n_g / n:.1f} clumps ({n_single / n:.1f} single units); clumps where the segmenter returns nothing anywhere: '
              f'{n_empty_g / n:.1f}/slide holding {n_empty_u / n:.1f} units', flush=True)
    keys = ['t_top16', 'p3'] + (['s_top16'] if a.seg3 else [])
    for name, veto in (('rule', 0.0), ('consensus', VETO)):
        base = [S.finish(s, *candidates(s, veto)) for s in sl]
        print(f'--- candidates of the {name}: all segmented   {fmt(S.summarise(base))}  lost {lost(base)}', flush=True)
        for key in keys:
            for m in (1, 2, 3):
                for rescue in (2.0, 0.9, 0.5):
                    recs = [play(s, veto, key, m, rescue) for s in sl]
                    print(f'  probe {m} by {key:8s} rescue {"none" if rescue > 1 else rescue}: {fmt(S.summarise(recs))}  lost {lost(recs)}  Dice below rule on '
                          f'{sum(recs[i]["dice"] < rule[i]["dice"] - 1e-9 for i in tum)}', flush=True)


if __name__ == '__main__':
    main()
