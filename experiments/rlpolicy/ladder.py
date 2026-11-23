# Where does the cascade lose? An oracle ladder in the fast simulator: every part of the cascade is either the
# real one or replaced by an oracle that knows the ground truth and what the segmenter would output, and the
# oracle parts are chosen jointly by coordinate ascent on the PPO reward (oracle.py's objective, lam ~ 0).
# The parts:
#   S  the L6 screen        real: read a unit at L3 iff pg >= 0.6        oracle: read exactly the units it wants
#   A  act or leave at L3   real: act iff p3 >= 0.5                       oracle: free
#   P  paint or zoom        real: paint iff p_core >= 0.99, else zoom     oracle: free
#   G  the segmenter        real: the L2 head's dense map                 ideal: a zoomed unit returns the ground truth
# One more row keeps the rule and only removes the units where the segmenter outputs nothing (no ground truth used):
# the map is unchanged by construction, so it shows the time a perfect "will the segmenter find anything" test saves.
# Border tracing stays on with the real segmenter (off with the ideal one: its borders are not in the tables).
# Seconds: base + an L3 read for every unit the screen passes (oracle screen: only the units acted on; no L3 read
# at all when S, A and P are all oracle) + an L2 pass per segmented unit.
# Also a breakdown of what the supervised rule's and the consensus policy's segmented units are, by what the
# segmenter outputs there, and of where the units the oracle needs are lost.
# Usage: python -m experiments.rlpolicy.ladder --split train|test [--workers 8]

import argparse
import os
import pickle
from multiprocessing import get_context

import numpy as np

from experiments.rlcascade import fast_score as FS
from experiments.rlpolicy import oracle as O
from experiments.rlpolicy import sim as S
from experiments.rlpolicy import where as W
from experiments.rlprobe.cost import seconds

OUT = f'{W.ROOT}/ladder'
LEAVE, ZOOM, PAINT = 0, 1, 2
U5 = FS.UNIT5
VETO = 0.03845
SLIDES = None
# name -> (S, A, P, G) with 'r' real / 'o' oracle (G: 'r' real / 'i' ideal)
ROWS = (('all real = the supervised rule', 'rrrr'), ('rule, minus the units where the segmenter outputs nothing', 'rerr'), ('oracle screen only', 'orrr'), ('oracle act/leave only', 'rorr'), ('oracle paint/zoom only', 'rror'),
        ('oracle act/leave + paint/zoom = routing oracle except the screen', 'roor'), ('all routing oracle, real segmenter', 'ooor'),
        ('routing oracle except act/leave', 'oror'), ('routing oracle except paint/zoom', 'oorr'),
        ('real routing, ideal segmenter', 'rrri'), ('all routing oracle, ideal segmenter', 'oooi'))


def gt_units(s):
    """Per unit: the lesion labels it holds (ideal-segmenter candidates). Cached beside the simulator tables."""
    f = f'{OUT}/gtu/{s.name}.npz'
    if os.path.exists(f):
        z = np.load(f)
        return z['unit'], z['lab']
    gt, _ = FS.gt_of(s.name)
    unit, lab = np.zeros(0, np.int64), np.zeros(0, np.int64)
    if gt['eval_mask'] is not None:
        rr, cc = np.nonzero(gt['eval_mask'])
        u = np.minimum(rr // U5, s.hu - 1) * s.wu + np.minimum(cc // U5, s.wu - 1)
        pair = np.unique(np.stack([u, gt['eval_mask'][rr, cc].astype(np.int64)], 1), axis=0)
        unit, lab = pair[:, 0], pair[:, 1]
    os.makedirs(f'{OUT}/gtu', exist_ok=True)
    np.savez(f'{f}.{os.getpid()}.npz', unit=unit, lab=lab)
    os.replace(f'{f}.{os.getpid()}.npz', f)
    return unit, lab


def ideal_outcome(s, zoom, paint):
    """A zoomed unit returns its ground truth: no false positive, every lesion it touches is a candidate at 1.0."""
    act = (zoom | paint).ravel()
    tp = np.zeros(s.n_lab, np.float32)
    fp = []
    if s.tumour:
        labs = s.gtu_lab[act[s.gtu_unit]]
        tp[labs[~np.isin(labs, s.itcs)] - 1] = 1.0
        tp = tp[s.lesions - 1]
    if paint.any():       # a painted group off any lesion is still a false positive
        from scipy import ndimage as nd
        grp, k = nd.label(paint)
        for g in range(1, k + 1):
            if not (s.gt_tis[grp == g] > 0).any():
                fp.append(1.0)
    out = dict(froc=(fp, tp, len(s.lesions)), n_fp=len(fp), found=int((tp > 0).sum()), fp_paint=fp)
    if s.tumour:
        iz, ip, pp = int(s.gt_tis[zoom].sum()), int(s.gt_tis[paint].sum()), int(s.n_tis[paint].sum())
        out['parts'] = (iz, iz, ip, pp)
        out['dice'] = float(2 * (iz + ip) / (iz + pp + s.G)) if iz + pp + s.G > 0 else 1.0
    return out


def play(s, code, act, screen):
    z0, p = (act == ZOOM).reshape(s.tissue.shape), (act == PAINT).reshape(s.tissue.shape)
    if code[3] == 'i':
        z = z0 & ~p
        out = ideal_outcome(s, z, p)
    else:
        z, _ = S.trace(s, z0, p)
        out = S.outcome(s, z, p)
    n_l3 = int(screen.sum()) if code[0] == 'r' else (0 if code[:3] == 'ooo' else int((z0 | p).sum()))
    out.update(n_l3=n_l3, n_l2=int(z.sum()), n_paint=int(p.sum()), sec=seconds(n_l3, int(z.sum())), zoom=z, paint=p)
    return out


def solve(s, code):
    hw = s.tissue.size
    tis = s.tissue.ravel()
    screen = s.tissue & (s.pg >= 0.6)
    scr = screen.ravel() if code[0] == 'r' else tis
    a_ok = (s.p3 >= 0.5).ravel() if code[1] in 're' else np.ones(hw, bool)
    core = (s.p_core >= 0.99).ravel()
    if code[1] == 'e':        # the rule's act/leave, told only whether the segmenter would output anything in the unit
        seen = np.zeros(hw, bool)
        seen[s.cu_unit] = True
        a_ok &= seen | core
    zok = scr & a_ok & (~core if code[2] == 'r' else True)
    pok = scr & a_ok & (core if code[2] == 'r' else True)
    must = scr & a_ok if (code[0] == 'r' and code[1] in 're') else np.zeros(hw, bool)      # nothing left to choose but the type
    gt = s.gt_tis.ravel() > 0
    has, good = np.zeros(hw, bool), np.zeros(hw, bool)
    if code[3] == 'i':
        has, good = gt.copy(), gt.copy()
    else:
        has[s.cu_unit] = True
        good[s.cu_unit[(s.cu_tp > 0) | ((s.cu_lab > 0) & ~np.isin(s.cu_lab, s.itcs))]] = True
    act = np.zeros(hw, np.uint8)
    act[zok & good] = ZOOM
    act[must & (act == LEAVE)] = np.where(zok, ZOOM, PAINT)[must & (act == LEAVE)]
    cand = np.flatnonzero((zok | pok) & (has | gt | must))
    out = play(s, code, act, screen)
    best = O.value(s, out, O.TINY, False)
    if code[:3] in ('rrr', 'rer'):
        return out
    for _ in range(8):
        changed = False
        for u in cand:
            cur = act[u]
            for a in (LEAVE, ZOOM, PAINT):
                if a == cur or (a == LEAVE and must[u]) or (a == ZOOM and not (zok[u] and (has[u] or must[u]))) or (a == PAINT and not (pok[u] and (gt[u] or must[u] or code[2] == 'r'))):
                    continue
                act[u] = a
                o = play(s, code, act, screen)
                v = O.value(s, o, O.TINY, False)
                if v > best + 1e-12:
                    best, out, cur, changed = v, o, a, True
            act[u] = cur
        if not changed:
            break
    return out


def breakdown(s):
    """What the segmented units of the rule / the consensus policy are, and where the oracle's units are lost."""
    need = np.load(f'{W.ROOT}/oracle/cascade_reward/{s.name}.npy').reshape(s.tissue.shape) != LEAVE
    hw = s.tissue.size
    has, tum = np.zeros(hw, bool), np.zeros(hw, bool)
    has[s.cu_unit] = True
    tum[s.cu_unit[s.cu_tp > 0]] = True
    has, tum = has.reshape(s.tissue.shape), tum.reshape(s.tissue.shape)
    rec = {}
    screen = s.tissue & (s.pg >= 0.6)
    for name, veto in (('rule', 0.0), ('consensus', VETO)):
        want = screen & (s.p3 >= 0.5) & ((s.t_top16 >= veto) if veto else True)
        paint = want & (s.p_core >= 0.99)
        r = S.finish(s, screen, want & ~paint, paint)
        z = r['zoom']
        rec[name] = dict(zoom=int(z.sum()), needed=int((z & need).sum()), tumour_not_needed=int((z & tum & ~need).sum()), fp_only=int((z & has & ~tum).sum()),
                         empty=int((z & ~has).sum()), need=int(need.sum()), need_covered=int((need & (z | r['paint'])).sum()),
                         need_unscreened=int((need & ~screen).sum()), need_left_at_l3=int((need & screen & ~(z | r['paint'])).sum()))
    return rec


def job(i):
    s = SLIDES[i]
    s.gtu_unit, s.gtu_lab = gt_units(s)
    keys = ('froc', 'n_fp', 'found', 'sec', 'n_l3', 'n_l2', 'n_paint')
    rows = {}
    for label, code in ROWS:
        o = solve(s, code)
        rows[label] = {k: o[k] for k in keys + (('dice',) if s.tumour else ())}
    return s.name, rows, breakdown(s)


def main():
    global SLIDES
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    SLIDES = S.load(a.split, 't1')
    order = np.argsort([-len(s.cu_unit) for s in SLIDES])
    os.makedirs(f'{OUT}/{a.split}', exist_ok=True)
    got, brk = {}, {}
    with get_context('fork').Pool(a.workers) as pool:
        for k, (name, rows, b) in enumerate(pool.imap_unordered(job, [int(i) for i in order]), 1):
            got[name], brk[name] = rows, b
            if k % 40 == 0:
                print(k, name, flush=True)
    pickle.dump(dict(rows=got, breakdown=brk), open(f'{OUT}/{a.split}/res.pkl', 'wb'))
    n_les = sum(len(s.lesions) for s in SLIDES)
    L = [f'## oracle ladder, {a.split}, {len(SLIDES)} slides, simulator; cleaned Dice, raw FROC; seconds = cost model', '',
         '| screen | act/leave | paint/zoom | segmenter | row | est s/slide | L3 units | L2 units | painted | Dice | FROC | FP comps/slide | lesions found |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    word = dict(r='real', o='oracle', i='ideal', e='real + empty-unit oracle')
    for label, code in ROWS:
        m = S.summarise([got[s.name][label] for s in SLIDES])
        L.append(f"| {word[code[0]]} | {word[code[1]]} | {word[code[2]]} | {word[code[3]]} | {label} | {m['sec']:.3f} | {m['l3']:.0f} | {m['l2']:.0f} | {m['paint']:.1f} | {m['dice']:.4f} | {m['froc']:.4f} | {m['fp']:.2f} | {m['found']}/{n_les} |")
    L += ['', '| policy | segmented units/slide | needed by the oracle | segmenter finds tumour, not needed | segmenter outputs only false positives | segmenter outputs nothing | oracle units/slide | of them covered | lost at the L6 screen | lost at L3 |', '|---|---|---|---|---|---|---|---|---|---|']
    n = len(SLIDES)
    for name in ('rule', 'consensus'):
        t = {k: sum(brk[s.name][name][k] for s in SLIDES) / n for k in brk[SLIDES[0].name][name]}
        L.append(f"| {name} | {t['zoom']:.1f} | {t['needed']:.1f} | {t['tumour_not_needed']:.1f} | {t['fp_only']:.1f} | {t['empty']:.1f} | {t['need']:.1f} | {t['need_covered']:.1f} | {t['need_unscreened']:.1f} | {t['need_left_at_l3']:.1f} |")
    open(f'{OUT}/{a.split}/table.md', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L), flush=True)


if __name__ == '__main__':
    main()
