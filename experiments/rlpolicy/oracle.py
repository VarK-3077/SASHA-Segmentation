# The routing ceiling: with the segmenter fixed, which units should be segmented, painted or left to get the
# highest score in the least time, if the router knew the ground truth and what the segmenter would output?
# Searched per slide in the fast simulator (the dense level-2 map of the real segmenter, out of fold on train
# slides): coordinate ascent over one action per unit (leave / zoom / paint) on the reward the PPO agents are
# trained on, with a price of time lam:
#     J = cleaned Dice + w_les x sum of the best candidate score of each lesion found
#         - w_fp x sum of the false-positive candidates' scores - lam x seconds
#   aware (default): the segmenter's false positives count as they are.
#   blind:           the search pretends the segmenter makes no false positive (its false-positive pixels and
#                    candidates are ignored when choosing); the route is then scored both as it really is and
#                    under that pretence (Dice without the segmenter's false-positive pixels, FROC = sensitivity).
# Where the router may act:
#   direct:  any tissue unit, no L3 read, no border tracing -> seconds = base + L2 units only (the floor).
#   cascade: the supervised L6 screen (pg >= 0.6) is kept, every screened unit is read at L3 and gets one action,
#            border tracing runs afterwards as in the cascade -> what a perfect L3 decision would give. Its final
#            actions are the labels for policies trained on "manual" labels (rl3.py --init oracle).
# Usage: python -m experiments.rlpolicy.oracle --split train|test [--workers 8]

import argparse
import os
import pickle
from multiprocessing import get_context

import numpy as np

from experiments.rlpolicy import sim as S
from experiments.rlpolicy import where as W
from experiments.rlprobe.cost import seconds

OUT = f'{W.ROOT}/oracle'
LEAVE, ZOOM, PAINT = 0, 1, 2
W_LES, W_FP, TINY, CORE = 8.0, 0.0497, 1e-3, 0.9
# (tag, where, lam, blind, w_fp, core): 'reward' = the PPO reward's weights, 'dice' = no price on false-positive candidates
# (lesions first, then Dice with every false-positive pixel counted), 'blind' = the no-false-positive pretence;
# '_core' = a unit may be painted only if at least CORE of its tissue is tumour (painting as the cascade means it:
# without it the oracle also paints units around lesions the segmenter misses, which is more than routing)
VARIANTS = (('direct_reward', 'direct', TINY, False, W_FP, 0.0), ('direct_dice', 'direct', TINY, False, 0.0, 0.0), ('direct_blind', 'direct', TINY, True, W_FP, 0.0),
            ('direct_reward_core', 'direct', TINY, False, W_FP, CORE), ('direct_blind_core', 'direct', TINY, True, W_FP, CORE),
            ('cascade_reward', 'cascade', TINY, False, W_FP, 0.0), ('cascade_dice', 'cascade', TINY, False, 0.0, 0.0), ('cascade_blind', 'cascade', TINY, True, W_FP, 0.0),
            ('cascade_reward_core', 'cascade', TINY, False, W_FP, CORE), ('cascade_blind_core', 'cascade', TINY, True, W_FP, CORE),
            ('cascade_reward_lam0.3', 'cascade', 0.3, False, W_FP, 0.0), ('cascade_reward_lam3', 'cascade', 3.0, False, W_FP, 0.0))
SLIDES = None


def play(s, mode, act):
    z, p = (act == ZOOM).reshape(s.tissue.shape), (act == PAINT).reshape(s.tissue.shape)
    if mode == 'cascade':
        return S.finish(s, s.tissue & (s.pg >= 0.6), z, p)
    out = S.outcome(s, z, p)
    out.update(n_l3=0, n_l2=int(z.sum()), n_paint=int(p.sum()), zoom=z, paint=p, sec=seconds(0, int(z.sum())))
    return out


def blind_dice(s, out):
    iz, pz, ip, pp = out['parts']
    return 2 * (iz + ip) / (iz + pp + s.G) if s.G else 1.0


def value(s, out, lam, blind, w_fp=W_FP):
    fp, tp, _ = out['froc']
    dice = (blind_dice(s, out) if blind else out['dice']) if s.tumour else 0.0
    return dice + W_LES * float(np.sum(tp)) - w_fp * float(np.sum(out['fp_paint'] if blind else fp)) - lam * out['sec']


def solve(s, mode, lam, blind, w_fp=W_FP, core=0.0):
    hw = s.tissue.size
    dom = (s.tissue if mode == 'direct' else s.tissue & (s.pg >= 0.6)).ravel()
    has, good = np.zeros(hw, bool), np.zeros(hw, bool)
    has[s.cu_unit] = True
    good[s.cu_unit[(s.cu_tp > 0) | ((s.cu_lab > 0) & ~np.isin(s.cu_lab, s.itcs))]] = True
    gt = (s.gt_tis.ravel() > 0) & (s.gt_tis.ravel() >= core * s.n_tis.ravel())
    cand = np.flatnonzero(dom & (has | gt))          # gt: units that may be painted
    act = np.zeros(hw, np.uint8)
    act[dom & good] = ZOOM
    out = play(s, mode, act)
    best = value(s, out, lam, blind, w_fp)
    for _ in range(8):
        changed = False
        for u in cand:
            cur = act[u]
            for a in (LEAVE, ZOOM, PAINT):
                if a == cur or (a == PAINT and not gt[u]) or (a == ZOOM and not has[u]):
                    continue
                act[u] = a
                o = play(s, mode, act)
                v = value(s, o, lam, blind, w_fp)
                if v > best + 1e-12:
                    best, out, cur, changed = v, o, a, True
            act[u] = cur
        if not changed:
            break
    return out


def job(i):
    s = SLIDES[i]
    rec = {}
    for tag, mode, lam, blind, w_fp, core in VARIANTS:
        o = solve(s, mode, lam, blind, w_fp, core)
        final = np.where(o['paint'], PAINT, np.where(o['zoom'], ZOOM, LEAVE)).astype(np.uint8)
        keep = {k: o[k] for k in ('froc', 'n_fp', 'found', 'sec', 'n_l3', 'n_l2', 'n_paint')}
        if s.tumour:
            keep.update(dice=o['dice'], blind_dice=blind_dice(s, o))
        rec[tag] = dict(out=keep, act=final)
    d = play(s, 'direct', np.where(s.tissue.ravel(), ZOOM, LEAVE).astype(np.uint8))
    rec['dense'] = dict(out={k: d[k] for k in ('froc', 'n_fp', 'found', 'sec', 'n_l3', 'n_l2', 'n_paint') + (('dice',) if s.tumour else ())})
    if s.tumour:
        rec['dense']['out']['blind_dice'] = blind_dice(s, d)
    r = S.rule(s)
    rec['rule'] = dict(out={k: r[k] for k in ('froc', 'n_fp', 'found', 'sec', 'n_l3', 'n_l2', 'n_paint') + (('dice',) if s.tumour else ())})
    return s.name, rec


def main():
    global SLIDES
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    SLIDES = S.load(a.split)
    order = np.argsort([-len(s.cu_unit) for s in SLIDES])          # the slow slides first
    os.makedirs(f'{OUT}/{a.split}', exist_ok=True)
    got = {}
    with get_context('fork').Pool(a.workers) as pool:
        for k, (name, rec) in enumerate(pool.imap_unordered(job, [int(i) for i in order]), 1):
            got[name] = rec
            for tag in [t for t, *_ in VARIANTS if t.startswith('cascade')]:
                os.makedirs(f'{OUT}/{tag}', exist_ok=True)
                np.save(f'{OUT}/{tag}/{name}.npy', rec[tag]['act'])
            if k % 20 == 0:
                print(k, name, flush=True)
    pickle.dump(got, open(f'{OUT}/{a.split}/res.pkl', 'wb'))
    n_les = sum(len(s.lesions) for s in SLIDES)
    rows = [('supervised rule (L6 -> L3 -> L2, as run)', 'rule'), ('dense L2 on every tissue unit', 'dense')] + [(t, t) for t, *_ in VARIANTS]
    lines = [f'## routing ceiling, {a.split}, {len(SLIDES)} slides, simulator (dense map of the real L2 segmenter{", out of fold" if a.split == "train" else ""}); seconds = cost model', '',
             '| routing | est s/slide | L3 units | L2 units | painted | Dice | FROC | FP comps/slide | lesions found | Dice without segmenter FPs | sensitivity |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for label, tag in rows:
        recs = [got[s.name][tag]['out'] for s in SLIDES]
        m = S.summarise(recs)
        bd = float(np.mean([r['blind_dice'] for r in recs if 'blind_dice' in r])) if any('blind_dice' in r for r in recs) else float('nan')
        lines.append(f"| {label} | {m['sec']:.3f} | {m['l3']:.0f} | {m['l2']:.0f} | {m['paint']:.1f} | {m['dice']:.4f} | {m['froc']:.4f} | {m['fp']:.2f} | {m['found']}/{n_les} | {bd:.4f} | {m['found'] / n_les:.4f} |")
    open(f'{OUT}/{a.split}/table.md', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
