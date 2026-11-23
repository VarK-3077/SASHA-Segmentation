# Exact-scorer tables for "base policy + probe-and-stop": the base policy's zoom candidates are grouped into
# connected clumps, visited best first in batches of B; the best unit of each clump is segmented (the probe); a
# clump whose probe returned something is segmented in full; in a clump whose probe returned nothing only units
# whose own token-head score reaches `rescue` are segmented; the slide stops after a batch in which no probe
# returned anything. Painting and border tracing are the cascade's. A pick is BASE:B:RESCUE with BASE = rule |
# consensus | SEARCH@RUN (an rl3 agent; on train each slide is played by the networks trained without its fold).
# --split test plays the all-train networks and logs one test read per pick in ROOT/TEST_READS.txt.
# Usage: python -m experiments.rlpolicy.probe_eval --split train|test --picks or2_e@lam0.3:5:0.5 ... --out FILE.md

import argparse
import os
import pickle
import time

import numpy as np
from scipy import ndimage as nd

from experiments.rlcascade import report as RP
from experiments.rlpolicy import exact as X
from experiments.rlpolicy import rl4
from experiments.rlpolicy import sim as S
from experiments.rlpolicy import where as W


def play(s, screen, cand, paint, B, rescue):
    grp, k = nd.label(cand, structure=rl4.S8)
    zoom = np.zeros(cand.size, bool)
    rounds = 0
    if k:
        flat, sc, pos = grp.ravel(), s.t_top16.ravel(), s.pos.ravel()
        idx = np.arange(1, k + 1)
        best = nd.maximum(s.t_top16, grp, idx)
        probe = np.array([p[0] * cand.shape[1] + p[1] for p in nd.maximum_position(s.t_top16, grp, idx)])
        order = np.argsort(-best, kind='stable')
        j = 0
        while j < k:
            b = order[j:j + B]
            j, rounds = j + B, rounds + 1
            zoom[probe[b]] = True
            hit = pos[probe[b]]
            for c, h in zip(b, hit):
                zoom[(flat == c + 1) & (h | (sc >= rescue))] = True
            if not hit.any():
                break
    return S.finish(s, screen, zoom.reshape(cand.shape), paint, rounds=max(rounds - 1, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=('train', 'test'), required=True)
    ap.add_argument('--picks', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    slides = S.load(a.split, 't1')
    for s in slides:
        s.add_tok3('s2', 's')
    routes = {s.name: {'rule': S.rule(s)} for s in slides}
    sims, keys = {'rule': [routes[s.name]['rule'] for s in slides]}, []
    for pick in a.picks:
        base, B, rescue = pick.rsplit(':', 2)
        acts = rl4.base_actions(slides, base)
        recs = [play(s, *act, int(B), float(rescue)) for s, act in zip(slides, acts)]
        k = f'{base} + probe {B} / rescue {rescue}'
        keys.append(k)
        sims[k] = recs
        for s, r in zip(slides, recs):
            routes[s.name][k] = r
        if a.split == 'test':
            open(f'{W.ROOT}/TEST_READS.txt', 'a').write(f"{time.strftime('%F %T')} {pick} base policy + probe-and-stop\n")
    ex = X.score_routes(a.split, routes, a.workers)
    head = f"## {a.split}, {len(slides)} slides, exact scorer ({'cross-fitted base policies' if a.split == 'train' else 'all-train policies, test read once'}); seconds = cost model\n"
    tum = [i for i, s in enumerate(slides) if s.tumour]
    rows = ['', '| simulator | s | Dice | FROC | L3 | L2 | L2 on normal slides | L2 on tumour slides | painted | FP comps |', '|---|---|---|---|---|---|---|---|---|---|']
    for k, recs in sims.items():
        m = S.summarise(recs)
        nor = np.mean([r['n_l2'] for i, r in enumerate(recs) if i not in set(tum)])
        rows.append(f"| {k} | {m['sec']:.3f} | {m['dice']:.4f} | {m['froc']:.4f} | {m['l3']:.0f} | {m['l2']:.1f} | {nor:.1f} | {np.mean([recs[i]['n_l2'] for i in tum]):.1f} | {m['paint']:.1f} | {m['fp']:.2f} |")
    text = '\n'.join([head, RP.table(ex, ['rule'] + keys), '', RP.BOOT_HEAD] + [RP.boot_line(ex, k) for k in keys] + rows) + '\n'
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    open(a.out, 'w').write(text)
    pickle.dump(ex, open(a.out[:-3] + '.pkl', 'wb'))
    print(text)


if __name__ == '__main__':
    main()
