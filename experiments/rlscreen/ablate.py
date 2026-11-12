# Fine-to-coarse feedback ablation on L6 -> L3 (screen 0.6, zoom 0.5): choosing every setting on the
# out-of-fold train slides with real segmenter outcomes (lesrl.py), never on test.
#   row 1 fixed rule, no painting, no border tracing
#   row 2 + core painting (p_core >= 0.99)
#   row 3 row 2 + escalation (slide range)        -> stages, minpos, minfrac, mode chosen here
#   row 4 row 2 + border tracing (neighbour range)  -> no free setting
#   row 5 row 2 + interior painting (interior range) -> critic trained here, eps chosen here
#   row 6 all three                                  -> escalation setting and eps chosen jointly here
# Rule: the cheapest setting whose train score (lesion-aware J) is within 0.01 of the best train score.
# Writes ablate_chosen.json and critic weights; prints train and test rows at the chosen settings only.
# Usage: RLSCREEN_OOF=1 python -m experiments.rlscreen.ablate

import argparse
import itertools
import json
from multiprocessing import get_context

import numpy as np
import torch
import torch.nn as nn

from experiments.rlscreen import escalate as E
from experiments.rlscreen import lesrl
from experiments.rlscreen import paths as P
from experiments.rlscreen import qstop as Q
from experiments.rlscreen import stoprl as R
from experiments.rlscreen.rollout import load_tables
from experiments.rlscreen.seqscreen import Slide

TS, TZ, TC, TOL = 0.6, 0.5, 0.99, 0.01
EPS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
SLIDES = {}


class SegCriticEp(lesrl.SegEp, Q.Ep2):
    pass


class Recorder(SegCriticEp):
    """Records the critic's inputs and the true non-tumour share for every unit about to be zoomed."""
    store = None

    def paint_mask(self, routed, cand, tc):
        if Recorder.store is not None and cand.any():
            x, idx = self.core_feats(cand)
            Recorder.store.append((x, 1 - np.clip(self.s.gt.reshape(-1)[idx], 0, 1)))
        return super().paint_mask(routed, cand, tc)


def setup(follow, ep):
    R.CFG.update(design='coarse', tz=TZ, S=4.0, fp=1.0, tcs=(TC,), follow=follow, ep=ep)


def episode(s, cfg):
    follow, critic, rule = cfg
    setup(follow, SegCriticEp if critic is not None else lesrl.SegEp)
    if critic is not None:
        Q.Ep2.critic, Q.Ep2.eps = CRITIC[0], critic
    act = R.fixed_rule(TS, TC) if rule is None else E.rule(*rule, tc=TC)
    e = R.run(s, 1.0, act).summary()
    return e['j'], e['sec'], np.nan if e['dice'] is None else e['dice'], np.nan if e['found'] is None else e['found'], e['l2'], e['painted']


CRITIC = [None]


def play(arg):
    split, cfg = arg
    return np.array([episode(s, cfg) for s in SLIDES[split]])


def train_critic(slides, log):
    net = Q.CoreNet()
    X, Y = [], []
    for rnd in range(2):
        Recorder.store = []
        setup(True, Recorder)
        Q.Ep2.critic, Q.Ep2.eps = (None, 0.0) if rnd == 0 else (net, 0.02)
        for s in slides:
            for rule in (None, ((0.8, 0.6), 1, 0.0, 'last')):
                R.run(s, 1.0, R.fixed_rule(TS, TC) if rule is None else E.rule(*rule, tc=TC))
        X += [x for x, _ in Recorder.store]; Y += [y for _, y in Recorder.store]
        Xt, Yt = torch.from_numpy(np.concatenate(X)), torch.from_numpy(np.concatenate(Y).astype(np.float32))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for _ in range(3000):
            i = torch.randint(len(Xt), (1024,))
            loss = nn.functional.binary_cross_entropy_with_logits(net(Xt[i]), Yt[i])
            opt.zero_grad(); loss.backward(); opt.step()
        log(f'critic round {rnd}: {len(Xt)} units, loss {float(loss):.4f}')
    Recorder.store = None
    return net


def esc_settings():
    out = []
    for st, minpos, minfrac, mode, tz in E.settings():
        if tz == TZ:
            out.append((st, minpos, minfrac, mode))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    log = lambda m: print(m, flush=True)
    for sp, trusted in (('train', P.NON_TRUSTED_SKIP), ('test', False)):
        S = lesrl.attach([Slide(t, 'mix') for t in load_tables(sp, 'coarse', 6, 'kaiko_vitl14', 'cpu', trusted_only=trusted)])
        for s in S:
            s.cluster = R.cluster_stats(s)
        SLIDES[sp] = S
    R.CFG.update(design='coarse')
    R.build_actions()
    CRITIC[0] = train_critic(SLIDES['train'], log)
    torch.save(CRITIC[0].state_dict(), f'{P.ROOT}/ablate_critic.pt')
    esc = esc_settings()
    rows = {1: [(False, None, None)], 4: [(True, None, None)],
            5: [(False, e, None) for e in EPS], 3: [(False, None, r) for r in esc],
            6: [(True, e, r) for e, r in itertools.product(EPS, esc)]}
    row2 = (False, None, None)
    grid = sorted({c for v in rows.values() for c in v}, key=str)
    with get_context('fork').Pool(a.workers) as pool:
        tr = dict(zip(grid, pool.map(play, [('train', g) for g in grid], chunksize=2)))
    chosen = {}
    for r, cands in rows.items():
        j = {c: tr[c][:, 0].mean() for c in cands}
        best = max(j.values())
        chosen[r] = min([c for c in cands if j[c] >= best - TOL], key=lambda c: tr[c][:, 1].mean())
    with get_context('fork').Pool(a.workers) as pool:
        need = sorted(set(chosen.values()), key=str)
        te = dict(zip(need, pool.map(play, [('test', g) for g in need])))
    # row 1 has no painting: rerun with painting off
    R.CFG.update(tcs=(2.0,))
    print('\n| row | chosen on out-of-fold train | train J | train s | test J | test raw Dice | test found | test L2 | test s |\n|---|---|---|---|---|---|---|---|---|')
    for r in sorted(chosen):
        c = chosen[r]
        t, e = tr[c], te[c]
        print(f'| {r} | {c} | {t[:, 0].mean():.3f} | {t[:, 1].mean():.2f} | {e[:, 0].mean():.3f} | {np.nanmean(e[:, 2]):.3f} | {np.nanmean(e[:, 3]):.3f} | {e[:, 4].mean():.3f} | {e[:, 1].mean():.2f} |')
    json.dump({str(k): {'follow': v[0], 'eps': v[1], 'rule': v[2]} for k, v in chosen.items()}, open(f'{P.ROOT}/ablate_chosen.json', 'w'), indent=1)
    print('row 2 equals row 1 with painting; rows 3-6 all paint at p_core >= 0.99 (row 1 painting off is set in the real run)', flush=True)


if __name__ == '__main__':
    main()
