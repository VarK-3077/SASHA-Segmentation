# The cascade with the level-0 token head as the segmenter (tok0_unit.py), timed end to end in cascade_pfa's
# harness (its tissue mask, global pass, map and time record, cold page cache per slide): either the supervised
# rule r4_trace (default) or a PPO screening policy (--policy SEARCH@RUN --tok3 TAG). Screening is unchanged;
# every zoomed 2048-px unit is segmented at level 0 instead of level 2, and border tracing follows the level-0
# map thresholded at the level-0 head's own validation-chosen 0.7. Maps are saved at 32 px like the PFA rows.
# Usage: TOK_LEVEL=2 python -m experiments.rlpolicy.run_tok0 --out_dir DIR [--policy SEARCH@RUN --tok3 TAG]

import argparse
import json
import os
import time
from multiprocessing import get_context

import numpy as np
import torch

from experiments.cascade_pfa import pfa_unit as U
from experiments.cascade_pfa import run_cascade as PC
from experiments.lowsep.extract import tokens
from experiments.lowsep.train import names_of
from experiments.rlpolicy import run_staged as RS
from experiments.rlpolicy import tok0_unit as T0
from experiments.rlscreen import run_real as RR
from experiments.rlscreen import stoprl as R

assert U.TILE == T0.TILE and U.STRIDE == 32


class Tok0Ep(PC.PFAEp):
    """PFAEp with the level-0 head's threshold in the border-tracing feedback."""

    def on_zoom(self, zm):
        units = [tuple(map(int, u)) for u in np.argwhere(zm)]
        if not units:
            return
        c = self.pfa
        maps = c['seg'].run(c['pools']['seg'], c['path'], units)
        t0 = time.perf_counter()
        s, g = self.s, U.TILE
        for (r, cc), m in maps.items():
            c['probs'][r * g:(r + 1) * g, cc * g:(cc + 1) * g] = m
            o = U.outcome(m, T0.THR)
            s.frac[r, cc] = s.seg_frac[r, cc] = o[0]
            s.pos[r, cc] = o[0] > 0
            s.edges[r, cc] = o[1:] > 0
            s.edge_frac[r, cc] = o[1:]
        c['timer']['feedback'] += time.perf_counter() - t0


def policy_seq(a, m, ctx, tissue, p_g):
    s = RS.new_state(m, tissue, p_g)
    ep = Tok0Ep(s, 1.0)
    ep.real = (ctx['pools'], m, ctx['path'], None, None)
    ep.pfa = ctx
    return RS.ppo_play(ep, s, m)


def tap_tissue(seg, orig):
    def f(slide):
        seg.t5 = orig(slide)
        return seg.t5
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', default=None, help='SEARCH@RUN, a PPO policy saved by rl3.py; default: the supervised rule')
    ap.add_argument('--tok3', default=None)
    ap.add_argument('--slides', nargs='*', default=None)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--screen_workers', type=int, default=8)
    ap.add_argument('--seg_workers', type=int, default=8)
    a = ap.parse_args()
    a.seg, a.design, a.g_level, a.backbone, a.actor, a.tag, a.ckpt, a.critic, a.eps, a.lam = 'tok0', 'coarse', 6, 'kaiko_vitl14', 'seq', None, 'final', None, 0.0, 1.0
    a.tau, a.tau_zoom, a.follow, a.paint, a.h, a.stages, a.cluster, a.rule, a.minpos, a.minfrac, a.mode = 0.6, 0.5, 1, 0.99, 0.02, None, 0, 'fixed', 1, 0.0, 'last'
    a.seg_ckpt = T0.CKPT                       # run_real's models load this head: the level-0 one
    R.CFG.update(design='coarse', tz=0.5, S=4.0, fp=1.0, tcs=(0.99,), cluster=False, follow=True)
    R.build_actions()
    os.makedirs(a.out_dir, exist_ok=True)
    ctx = get_context('fork')   # pools before CUDA; readers never touch the GPU
    pools = {'screen': ctx.Pool(a.screen_workers), 'seg': ctx.Pool(a.seg_workers)}
    m = RR.Models(a)
    seg = T0.UnitSegmenter(m.seg, a.device)
    PC.tissue_mask = tap_tissue(seg, PC.tissue_mask)
    PC.PFAEp = Tok0Ep
    if a.policy:
        RS.attach_policy(m, a.policy, a.tok3, a.device)
        PC.run_seq = policy_seq
    seg.warmup()
    with torch.no_grad():
        tokens(m.enc, torch.zeros(2, 3, 224, 224, dtype=torch.uint8, device=a.device))
        if a.policy:
            m.tok3(m.embed([np.zeros((3, 224, 224), np.uint8)] * 2, m.pca_3).float())
    torch.cuda.synchronize()
    names = a.slides or names_of('test', 6)
    for n in names:
        out = os.path.join(a.out_dir, n + '.npz')
        if not os.path.exists(out):
            PC.run_slide(n, a, m, seg, pools, out)
    json.dump(dict({k: v for k, v in vars(a).items()}, thr=T0.THR, gpu=torch.cuda.get_device_name(0)), open(os.path.join(a.out_dir, 'run.json'), 'w'), indent=1)
    open(os.path.join(a.out_dir, 'DONE'), 'w').write(f'{len(names)} slides\n')


if __name__ == '__main__':
    main()
