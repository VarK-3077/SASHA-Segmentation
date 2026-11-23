# Real timed run of the cascade without its L3 stage (rl6.py): tissue mask, the L6 pass, one decision per tissue
# unit from the L6 view alone, then the segmenter on the chosen units with border tracing. No L3 read, no painting.
#   --mode thr | clone | ppo   the supervised L6 head at rl6's re-chosen threshold, its MLP clone, or the PPO screener
#   --seg  l2 | pfa            level-2 token head (rlscreen.run_real's harness) or PFA-ScanNet's level-0 detector
#                              (cascade_pfa's harness; tracing follows PFA's output)
# Cold page cache per slide, one slide at a time, as every other timed row.
# Usage: python -m experiments.rlpolicy.run_l6 --mode ppo --seg l2 --out_dir DIR [--slides ...]

import argparse
import json
import os
import pickle
from multiprocessing import get_context
from types import SimpleNamespace

import numpy as np
import torch

from experiments.lowsep.extract import tokens
from experiments.lowsep.train import names_of
from experiments.rlcascade.run_real_rl import cpu_of, cpu_total, slide_state
from experiments.rlpolicy import rl6
from experiments.rlpolicy import run_staged as RS
from experiments.rlscreen import run_real as RR
from experiments.rlscreen import stoprl as R

CFG = {}


def decide(s):
    """The units to segment, from the L6 view alone."""
    if CFG['mode'] == 'thr':
        return s.tissue & (s.pg >= CFG['t'])
    with torch.no_grad():
        z = (CFG['pi'](rl6.state(s))[:, 0] >= 0).numpy()
    zoom = np.zeros(s.tissue.size, bool)
    zoom[np.flatnonzero(s.tissue.ravel())] = z
    return zoom.reshape(s.tissue.shape)


def play(ep, s):
    none = np.zeros_like(s.tissue)
    z = RS.segment_traced(ep, s, decide(s), none, none)
    return none, int(z.sum()), 1, 0


def seq_l2(a, m, pools, path, tissue, p_g, probs, tab):
    s = slide_state(tissue, p_g)
    ep = RR.RealEp(s, 1.0)
    ep.real = (pools, m, path, probs, tab)
    return play(ep, s)


def seq_pfa(a, m, ctx, tissue, p_g):
    from experiments.cascade_pfa import run_cascade as PC
    s = slide_state(tissue, p_g)
    ep = PC.PFAEp(s, 1.0)
    ep.real = (ctx['pools'], m, ctx['path'], None, None)
    ep.pfa = ctx
    return play(ep, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('thr', 'clone', 'ppo'), required=True)
    ap.add_argument('--seg', choices=('l2', 'pfa'), required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--seg_ckpt', default='/data2/venkatavks/pfa_ckpt/named/tokhead_kaiko_vitl14_stageA_input_l2.pt')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--slides', nargs='*', default=None)
    a = ap.parse_args()
    res = pickle.load(open(f'{rl6.OUT}/res.pkl', 'rb'))
    CFG.update(mode=a.mode, t=res['t'], pi=None if a.mode == 'thr' else rl6.load_pi(res['nets'][a.mode]))
    ra = SimpleNamespace(design='coarse', g_level=6, backbone='kaiko_vitl14', actor='seq', tag=None, ckpt='final', lam=1.0, tau=0.6, tau_zoom=0.5, follow=1, paint=2.0,
                         h=0.02, stages=None, cluster=0, rule='fixed', minpos=1, minfrac=0.0, mode='last', critic=None, eps=0.0, seg_ckpt=a.seg_ckpt,
                         out_dir=a.out_dir, device=a.device, screen_workers=8, seg_workers=8, seg='det', model=f'l6only_{a.mode}')
    R.CFG.update(design='coarse', tz=0.5, S=4.0, fp=1.0, tcs=(2.0,), cluster=False, follow=True)
    R.build_actions()
    os.makedirs(a.out_dir, exist_ok=True)
    ctx = get_context('fork')   # pools before CUDA; readers never touch the GPU
    pools = {'screen': ctx.Pool(8), 'seg': ctx.Pool(8)}
    mine = [os.getpid()] + [p.pid for pl in pools.values() for p in pl._pool]
    m = RR.Models(ra)
    names = a.slides or names_of('test', 6)
    if a.seg == 'l2':
        RR.run_seq = seq_l2
        with torch.no_grad():
            m.seg(torch.zeros(2, 4, 3, 224, 224, dtype=torch.uint8, device=a.device))
            tokens(m.enc, torch.zeros(2, 3, 224, 224, dtype=torch.uint8, device=a.device))
        torch.cuda.synchronize()
        for n in names:
            out = os.path.join(a.out_dir, n + '.npz')
            if os.path.exists(out):
                continue
            c0, o0 = cpu_total(), cpu_of(mine)
            RR.run_slide(n, ra, m, pools, out)
            tj = out[:-4] + '_time.json'
            rec = json.load(open(tj))
            rec.update(load1=os.getloadavg()[0], foreign_cpu_s=(cpu_total() - c0) - (cpu_of(mine) - o0), model=ra.model)
            json.dump(rec, open(tj, 'w'))
    else:
        from experiments.cascade_pfa import pfa_unit as U
        from experiments.cascade_pfa import run_cascade as PC
        PC.run_seq = seq_pfa
        seg = U.UnitSegmenter(U.load_model(a.device), 'det', a.device)
        seg.warmup()
        with torch.no_grad():
            tokens(m.enc, torch.zeros(2, 3, 224, 224, dtype=torch.uint8, device=a.device))
        torch.cuda.synchronize()
        for n in names:
            out = os.path.join(a.out_dir, n + '.npz')
            if not os.path.exists(out):
                PC.run_slide(n, ra, m, seg, pools, out)
    json.dump(dict(mode=a.mode, seg=a.seg, t=res['t'], seg_ckpt=a.seg_ckpt), open(os.path.join(a.out_dir, 'run.json'), 'w'), indent=1)
    open(os.path.join(a.out_dir, 'DONE'), 'w').write(f'{len(names)} slides\n')


if __name__ == '__main__':
    main()
