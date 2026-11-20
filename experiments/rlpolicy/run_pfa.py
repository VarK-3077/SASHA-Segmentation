# A PPO screening policy (rl3.py) with PFA-ScanNet at level 0 as the segmenter, timed end to end: cascade_pfa's
# run_cascade (its tissue mask, global pass, PFA unit segmenter, map and time record, cold page cache per slide)
# with the policy's decisions in place of the rlscreen actors. Every L3 screen feeds both L3 heads, the L6 and L3
# policies decide, and border tracing follows what PFA returned for each segmented unit.
# Usage: TOK_LEVEL=2 python -m experiments.rlpolicy.run_pfa --seg det|fused --policy SEARCH@RUN --tok3 TAG --out_dir DIR

import argparse
import json
import os
from multiprocessing import get_context

import torch

from experiments.cascade_pfa import pfa_unit as U
from experiments.cascade_pfa import run_cascade as PC
from experiments.lowsep.extract import tokens
from experiments.lowsep.train import names_of
from experiments.rlpolicy import run_staged as RS
from experiments.rlscreen import run_real as RR
from experiments.rlscreen import stoprl as R


def run_seq(a, m, ctx, tissue, p_g):
    s = RS.new_state(m, tissue, p_g)
    ep = PC.PFAEp(s, 1.0)
    ep.real = (ctx['pools'], m, ctx['path'], None, None)
    ep.pfa = ctx
    return RS.ppo_play(ep, s, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seg', choices=('det', 'fused'), required=True)
    ap.add_argument('--policy', required=True, help='SEARCH@RUN, a PPO policy saved by rl3.py')
    ap.add_argument('--tok3', required=True)
    ap.add_argument('--pfa_ckpt', default=U.CKPT)
    ap.add_argument('--slides', nargs='*', default=None)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--screen_workers', type=int, default=8)
    ap.add_argument('--seg_workers', type=int, default=8)
    a = ap.parse_args()
    a.design, a.g_level, a.backbone, a.actor, a.tag, a.ckpt, a.critic, a.eps, a.lam = 'coarse', 6, 'kaiko_vitl14', 'seq', None, 'final', None, 0.0, 1.0
    a.seg_ckpt = '/data2/venkatavks/pfa_ckpt/named/tokhead_kaiko_vitl14_stageA_input_l2.pt'
    R.CFG.update(design='coarse', tz=0.5, S=4.0, fp=1.0, tcs=(0.99,), cluster=False, follow=True)
    R.build_actions()
    os.makedirs(a.out_dir, exist_ok=True)
    ctx = get_context('fork')   # pools before CUDA; readers never touch the GPU
    pools = {'screen': ctx.Pool(a.screen_workers), 'seg': ctx.Pool(a.seg_workers)}
    m = RR.Models(a)
    RS.attach_policy(m, a.policy, a.tok3, a.device)
    PC.run_seq = run_seq
    seg = U.UnitSegmenter(U.load_model(a.device, a.pfa_ckpt), a.seg, a.device)
    seg.warmup()
    with torch.no_grad():
        tokens(m.enc, torch.zeros(2, 3, 224, 224, dtype=torch.uint8, device=a.device))
        m.tok3(m.embed([torch.zeros(3, 224, 224, dtype=torch.uint8).numpy()] * 2, m.pca_3).float())
    torch.cuda.synchronize()
    names = a.slides or names_of('test', 6)
    for n in names:
        out = os.path.join(a.out_dir, n + '.npz')
        if not os.path.exists(out):
            PC.run_slide(n, a, m, seg, pools, out)
    json.dump(dict(vars(a), gpu=torch.cuda.get_device_name(0), peak_mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30),
              open(os.path.join(a.out_dir, 'run.json'), 'w'), indent=1)
    open(os.path.join(a.out_dir, 'DONE'), 'w').write(f'{len(names)} slides\n')


if __name__ == '__main__':
    main()
