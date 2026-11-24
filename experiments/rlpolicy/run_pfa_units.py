# PFA-ScanNet's level-0 detector on EVERY tissue unit of a slide, through the cascade's own unit segmenter
# (cascade_pfa: one 2708-px ROI per 2048-px unit), so that the simulator can be built for PFA as the segmenter:
# the saved map is what the cascade would get if it zoomed everywhere, and the side file holds what a policy sees
# after each zoom (painted share and border strips per unit, pfa_unit.outcome). No routing, no tracing, no
# painting; not a timing run. Default slides: the 222 trusted train slides. NOTE: PFA was trained on the train
# slides, so these maps are in-sample for the segmenter (there are no out-of-fold PFA models).
# Usage: python -m experiments.rlpolicy.run_pfa_units --out_dir DIR [--split train] [--slides ...] [--part i/n]

import argparse
import json
import os
from multiprocessing import get_context
from types import SimpleNamespace

import numpy as np
import torch

from experiments.lowsep.extract import tokens
from experiments.rlcascade.run_real_rl import slide_state
from experiments.rlscreen import run_real as RR
from experiments.rlscreen import stoprl as R

STASH = {}


def seq_all(a, m, ctx, tissue, p_g):
    from experiments.cascade_pfa import run_cascade as PC
    s = slide_state(tissue, p_g)
    ep = PC.PFAEp(s, 1.0)
    ep.real = (ctx['pools'], m, ctx['path'], None, None)
    ep.pfa = ctx
    ep.on_zoom(tissue.copy())
    STASH['side'] = dict(seg_frac=np.asarray(s.seg_frac, np.float32), seg_edges=np.asarray(s.edge_frac, np.float32), tissue=tissue)
    none = np.zeros_like(tissue)
    return none, int(tissue.sum()), 1, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--split', default='train')
    ap.add_argument('--slides', nargs='*', default=None)
    ap.add_argument('--part', default=None, help='i/n: every n-th slide starting at i (to split the work over processes)')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seg_ckpt', default='/data2/venkatavks/pfa_ckpt/named/tokhead_kaiko_vitl14_stageA_input_l2.pt')
    a = ap.parse_args()
    from experiments.cascade_pfa import pfa_unit as U
    from experiments.cascade_pfa import run_cascade as PC
    from experiments.rlpolicy.simtab import names_of
    ra = SimpleNamespace(design='coarse', g_level=6, backbone='kaiko_vitl14', actor='seq', tag=None, ckpt='final', lam=1.0, tau=0.6, tau_zoom=0.5, follow=1, paint=2.0,
                         h=0.02, stages=None, cluster=0, rule='fixed', minpos=1, minfrac=0.0, mode='last', critic=None, eps=0.0, seg_ckpt=a.seg_ckpt,
                         out_dir=a.out_dir, device=a.device, screen_workers=8, seg_workers=8, seg='det', model='pfa_all_units')
    R.CFG.update(design='coarse', tz=0.5, S=4.0, fp=1.0, tcs=(2.0,), cluster=False, follow=True)
    R.build_actions()
    os.makedirs(f'{a.out_dir}/side', exist_ok=True)
    ctx = get_context('fork')
    pools = {'screen': ctx.Pool(8), 'seg': ctx.Pool(8)}
    m = RR.Models(ra)
    PC.run_seq = seq_all
    seg = U.UnitSegmenter(U.load_model(a.device), 'det', a.device)
    seg.warmup()
    with torch.no_grad():
        tokens(m.enc, torch.zeros(2, 3, 224, 224, dtype=torch.uint8, device=a.device))
    names = a.slides or names_of(a.split)
    if a.part:
        i, n = map(int, a.part.split('/'))
        names = names[i::n]
    for n in names:
        out = os.path.join(a.out_dir, n + '.npz')
        if os.path.exists(out) and os.path.exists(f'{a.out_dir}/side/{n}.npz'):
            continue
        PC.run_slide(n, ra, m, seg, pools, out)
        np.savez(f'{a.out_dir}/side/{n}.npz', **STASH['side'])
        t = json.load(open(out[:-4] + '_time.json'))
        print(f"{n}: {int(STASH['side']['tissue'].sum())} units, {t['wall']:.0f} s", flush=True)
    open(os.path.join(a.out_dir, f'DONE{("_" + a.part.replace("/", "of")) if a.part else ""}'), 'w').write(f'{len(names)} slides\n')


if __name__ == '__main__':
    main()
