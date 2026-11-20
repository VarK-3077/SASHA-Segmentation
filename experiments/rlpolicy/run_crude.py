# Real timed run of the crude CLS-token cascade on the test slides, in the rule's own harness (rlscreen.run_real:
# same reader pools, ViT-L, level-2 token head, cold page cache per slide, one slide at a time). Per slide: tissue
# mask, an L3 read of every tissue unit (no L6 pass), the CLS token through the two-layer MLP, then the L2
# segmenter on the units at or above the threshold. No painting, no border tracing.
# Usage: python -m experiments.rlpolicy.run_crude --out_dir DIR [--slides ...]

import argparse
import json
import os
from multiprocessing import get_context
from types import SimpleNamespace

import numpy as np
import torch

from experiments.lowsep.cascade_pipe import read_screen
from experiments.lowsep.extract import tokens
from experiments.lowsep.train import names_of
from experiments.rlcascade.run_real_rl import cpu_of, cpu_total, slide_state
from experiments.rlpolicy import cls3
from experiments.rlscreen import run_real as RR
from experiments.rlscreen import stoprl as R


def no_global(m, path, level, t5, hu, wu, pool):
    """The crude cascade has no L6 pass: empty global view, every unit passes."""
    return torch.zeros((hu, wu, 516), device=m.dev), np.zeros((hu, wu, 4), np.float32)


def run(a, m, pools, path, tissue, p_g, probs, tab):
    s = slide_state(tissue, p_g)
    io = RR.RealEp(s, 1.0)
    io.real = (pools, m, path, probs, tab)
    rc = np.argwhere(tissue)
    zoom = np.zeros_like(tissue)
    for k in range(0, len(rc), 256):
        items = pools['screen'].map(read_screen, [(path, 3, int(r), int(c)) for r, c in rc[k:k + 256]])
        with torch.no_grad():
            p = torch.sigmoid(m.cls_head(cls3.cls_tokens(m.enc, [it[2] for it in items]))).cpu().numpy()
        for (r, c, _), v in zip(items, p):
            zoom[r, c] = v >= a.thr
    io.on_zoom(zoom)
    return np.zeros_like(tissue), int(zoom.sum()), 1, int(tissue.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--seg_ckpt', default='/data2/venkatavks/pfa_ckpt/named/tokhead_kaiko_vitl14_stageA_input_l2.pt')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--slides', nargs='*', default=None)
    a, _ = ap.parse_known_args()
    thr = json.load(open(f'{cls3.OUT}/pick.json'))['t']
    ra = SimpleNamespace(design='coarse', g_level=6, backbone='kaiko_vitl14', actor='seq', tag=None, ckpt='final', lam=1.0, tau=0.6, tau_zoom=0.5, follow=0, paint=2.0,
                         h=0.02, stages=None, cluster=0, rule='fixed', minpos=1, minfrac=0.0, mode='last', critic=None, eps=0.02, seg_ckpt=a.seg_ckpt,
                         out_dir=a.out_dir, device=a.device, screen_workers=8, seg_workers=8, model='crude', thr=thr)
    R.CFG.update(design='coarse', tz=0.5, S=4.0, fp=1.0, tcs=(2.0,), cluster=False, follow=False)
    R.build_actions()
    os.makedirs(a.out_dir, exist_ok=True)
    ctx = get_context('fork')   # pools before CUDA; readers never touch the GPU
    pools = {'screen': ctx.Pool(8), 'seg': ctx.Pool(8)}
    mine = [os.getpid()] + [p.pid for pl in pools.values() for p in pl._pool]
    m = RR.Models(ra)
    sd = torch.load(f'{cls3.OUT}/f-1.pt', map_location=a.device)
    m.cls_head = cls3.Head(sd['mu'], sd['sd']).to(a.device).eval()
    m.cls_head.load_state_dict(sd)
    RR.global_pass = no_global
    RR.run_seq = run
    with torch.no_grad():
        m.seg(torch.zeros(2, 4, 3, 224, 224, dtype=torch.uint8, device=a.device))
        m.cls_head(cls3.cls_tokens(m.enc, [np.zeros((3, 224, 224), np.uint8)] * 2))
        tokens(m.enc, torch.zeros(2, 3, 224, 224, dtype=torch.uint8, device=a.device))
    torch.cuda.synchronize()
    for n in a.slides or names_of('test', 6):
        out = os.path.join(a.out_dir, n + '.npz')
        if os.path.exists(out):
            continue
        c0, o0 = cpu_total(), cpu_of(mine)
        RR.run_slide(n, ra, m, pools, out)
        tj = out[:-4] + '_time.json'
        rec = json.load(open(tj))
        rec.update(load1=os.getloadavg()[0], foreign_cpu_s=(cpu_total() - c0) - (cpu_of(mine) - o0), model='crude')
        json.dump(rec, open(tj, 'w'))
    json.dump(dict(thr=thr, seg_ckpt=a.seg_ckpt), open(os.path.join(a.out_dir, 'run.json'), 'w'), indent=1)
    open(os.path.join(a.out_dir, 'DONE'), 'w').write('done\n')


if __name__ == '__main__':
    main()
