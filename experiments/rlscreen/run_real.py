# Runs a screening policy on the test slides with real reads, the real screen backbone and heads,
# and the real level-2 token head, timed end to end like the lowsep cascades (solo, cold page
# cache per slide, persistent reader pools). Per slide: tissue mask; the global pass (whole slide
# at level 3 for the full design, at the global level for the coarse one) gives every unit its
# pooled tokens and head probabilities; then rounds: the actor picks units to screen at level 3
# and units to zoom, both read concurrently, the screen is embedded and scored by the per-unit
# level-3 head, the zoomed units are segmented and their outcome (tumour fraction, border strips at
# the token head's 0.4) is fed back; until the actor stops. The map is saved in the token head's
# layout for pfa_scannet.clean_sweep, with a per-slide time json.
# Usage: TOK_LEVEL=2 python -m experiments.rlscreen.run_real --design coarse --g_level 6 --actor policy --tag RUN --lam 1 --out_dir DIR

import argparse
import json
import os
import time
from multiprocessing import get_context
from types import SimpleNamespace

import cv2
import numpy as np
import openslide
import torch

from experiments.lowsep import paths as LP
from experiments.lowsep.cascade_pipe import read_screen, read_unit
from experiments.lowsep.cascade_prep import load_head
from experiments.lowsep.extract import tokens
from experiments.lowsep.train import names_of
from experiments.pfa_scannet.data import L5, slide_path, tissue_mask
from experiments.rlscreen import costs as C
from experiments.rlscreen import paths as P
from experiments.rlscreen import teacher as T
from experiments.rlscreen.env import SCREEN, ZOOM, observation
from experiments.rlscreen.paths import R_MAX
from experiments.rlscreen.policy import ScreenPolicy
from experiments.sep.backbones import Encoder
from experiments.tokhead.data import CROP, TOK_PX
from experiments.tokhead.model import TOK, TokenNet

SEG_THR = 0.4   # the token head's validation-chosen threshold


class Models:
    def __init__(self, a):
        self.dev = a.device
        self.enc = Encoder(a.backbone, a.device)
        heads = json.load(open(f'{P.units_dir(a.backbone)}/heads.json'))
        g = 3 if a.design == 'full' else a.g_level
        self.g_level = g
        self.g_head = load_head(heads[f'L{g}_full'][1], a.device)
        self.l3_unit_head = load_head(heads['L3_unit'][1], a.device)
        pca = lambda lv: {k: v.to(a.device) for k, v in torch.load(f'{LP.tok_dir(lv, a.backbone)}/pca.pt', weights_only=False).items() if k in ('mu', 'W')}
        self.pca_g, self.pca_3 = pca(g), pca(3)
        self.seg = TokenNet().to(a.device).eval()
        self.seg.head.load_state_dict(torch.load(a.seg_ckpt, map_location=a.device)['model'])
        self.policy = None
        if a.actor == 'policy':
            args = json.load(open(f'{P.run_dir(a.tag)}/args.json'))
            self.state = args['state']
            self.policy = ScreenPolicy(self.state).to(a.device).eval()
            self.policy.load_state_dict(torch.load(f'{P.run_dir(a.tag)}/{a.ckpt}.pt', map_location=a.device))

    def embed(self, imgs, pca):
        x = torch.from_numpy(np.stack(imgs)).to(self.dev, non_blocking=True)
        t = tokens(self.enc, x).permute(0, 2, 3, 1)
        return ((t - pca['mu']) @ pca['W']).permute(0, 3, 1, 2)          # (B, 256, 16, 16)


def global_pass(m, path, level, t5, hu, wu, pool):
    """Every unit's pooled tokens and head probabilities from a whole-slide pass at `level`."""
    fp = LP.CROP_PX * 2 ** level
    k5, mm = fp // L5, LP.units_per_crop(level)
    h_px, w_px = hu * CROP, wu * CROP
    hc, wc = -(-h_px // fp), -(-w_px // fp)
    tt = np.pad(t5, ((0, max(0, hc * k5 - t5.shape[0])), (0, max(0, wc * k5 - t5.shape[1]))))[:hc * k5, :wc * k5]
    rc = np.argwhere(tt.reshape(hc, k5, wc, k5).any(axis=(1, 3)))
    canvas = torch.zeros((LP.D_PCA, hc * LP.TOK, wc * LP.TOK), device=m.dev)
    for i in range(0, len(rc), 128):
        items = pool.map(read_screen, [(path, level, int(r), int(c)) for r, c in rc[i:i + 128]])
        tk = m.embed([it[2] for it in items], m.pca_g)
        for b, (r, c, _) in enumerate(items):
            canvas[:, r * LP.TOK:(r + 1) * LP.TOK, c * LP.TOK:(c + 1) * LP.TOK] = tk[b]
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        p = torch.softmax(m.g_head(canvas[None]).float(), 1)[0][:, :hu, :wu].permute(1, 2, 0)
    k = LP.tokens_per_unit(level)
    c = canvas.reshape(LP.D_PCA, hc * mm, k, wc * mm, k)
    mean = c.mean((2, 4))[:, :hu, :wu].permute(1, 2, 0)
    mx = c.amax((2, 4))[:, :hu, :wu].permute(1, 2, 0)
    return torch.cat([mean, mx, p], -1), p.cpu().numpy()


def screen_units(m, items):
    """Level-3 view of screened units: pooled tokens and the per-unit head's probabilities."""
    tk = m.embed([it[2] for it in items], m.pca_3)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        p = torch.softmax(m.l3_unit_head(tk).float(), 1)[:, :, 0, 0]
    return torch.cat([tk.mean((2, 3)), tk.amax((2, 3)), p], -1)


def seg_units(m, items, probs, outcome):
    x = torch.from_numpy(np.stack([it[2] for it in items])).to(m.dev, non_blocking=True)
    with torch.no_grad():
        p = torch.sigmoid(m.seg(x))                                    # (B, 32, 32)
    g = 2 * TOK
    pos = (p >= SEG_THR).float()
    o = torch.stack([pos.mean((1, 2)), pos[:, 0].mean(1), pos[:, -1].mean(1), pos[:, :, 0].mean(1), pos[:, :, -1].mean(1)], -1)
    pn = p.cpu().numpy()
    for k, (gy, gx, _) in enumerate(items):
        probs[gy * g:(gy + 1) * g, gx * g:(gx + 1) * g] = pn[k]
        outcome[gy, gx] = o[k]


def run_slide(name, a, m, pools, out_path):
    path = slide_path(name)
    fd = os.open(path, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
    torch.cuda.synchronize()
    t0 = time.time()
    s = openslide.OpenSlide(path)
    w, h = s.dimensions
    t5 = tissue_mask(s)
    ht, wt = -(-h // TOK_PX), -(-w // TOK_PX)
    tissue_tok = cv2.resize(t5.astype(np.float32), (wt, ht), interpolation=cv2.INTER_AREA if TOK_PX > L5 else cv2.INTER_NEAREST) > 0
    g = 2 * TOK
    hu, wu = -(-h // CROP), -(-w // CROP)
    tissue = np.zeros((hu, wu), bool)
    for gy in range(hu):
        for gx in range(wu):
            tissue[gy, gx] = tissue_tok[gy * g:(gy + 1) * g, gx * g:(gx + 1) * g].any()
    t1 = time.time()
    gview, p_g = global_pass(m, path, m.g_level, t5, hu, wu, pools['screen'])
    dev = m.dev
    tab = SimpleNamespace(design=a.design, g_level=a.g_level, backbone=a.backbone, device=dev, hu=hu, wu=wu, n_units=int(tissue.sum()),
                          tissue=torch.from_numpy(tissue).to(dev), g=gview, l3=gview.clone() if a.design == 'full' else torch.zeros_like(gview),
                          p_g=p_g, p3=p_g.copy() if a.design == 'full' else np.zeros_like(p_g),
                          outcome=torch.zeros((hu, wu, 5), device=dev))
    st = SimpleNamespace(t=tab, screened=tab.tissue.clone() if a.design == 'full' else torch.zeros_like(tab.tissue),
                         zoomed=torch.zeros_like(tab.tissue), round=0, spent=0.0, lam=a.lam)
    t2 = time.time()
    probs = np.zeros((hu * g, wu * g), np.float32)
    n_screen = n_zoom = 0
    while st.round < R_MAX:
        free = tab.tissue & ~st.zoomed
        if a.actor == 'teacher':
            stop, act = T.act(st, tau=a.tau, follow=a.follow, outcome=tab.outcome)
        else:
            snap = dict(screened=st.screened, zoomed=st.zoomed, round=st.round, spent=st.spent, lam=st.lam)
            planes, scal = observation(tab, snap, m.state, outcome=tab.outcome)
            allowed = torch.stack([torch.ones_like(free), free & ~st.screened if a.design == 'coarse' else torch.zeros_like(free), free], -1)
            with torch.no_grad():
                logits, stop_logit, _ = m.policy(planes[None], scal[None], tab.tissue[None])
            stop = bool(stop_logit[0] > 0)
            act = logits[0].masked_fill(~allowed, -1e9).argmax(-1)
        sc = (act == SCREEN) & free & ~st.screened
        zm = (act == ZOOM) & free
        if stop or not (sc.any() or zm.any()):
            break
        sc_list = [tuple(map(int, u)) for u in torch.nonzero(sc).tolist()]
        zm_list = [tuple(map(int, u)) for u in torch.nonzero(zm).tolist()]
        f_sc = pools['screen'].map_async(read_screen, [(path, 3, r, c) for r, c in sc_list]) if sc_list else None
        f_zm = pools['seg'].map_async(read_unit, [(path, r, c) for r, c in zm_list]) if zm_list else None
        if f_sc is not None:
            items = f_sc.get()
            for i in range(0, len(items), 128):
                v = screen_units(m, items[i:i + 128])
                for k, (r, c, _) in enumerate(items[i:i + 128]):
                    tab.l3[r, c] = v[k]
                    tab.p3[r, c] = v[k, -4:].cpu().numpy()
        if f_zm is not None:
            items = f_zm.get()
            for i in range(0, len(items), 64):
                seg_units(m, items[i:i + 64], probs, tab.outcome)
        st.screened |= sc
        st.zoomed |= zm
        n_screen += len(sc_list); n_zoom += len(zm_list)
        st.round += 1
        st.spent += C.cost_fraction(a.backbone, len(sc_list), len(zm_list), tab.n_units)
    torch.cuda.synchronize()
    wall = time.time() - t0
    probs = probs[:ht, :wt]
    probs[~tissue_tok] = 0
    np.savez_compressed(out_path, probs=probs.astype(np.float16), alpha=16, stride=TOK_PX, level=int(np.log2(TOK_PX)), crops=n_zoom, wall=wall)
    rec = dict(slide=name, units=tab.n_units, screened=n_screen, zoomed=n_zoom, rounds=st.round, wall=wall,
               tissue=t1 - t0, global_pass=t2 - t1, rounds_time=wall - (t2 - t0))
    json.dump(rec, open(out_path[:-4] + '_time.json', 'w'))
    print(f"{name}: screened {n_screen} zoomed {n_zoom}/{tab.n_units} in {st.round} rounds, {wall:.1f}s "
          f"(tissue {rec['tissue']:.1f}, global {rec['global_pass']:.1f}, rounds {rec['rounds_time']:.1f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--design', choices=('coarse', 'full'), required=True)
    ap.add_argument('--g_level', type=int, default=6)
    ap.add_argument('--backbone', default='kaiko_vitl14')
    ap.add_argument('--actor', choices=('policy', 'teacher'), required=True)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--ckpt', default='final')
    ap.add_argument('--lam', type=float, default=1.0)
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--follow', type=int, default=1)
    ap.add_argument('--seg_ckpt', default='/data2/venkatavks/pfa_ckpt/named/tokhead_kaiko_vitl14_stageA_input_l2.pt')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--screen_workers', type=int, default=8)
    ap.add_argument('--seg_workers', type=int, default=8)
    a = ap.parse_args()
    if a.actor == 'policy':  # the run's own design / views, never the command line's
        args = json.load(open(f'{P.run_dir(a.tag)}/args.json'))
        a.design, a.g_level, a.backbone = args['design'], args['g_level'], args['backbone']
    os.makedirs(a.out_dir, exist_ok=True)
    ctx = get_context('fork')  # pools before CUDA; readers never touch the GPU
    pools = {'screen': ctx.Pool(a.screen_workers), 'seg': ctx.Pool(a.seg_workers)}
    m = Models(a)
    with torch.no_grad():
        m.seg(torch.zeros(2, 4, 3, 224, 224, dtype=torch.uint8, device=a.device))
        tokens(m.enc, torch.zeros(2, 3, 224, 224, dtype=torch.uint8, device=a.device))
    torch.cuda.synchronize()
    names = names_of('test', 6)
    for n in names:
        out = os.path.join(a.out_dir, n + '.npz')
        if not os.path.exists(out):
            run_slide(n, a, m, pools, out)
    json.dump(vars(a), open(os.path.join(a.out_dir, 'run.json'), 'w'), indent=1)
    open(os.path.join(a.out_dir, 'DONE'), 'w').write(f'{len(names)} slides\n')


if __name__ == '__main__':
    main()
