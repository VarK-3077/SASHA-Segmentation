# Pipelined screen + segment cascade. Same models, head, threshold and output layout as
# cascade.py; only the scheduling changes. Two reader pools live for the whole run (no worker
# start-up per slide): one reads screen crops in row order, the other reads level-2 children of
# units that have been decided. The screen head is rerun on the screened part of the map after
# each GPU batch; a unit is final once it is more than MARGIN unit-rows behind the screening front
# (the head's receptive radius is under 4 units, so its decision equals the whole-map one), and a
# final zoom unit is queued for reading at once, so segmenting overlaps with screening.
# Usage: TOK_LEVEL=2 python -m experiments.lowsep.cascade_pipe --level 3 --budget 0.2 --seg_ckpt HEAD.pt --out_dir DIR

import argparse
import json
import os
import time
from multiprocessing import get_context

import cv2
import numpy as np
import openslide
import torch
from PIL import Image

from experiments.lowsep import paths as P
from experiments.lowsep.cascade_prep import load_head
from experiments.lowsep.extract import tokens
from experiments.lowsep.train import names_of
from experiments.pfa_scannet.data import L5, read_rgb, slide_path, tissue_mask
from experiments.sep.backbones import Encoder
from experiments.tokhead.data import CHILD_PX, CROP, LEVEL, TOK_PX, children
from experiments.tokhead.model import TOK, TokenNet

assert CROP == P.UNIT, 'run with TOK_LEVEL=2 so the segmenter crop is the 2048 px unit'
MARGIN = 5          # unit rows behind the screening front before a unit's decision is final
SCREEN_BATCH, SEG_BATCH = 128, 64

_handle = {}


def _slide(path):
    if _handle.get('path') != path:
        _handle.clear()
        _handle.update(path=path, s=openslide.OpenSlide(path))
    return _handle['s']


def read_screen(task):
    path, level, r, c = task
    fp = P.CROP_PX * 2 ** level
    a = np.asarray(_slide(path).read_region((int(c * fp), int(r * fp)), level, (P.CROP_PX, P.CROP_PX)), np.float32)
    alpha = a[..., 3:4] / 255.0
    rgb = (a[..., :3] * alpha + 255.0 * (1 - alpha)).astype(np.uint8)
    return r, c, np.asarray(Image.fromarray(rgb).resize((224, 224), Image.BILINEAR)).transpose(2, 0, 1).copy()


def read_unit(task):
    path, gy, gx = task
    return gy, gx, children(read_rgb(_slide(path), gx * CROP, gy * CROP, 2 * CHILD_PX, level=LEVEL)).numpy()


class Segmenter:
    """Collects read children and runs the token head in batches as they arrive."""

    def __init__(self, net, probs, g, device):
        self.net, self.probs, self.g, self.device = net, probs, g, device
        self.pending, self.ready, self.gpu = [], [], 0.0

    def submit(self, pool, path, units):
        self.pending += [pool.apply_async(read_unit, ((path, gy, gx),)) for gy, gx in units]

    def poll(self, block=False):
        keep = []
        for f in self.pending:
            if block or f.ready():
                self.ready.append(f.get())
            else:
                keep.append(f)
        self.pending = keep
        while len(self.ready) >= SEG_BATCH or (block and self.ready):
            self.run(self.ready[:SEG_BATCH]); self.ready = self.ready[SEG_BATCH:]

    def run(self, items):
        t = time.time()
        x = torch.from_numpy(np.stack([it[2] for it in items])).to(self.device, non_blocking=True)
        with torch.no_grad():
            p = torch.sigmoid(self.net(x)).cpu().numpy()
        g = self.g
        for k, (gy, gx, _) in enumerate(items):
            self.probs[gy * g:(gy + 1) * g, gx * g:(gx + 1) * g] = p[k]
        self.gpu += time.time() - t


def run_slide(name, a, enc, pca, head, net, thr, pools, out_path):
    path = slide_path(name)
    fd = os.open(path, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
    torch.cuda.synchronize()
    t0 = time.time()
    s = openslide.OpenSlide(path)
    w, h = s.dimensions
    t5 = tissue_mask(s)
    ht, wt = -(-h // TOK_PX), -(-w // TOK_PX)
    tissue = cv2.resize(t5.astype(np.float32), (wt, ht), interpolation=cv2.INTER_AREA if TOK_PX > L5 else cv2.INTER_NEAREST) > 0
    g = 2 * TOK
    hu, wu = -(-h // CROP), -(-w // CROP)
    unit_tissue = np.zeros((hu, wu), bool)
    for gy in range(hu):
        for gx in range(wu):
            unit_tissue[gy, gx] = tissue[gy * g:(gy + 1) * g, gx * g:(gx + 1) * g].any()
    m = P.units_per_crop(a.level)
    fp = P.CROP_PX * 2 ** a.level
    k5 = fp // L5
    hc, wc = -(-h // fp), -(-w // fp)
    tt = np.pad(t5, ((0, max(0, hc * k5 - t5.shape[0])), (0, max(0, wc * k5 - t5.shape[1]))))[:hc * k5, :wc * k5]
    rc = np.argwhere(tt.reshape(hc, k5, wc, k5).any(axis=(1, 3))).astype(np.int32)  # row-major order
    t1 = time.time()
    canvas = torch.zeros((P.D_PCA, hc * P.TOK, wc * P.TOK), device=a.device, dtype=torch.bfloat16)
    probs = np.zeros((hu * g, wu * g), np.float32)
    seg = Segmenter(net, probs, g, a.device)
    decided = np.zeros((hu, wu), bool)
    zoom_all = []
    screen_gpu = 0.0

    def decide(crop_rows_done, final):
        """Head on the fully screened crop rows; units more than MARGIN rows above their end are final."""
        nonlocal screen_gpu
        tg = time.time()
        rows_tok = hc * P.TOK if final else crop_rows_done * P.TOK
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            pr = torch.softmax(head(canvas[None, :, :rows_tok].float()).float(), 1)[0]
        p_tum = (1 - pr[0]).cpu().numpy()
        screen_gpu += time.time() - tg
        lim = hu if final else max(0, min(hu, crop_rows_done * m - MARGIN))
        new = [(gy, gx) for gy in range(lim) for gx in range(wu)
               if not decided[gy, gx] and unit_tissue[gy, gx] and gy < p_tum.shape[0] and gx < p_tum.shape[1] and p_tum[gy, gx] >= thr]
        decided[:lim] = True
        zoom_all.extend(new)
        seg.submit(pools['seg'], path, new)

    it = pools['screen'].imap(read_screen, [(path, a.level, int(r), int(c)) for r, c in rc], chunksize=4)
    batch, front = [], 0
    for i in range(len(rc)):
        batch.append(next(it))
        if len(batch) == SCREEN_BATCH or i == len(rc) - 1:
            tg = time.time()
            x = torch.from_numpy(np.stack([b[2] for b in batch])).to(a.device, non_blocking=True)
            tk = tokens(enc, x).permute(0, 2, 3, 1)
            tk = ((tk - pca['mu']) @ pca['W']).permute(0, 3, 1, 2)
            for b, (r, c, _) in enumerate(batch):
                canvas[:, r * P.TOK:(r + 1) * P.TOK, c * P.TOK:(c + 1) * P.TOK] = tk[b]
            screen_gpu += time.time() - tg
            rows_done = int(batch[-1][0])  # crop rows strictly above the last crop read are complete
            batch = []
            if rows_done * m - MARGIN > 0:
                decide(rows_done, final=False)
            seg.poll()
    decide(hc, final=True)
    t2 = time.time()
    seg.poll(block=True)
    torch.cuda.synchronize()
    wall = time.time() - t0
    probs = probs[:ht, :wt]
    probs[~tissue] = 0
    np.savez_compressed(out_path, probs=probs.astype(np.float16), alpha=16, stride=TOK_PX, level=int(np.log2(TOK_PX)),
                        crops=len(zoom_all), wall=wall)
    rec = dict(slide=name, units=int(unit_tissue.sum()), zoomed=len(zoom_all), wall=wall, tissue=t1 - t0,
               screen_phase=t2 - t1, tail=wall - (t2 - t0), screen_gpu=screen_gpu, seg_gpu=seg.gpu,
               zoom_units=[list(map(int, u)) for u in sorted(zoom_all)])
    json.dump(rec, open(out_path[:-4] + '_time.json', 'w'))
    print(f"{name}: {len(zoom_all)}/{rec['units']} units zoomed, {wall:.1f}s (tissue {rec['tissue']:.1f}, "
          f"screen+overlapped segment {rec['screen_phase']:.1f}, tail {rec['tail']:.1f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--level', type=int, required=True)
    ap.add_argument('--budget', required=True)
    ap.add_argument('--seg_ckpt', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--screen_workers', type=int, default=8)
    ap.add_argument('--seg_workers', type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    ctx = get_context('fork')  # pools before any CUDA initialisation; readers never touch the GPU
    pools = {'screen': ctx.Pool(a.screen_workers), 'seg': ctx.Pool(a.seg_workers)}
    cfg = json.load(open(f'{P.ROOT}/cascade_cfg.json'))[f'L{a.level}']
    thr = cfg['thresholds'][a.budget]
    enc = Encoder('kaiko_vitl14', a.device)
    pca = torch.load(f'{P.tok_dir(a.level)}/pca.pt', weights_only=False)
    pca = {'mu': pca['mu'].to(a.device), 'W': pca['W'].to(a.device)}
    head = load_head(cfg['tag'], a.device)
    net = TokenNet().cuda().eval()
    net.head.load_state_dict(torch.load(a.seg_ckpt, map_location='cuda')['model'])
    names = names_of('test', a.level)
    with torch.no_grad():
        net(torch.zeros(2, 4, 3, 224, 224, dtype=torch.uint8, device='cuda'))
        tokens(enc, torch.zeros(2, 3, 224, 224, dtype=torch.uint8, device=a.device))
    torch.cuda.synchronize()
    for n in names:
        out = os.path.join(a.out_dir, n + '.npz')
        if not os.path.exists(out):
            run_slide(n, a, enc, pca, head, net, thr, pools, out)
    json.dump({'level': a.level, 'budget': a.budget, 'threshold': thr, 'head': cfg['tag'], 'margin_units': MARGIN,
               'screen_workers': a.screen_workers, 'seg_workers': a.seg_workers}, open(os.path.join(a.out_dir, 'cascade.json'), 'w'), indent=1)
    open(os.path.join(a.out_dir, 'DONE'), 'w').write(f'{len(names)} slides\n')


if __name__ == '__main__':
    main()
