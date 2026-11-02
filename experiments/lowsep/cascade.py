# Screen + segment cascade, timed end to end on the test slides. Per slide, with the slide's page
# cache dropped first: open the slide, Otsu tissue mask at level 5 (shared by both stages), the
# screen (256 px crops at level L, frozen Kaiko ViT-L/14, PCA, the val-chosen head) gives
# p(tumour) per 2048 px unit, units at or above the val-chosen threshold are segmented by the
# real level-2 token head (four level-2 children per unit, the same code path as
# tokhead/infer.py), every other unit's map stays 0. The map is saved in tokhead/infer.py's
# layout so pfa_scannet.score_maps scores it unchanged; a per-slide time json splits the wall
# time into tissue, screen and segment.
# Usage: TOK_LEVEL=2 python -m experiments.lowsep.cascade --level 3 --budget 0.1 --seg_ckpt HEAD.pt --out_dir DIR

import argparse
import json
import os
import time

import cv2
import numpy as np
import openslide
import torch
from torch.utils.data import DataLoader

from experiments.lowsep import paths as P
from experiments.lowsep.cascade_prep import load_head
from experiments.lowsep.extract import Crops as ScreenCrops, tokens
from experiments.lowsep.train import names_of
from experiments.pfa_scannet.data import L5, slide_path, tissue_mask
from experiments.sep.backbones import Encoder
from experiments.tokhead.data import CROP, TOK_PX
from experiments.tokhead.infer import Crops as SegCrops
from experiments.tokhead.model import TOK, TokenNet

assert CROP == P.UNIT, 'run with TOK_LEVEL=2 so the segmenter crop is the 2048 px unit'


def screen(slide_name, level, t5, h, w, enc, pca, head, workers, device):
    """p(tumour) per unit on the (hu, wu) unit grid, from the level-L screen."""
    fp = P.CROP_PX * 2 ** level
    k5 = fp // L5
    hc, wc = -(-h // fp), -(-w // fp)
    t = np.pad(t5, ((0, max(0, hc * k5 - t5.shape[0])), (0, max(0, wc * k5 - t5.shape[1]))))[:hc * k5, :wc * k5]
    rc = np.argwhere(t.reshape(hc, k5, wc, k5).any(axis=(1, 3))).astype(np.int32)
    canvas = torch.zeros((P.D_PCA, hc * P.TOK, wc * P.TOK), device=device, dtype=torch.bfloat16)
    k = 0
    for x in DataLoader(ScreenCrops(slide_name, level, rc), batch_size=128, num_workers=workers, pin_memory=True):
        tk = tokens(enc, x.to(device, non_blocking=True)).permute(0, 2, 3, 1)
        tk = ((tk - pca['mu']) @ pca['W']).permute(0, 3, 1, 2)
        for b, (r, c) in enumerate(rc[k:k + len(x)]):
            canvas[:, r * P.TOK:(r + 1) * P.TOK, c * P.TOK:(c + 1) * P.TOK] = tk[b]
        k += len(x)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        prob = torch.softmax(head(canvas[None].float()).float(), 1)[0]
    return (1 - prob[0]).cpu().numpy()


def run_slide(name, a, enc, pca, head, net, thr, out_path):
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
    units = [(gy, gx) for gy in range(-(-h // CROP)) for gx in range(-(-w // CROP)) if tissue[gy * g:(gy + 1) * g, gx * g:(gx + 1) * g].any()]
    t1 = time.time()
    p_tum = screen(name, a.level, t5, h, w, enc, pca, head, a.workers, a.device)
    torch.cuda.synchronize()
    t2 = time.time()
    zoom = [(gy, gx) for gy, gx in units if gy < p_tum.shape[0] and gx < p_tum.shape[1] and p_tum[gy, gx] >= thr]
    probs = np.zeros((-(-h // CROP) * g, -(-w // CROP) * g), np.float32)
    if zoom:
        with torch.no_grad():
            for x, gys, gxs in DataLoader(SegCrops(path, zoom), batch_size=64, num_workers=a.workers, pin_memory=True):
                p = torch.sigmoid(net(x.cuda(non_blocking=True))).cpu().numpy()
                for kk, (gy, gx) in enumerate(zip(gys.tolist(), gxs.tolist())):
                    probs[gy * g:(gy + 1) * g, gx * g:(gx + 1) * g] = p[kk]
    torch.cuda.synchronize()
    wall = time.time() - t0
    probs = probs[:ht, :wt]
    probs[~tissue] = 0
    np.savez_compressed(out_path, probs=probs.astype(np.float16), alpha=16, stride=TOK_PX, level=int(np.log2(TOK_PX)),
                        crops=len(zoom), wall=wall)
    rec = dict(slide=name, units=len(units), zoomed=len(zoom), wall=wall, tissue=t1 - t0, screen=t2 - t1, segment=wall - (t2 - t0))
    json.dump(rec, open(out_path[:-4] + '_time.json', 'w'), indent=1)
    print(f"{name}: {len(zoom)}/{len(units)} units zoomed, {wall:.1f}s (tissue {rec['tissue']:.1f}, screen {rec['screen']:.1f}, segment {rec['segment']:.1f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--level', type=int, required=True)
    ap.add_argument('--budget', required=True, help='val zoom fraction key in cascade_cfg.json, e.g. 0.1')
    ap.add_argument('--seg_ckpt', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    cfg = json.load(open(f'{P.ROOT}/cascade_cfg.json'))[f'L{a.level}']
    thr = cfg['thresholds'][a.budget]
    enc = Encoder('kaiko_vitl14', a.device)
    pca = torch.load(f'{P.tok_dir(a.level)}/pca.pt', weights_only=False)
    pca = {'mu': pca['mu'].to(a.device), 'W': pca['W'].to(a.device)}
    head = load_head(cfg['tag'], a.device)
    net = TokenNet().cuda().eval()
    net.head.load_state_dict(torch.load(a.seg_ckpt, map_location='cuda')['model'])
    names = names_of('test', a.level)
    with torch.no_grad():  # warm-up of both networks so setup is not billed to the first slide
        net(torch.zeros(2, 4, 3, 224, 224, dtype=torch.uint8, device='cuda'))
        tokens(enc, torch.zeros(2, 3, 224, 224, dtype=torch.uint8, device=a.device))
    torch.cuda.synchronize()
    for n in names:
        out = os.path.join(a.out_dir, n + '.npz')
        if not os.path.exists(out):
            run_slide(n, a, enc, pca, head, net, thr, out)
    json.dump({'level': a.level, 'budget': a.budget, 'threshold': thr, 'head': cfg['tag']}, open(os.path.join(a.out_dir, 'cascade.json'), 'w'), indent=1)
    open(os.path.join(a.out_dir, 'DONE'), 'w').write(f'{len(names)} slides\n')


if __name__ == '__main__':
    main()
