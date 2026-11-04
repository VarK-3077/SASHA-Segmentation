# Builds one table per slide that every RL screening experiment reads, so episodes never touch
# slides or backbones. Per 2048 px unit: tissue flag, tumour fraction and bucket (lowsep labels);
# how much of each border strip is tumour (the signal a segmented unit gives about its
# neighbours); the lesions it touches and their pixel counts, lesions being the official
# evaluation components (75 um dilation, holes filled) with ITCs (< 275 um) flagged; and the views: mean and max pooled screen tokens plus the supervised
# head's four bucket probabilities at the global levels 5 / 6 and at level 3. Level 3 has two
# head outputs: the best per-unit (no-context) head, valid when only some units are screened, and
# the best head overall on the full map. Heads are picked per level by val AUC, nothing else.
# Usage: LOWSEP_BACKBONE=kaiko_vitl14 python -m experiments.rlscreen.prep --device cuda:0

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from experiments.lowsep import paths as LP
from experiments.lowsep.cascade_prep import head_tags, load_head, val_scores
from experiments.lowsep.maps import SlideMaps
from experiments.lowsep.train import names_of, predict
from experiments.rlscreen import paths as P
from utils.segmentation_metrics import evaluation_mask, itc_labels, rasterize_mask, read_xml_polygons

KM = LP.UNIT // LP.MASK_DS   # mask cells per unit side (128)
STRIP = 64 // LP.MASK_DS     # border strip = one level-2 token row (64 level-0 px), as run_real measures it


def pick_head(level, ctx, device):
    best = None
    for tag in head_tags(level, ctx):
        p, y = val_scores(load_head(tag, device), level, device)
        auc = float(roc_auc_score(y, p))
        if best is None or auc > best[0]:
            best = (auc, tag)
    return best


def pooled(maps, hu, wu):
    """(hu, wu, 256) mean and max of each unit's token block."""
    tk, _ = maps.full()
    k = LP.tokens_per_unit(maps.level)
    t = torch.from_numpy(tk).float()
    c, H, W = t.shape
    t = t.reshape(c, H // k, k, W // k, k)
    mean = t.mean((2, 4))[:, :hu, :wu].permute(1, 2, 0).numpy().astype(np.float16)
    mx = t.amax((2, 4))[:, :hu, :wu].permute(1, 2, 0).numpy().astype(np.float16)
    return mean, mx


def gt_tables(name, hu, wu, mpp, dims):
    xml = LP.xml_path(name)
    tum, exc = read_xml_polygons(xml) if os.path.exists(xml) else ([], [])
    edges = np.zeros((hu, wu, 4), np.float16)
    empty = dict(edges=edges, lu_unit=np.zeros(0, np.int32), lu_lesion=np.zeros(0, np.int32), lu_px=np.zeros(0, np.int32),
                 lesion_px=np.zeros(0, np.int64), lesion_itc=np.zeros(0, bool))
    if not tum:
        return empty
    m = rasterize_mask(tum, exc, dims, LP.MASK_DS)
    m = np.pad(m, ((0, hu * KM - m.shape[0]), (0, wu * KM - m.shape[1])))[:hu * KM, :wu * KM]
    b = m.reshape(hu, KM, wu, KM).astype(np.float32)
    edges = np.stack([b[:, :STRIP].mean((1, 3)), b[:, -STRIP:].mean((1, 3)),
                      b[:, :, :, :STRIP].mean((1, 3)), b[:, :, :, -STRIP:].mean((1, 3))], -1).astype(np.float16)  # N S W E
    ev = evaluation_mask(m, mpp, LP.MASK_DS)   # official lesions: 75 um dilation, holes filled
    n = int(ev.max())
    if n == 0:
        return dict(empty, edges=edges)
    itc = np.zeros(n, bool)
    itc[np.array(itc_labels(ev, mpp, LP.MASK_DS), dtype=int) - 1] = True
    lab = ev * (m > 0)                          # annotated tumour pixels, labelled by their lesion
    rr, cc = np.nonzero(lab)
    unit = (rr // KM) * wu + (cc // KM)
    key = unit.astype(np.int64) * (n + 1) + lab[rr, cc]
    uk, cnt = np.unique(key, return_counts=True)
    return dict(edges=edges, lu_unit=(uk // (n + 1)).astype(np.int32), lu_lesion=(uk % (n + 1) - 1).astype(np.int32),
                lu_px=cnt.astype(np.int32), lesion_px=np.bincount(lab[rr, cc] - 1, minlength=n).astype(np.int64), lesion_itc=itc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    bb = LP.BACKBONE
    out = P.units_dir(bb)
    os.makedirs(out, exist_ok=True)
    choice_path = f'{out}/heads.json'
    if os.path.exists(choice_path):
        choice = json.load(open(choice_path))
    else:
        choice = {}
        for lv in P.G_LEVELS + (3,):
            choice[f'L{lv}_full'] = pick_head(lv, None, a.device)
        choice['L3_unit'] = pick_head(3, False, a.device)
        json.dump(choice, open(choice_path, 'w'), indent=1)
    print('heads:', choice, flush=True)
    heads = {k: load_head(v[1], a.device) for k, v in choice.items()}
    split = LP.split_of()
    names = sorted(n for n in split if os.path.exists(LP.label_npz(n)) and os.path.exists(f'{LP.tok_dir(3)}/{n}.npy'))
    for i, name in enumerate(names):
        f = f'{out}/{name}.npz'
        if os.path.exists(f):
            continue
        z = np.load(LP.label_npz(name))
        tissue, frac, bucket = z['tissue'], z['frac'].astype(np.float32), z['bucket']
        hu, wu = tissue.shape
        rec = dict(tissue=tissue, frac=frac, bucket=bucket, trust=bool(z['trust']), split=split[name],
                   **gt_tables(name, hu, wu, float(z['mpp']), tuple(int(v) for v in z['dims'])))
        for lv in P.G_LEVELS + (3,):
            maps = SlideMaps(name, lv)
            rec[f'f{lv}_mean'], rec[f'f{lv}_max'] = pooled(maps, hu, wu)
            rec[f'p{lv}'] = predict(heads[f'L{lv}_full'], maps, a.device)[:, :hu, :wu].transpose(1, 2, 0).astype(np.float16)
            if lv == 3:
                rec['p3_unit'] = predict(heads['L3_unit'], maps, a.device)[:, :hu, :wu].transpose(1, 2, 0).astype(np.float16)
        np.savez(f, **rec)
        if i % 25 == 0:
            print(f'{i}/{len(names)} {name}: {hu}x{wu} units, {int(tissue.sum())} tissue', flush=True)
    print('prep done', flush=True)


if __name__ == '__main__':
    main()
