# Screens at levels 3-6 scored as if the segmenter were perfect. Per level: the trained head
# (seed 0 of each config) picked on val, either by AUC for tumour-vs-normal units (cascade_prep's
# rule) or by the val ideal-segmenter Dice at the budget, and the p(tumour) threshold that zooms
# 20 % of val tissue units. On every test slide
# the zoomed units get the ground-truth mask, everything else stays empty; the maps are written
# at level 5 (32 px cells) in the token-head layout so clean_sweep / small_probe score them with
# the same per-slide, pooled and per-lesion Dice and found-lesion counts as the real models.
# Usage: python -m experiments.lowsep.screen_ideal --out_root DIR [--budget 0.2]

import argparse
import glob
import json
import os

import numpy as np
import openslide

from experiments.lowsep import paths as P
from experiments.lowsep.cascade_prep import load_head, val_scores
from experiments.lowsep.maps import SlideMaps
from experiments.lowsep.train import names_of, predict
from experiments.pfa_scannet.data import slide_path, xml_path
from utils.segmentation_metrics import rasterize_mask, read_xml_polygons

CELL = 32


def val_ideal_dice(head, level, budget, device):
    """Mean ideal-segmenter Dice over the val tumour slides when the top `budget` of val tissue
    units by p(tumour) are zoomed (zoomed units return their ground truth), and that threshold."""
    per = []
    for n in names_of('val', level):
        m = SlideMaps(n, level)
        per.append((1 - predict(head, m, device)[0][m.tissue], m.frac[m.tissue].astype(np.float64)))
    thr = float(np.quantile(np.concatenate([p for p, _ in per]), 1 - budget))
    dice = [2 * f[p >= thr].sum() / (f[p >= thr].sum() + f.sum()) for p, f in per if f.sum() > 0]
    return float(np.mean(dice)), thr


def choose(level, budget, device, rule):
    """Seed-0 head per config, picked on val: by AUC of tumour vs normal units, or by the val
    ideal-segmenter Dice at the budget (the quantity the deck reports)."""
    from sklearn.metrics import roc_auc_score
    best = None
    for d in sorted(glob.glob(f'{P.ROOT}/runs/L{level}_*_s0')):
        tag = os.path.basename(d)
        head = load_head(tag, device)
        if rule == 'auc':
            p, y = val_scores(head, level, device)
            score, thr = float(roc_auc_score(y, p)), float(np.quantile(p, 1 - budget))
        else:
            score, thr = val_ideal_dice(head, level, budget, device)
        print(f'  L{level} {tag}: val {rule} {score:.4f}', flush=True)
        if best is None or score > best[0]:
            best = (score, tag, thr)
    return best


def ideal_map(name, zoom_units):
    s = openslide.OpenSlide(slide_path(name))
    w, h = s.dimensions
    shape = (-(-h // CELL), -(-w // CELL))
    if not os.path.exists(xml_path(name)):
        return np.zeros(shape, np.float32)
    gt = rasterize_mask(*read_xml_polygons(xml_path(name)), (w, h), CELL).astype(np.float32)[:shape[0], :shape[1]]
    k = P.UNIT // CELL
    keep = np.kron(zoom_units, np.ones((k, k), bool))[:shape[0], :shape[1]]
    keep = np.pad(keep, ((0, shape[0] - keep.shape[0]), (0, shape[1] - keep.shape[1])))
    return gt * keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--budget', type=float, default=0.2)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--select', default='auc', choices=('auc', 'dice'))
    a = ap.parse_args()
    summary = {}
    for level in (3, 4, 5, 6):
        auc, tag, thr = choose(level, a.budget, a.device, a.select)
        head = load_head(tag, a.device)
        out = f'{a.out_root}/ideal_L{level}'
        os.makedirs(out, exist_ok=True)
        units = zoomed = 0
        for n in names_of('test', level):
            m = SlideMaps(n, level)
            p_tum = 1 - predict(head, m, a.device)[0]
            zoom = m.tissue & (p_tum >= thr)
            units += int(m.tissue.sum()); zoomed += int(zoom.sum())
            np.savez_compressed(f'{out}/{n}.npz', probs=ideal_map(n, zoom).astype(np.float16), stride=CELL, level=5)
        test = json.load(open(f'{P.run_dir(tag)}/test.json'))
        summary[f'L{level}'] = {'head': tag, 'select': a.select, 'val_score': auc, 'threshold': thr, 'zoom_frac': zoomed / units,
                                'recall_at_99spec': test['recall_at_spec']['0.99'], 'recall_at_95spec': test['recall_at_spec']['0.95']}
        print(f"L{level}: {tag} (val {a.select} {auc:.4f}) thr {thr:.4f}, test zoom {zoomed / units:.3f}, @99 {test['recall_at_spec']['0.99']}", flush=True)
    json.dump(summary, open(f'{a.out_root}/screen_ideal.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
