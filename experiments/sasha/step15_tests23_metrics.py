"""
Part C of tests23: turns the step13/step14 dumps into segmentation numbers. Four heatmaps
are scored - full-high-res HAFED phase-1 (512 px sub-patch grid) and phase-2 (2048 px tile
grid), and the SASHA-0.2 map that splices phase-1 sub-patch scores on visited tiles with
the phase-2 score on the rest, normalised either by one joint softmax or elementwise
sigmoid. Each is scored with a fixed 0.5 threshold and with the confidence rule thr=1-p
(p<0.5 -> empty prediction and no FROC candidates), and with the 5 phase-2 branches
collapsed by mean or by max. GT conventions follow step11.

python experiments/sasha/step15_tests23_metrics.py --partA .../partA_dump.pkl --partB .../partB_dump.pkl \
       --out ~/seg_outputs/tests23/results.json
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import openslide
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from step13_common import LR_TILE_L0, MASK_DIR, SLIDE_DIR, SUB_TILE_L0, XML_DIR, sub_coords  # noqa: E402
from utils.segmentation_metrics import (dice, evaluation_mask, fp_tp_probs, froc, itc_labels, minmax,  # noqa: E402
                                        nms_candidates, predicted_mask, rasterize_mask, read_xml_polygons)

FROC_FPS = (0.25, 0.5, 1, 2, 4, 8)
SUB_NMS_RADIUS = 4   # 4 sub-patch tiles of 512 px == 1 low-res tile of 2048 px
LR_NMS_RADIUS = 1


def get_arguments():
    p = argparse.ArgumentParser('tests23 segmentation metrics')
    p.add_argument('--partA', required=True)
    p.add_argument('--partB', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--slide_dir', default=SLIDE_DIR)
    p.add_argument('--xml_dir', default=XML_DIR)
    p.add_argument('--mask_dir', default=MASK_DIR)
    p.add_argument('--level', type=int, default=5)
    return p.parse_args()


def find_file(root, name):
    for d, _, files in os.walk(root):
        if name in files:
            return os.path.join(d, name)
    return None


def ground_truth(name, args, cache):
    if name in cache:
        return cache[name]
    slide_path = find_file(args.slide_dir, name + '.tif') or find_file(args.mask_dir, name + '_mask.tif')
    if slide_path is None:
        raise FileNotFoundError(f'no slide or mask for {name}')
    s = openslide.OpenSlide(slide_path)
    mpp = float(s.properties.get('openslide.mpp-x', 0.243))
    dims = s.dimensions
    s.close()
    ds = 2 ** args.level
    gt = {'downsample': ds, 'mpp': mpp, 'shape': (int(np.ceil(dims[1] / ds)), int(np.ceil(dims[0] / ds)))}
    xml_path = find_file(args.xml_dir, name + '.xml')
    if xml_path is None:
        gt.update(mask=None, eval_mask=None, itcs=[])
    else:
        mask = rasterize_mask(*read_xml_polygons(xml_path), dims, ds)
        em = evaluation_mask(mask, mpp, ds)
        gt.update(mask=mask, eval_mask=em, itcs=itc_labels(em, mpp, ds))
    cache[name] = gt
    return gt


# ---------------------------------------------------------------- heatmap builders

def hafed_p1(A, B, name, agg):
    e = A[name]
    return e['A1_soft'].astype(np.float64).reshape(-1), sub_coords(e['coords']), SUB_TILE_L0, SUB_NMS_RADIUS, e['p_tumor']


def hafed_p1_globalsm(A, B, name, agg):
    e = A[name]
    v = e['A1_raw'].astype(np.float64).reshape(-1)
    ex = np.exp(v - v.max())
    return ex / ex.sum(), sub_coords(e['coords']), SUB_TILE_L0, SUB_NMS_RADIUS, e['p_tumor']


def hafed_p1_globalsig(A, B, name, agg):
    e = A[name]
    v = e['A1_raw'].astype(np.float64).reshape(-1)
    return 1.0 / (1.0 + np.exp(-v)), sub_coords(e['coords']), SUB_TILE_L0, SUB_NMS_RADIUS, e['p_tumor']


def hafed_p2(A, B, name, agg):
    e = A[name]
    a2 = e['A2_soft'].astype(np.float64)
    scores = a2.mean(0) if agg == 'mean' else a2.max(0)
    return scores, e['coords'], LR_TILE_L0, LR_NMS_RADIUS, e['p_tumor']


def _sasha_raw(A, B, name, agg):
    ea, eb = A[name], B[name]
    p2 = eb['A2f_raw'].astype(np.float64)
    p2 = p2.mean(0) if agg == 'mean' else p2.max(0)
    comb = np.repeat(p2[:, None], 16, axis=1)
    visited = np.unique(np.asarray(eb['visited'], dtype=np.int64))
    comb[visited] = ea['A1_raw'][visited].astype(np.float64)
    return comb.reshape(-1), sub_coords(ea['coords']), eb['p_tumor']


def sasha_softmax(A, B, name, agg):
    v, xy, p = _sasha_raw(A, B, name, agg)
    e = np.exp(v - v.max())
    return e / e.sum(), xy, SUB_TILE_L0, SUB_NMS_RADIUS, p


def sasha_sigmoid(A, B, name, agg):
    v, xy, p = _sasha_raw(A, B, name, agg)
    return 1.0 / (1.0 + np.exp(-v)), xy, SUB_TILE_L0, SUB_NMS_RADIUS, p


def hafed_p2_sigmoid(A, B, name, agg):
    e = A[name]
    a2 = e['A2_raw'].astype(np.float64)
    v = a2.mean(0) if agg == 'mean' else a2.max(0)
    return 1.0 / (1.0 + np.exp(-v)), e['coords'], LR_TILE_L0, LR_NMS_RADIUS, e['p_tumor']


ROWS = {
    'hafed-full-res-p1': hafed_p1,
    'hafed-full-res-p1-globalsm': hafed_p1_globalsm,
    'hafed-full-res-p1-globalsig': hafed_p1_globalsig,
    'hafed-full-res-p2': hafed_p2,
    'hafed-full-res-p2-sigmoid': hafed_p2_sigmoid,
    'sasha-0.2-softmax': sasha_softmax,
    'sasha-0.2-sigmoid': sasha_sigmoid,
}


# ---------------------------------------------------------------- scoring

def score_row(builder, A, B, names, args, gt_cache, agg, rules):
    """Returns {rule: {'froc':..., 'sens_at_fp':..., 'dice':..., ...}} for one aggregation."""
    froc_data = {r: [] for r in rules}
    dices = {r: [] for r in rules}
    skipped = {r: 0 for r in rules}
    empty = {r: 0 for r in rules}
    n_tumor = 0
    for name in tqdm(names, desc=f'{agg}', mininterval=30):
        gt = ground_truth(name, args, gt_cache)
        is_tumor = gt['mask'] is not None
        n_tumor += is_tumor
        raw, xy, tile, radius, p = builder(A, B, name, agg)
        scores = minmax(raw)
        cands = nms_candidates(scores, xy, tile, radius)
        for rule in rules:
            thr = 0.5 if rule == '0.5' else (None if p < 0.5 else 1.0 - p)
            use = [] if (rule == '1-p' and thr is None) else cands
            froc_data[rule].append(fp_tp_probs(use, is_tumor, gt['eval_mask'], gt['itcs'], gt['downsample']))
            if not is_tumor:
                continue
            if thr is None:
                skipped[rule] += 1
                dices[rule].append(0.0)
                continue
            pred = predicted_mask(scores, thr, xy, tile, gt['downsample'], gt['shape'])
            if pred.sum() == 0:
                empty[rule] += 1
            dices[rule].append(dice(pred, gt['mask']))
    out = {}
    for rule in rules:
        s, at, _, _ = froc(froc_data[rule])
        out[rule] = {'froc': s, 'sens_at_fp': dict(zip(map(str, FROC_FPS), map(float, at))),
                     'dice': float(np.mean(dices[rule])), 'n_dice_slides': len(dices[rule]),
                     'tumor_slides_skipped': skipped[rule], 'tumor_slides_empty_mask': empty[rule],
                     'n_lesions': int(sum(n for _, _, n in froc_data[rule])), 'n_tumor_slides': int(n_tumor)}
    return out


def main():
    args = get_arguments()
    with open(args.partA, 'rb') as f:
        pa = pickle.load(f)
    with open(args.partB, 'rb') as f:
        pb = pickle.load(f)
    A, B = pa['slides'], pb['slides']
    names = [n for n in A if n in B]
    print(f'{len(names)} slides scored', flush=True)

    gt_cache, results = {}, {}
    for row, builder in ROWS.items():
        print(f'== {row}', flush=True)
        mean_res = score_row(builder, A, B, names, args, gt_cache, 'mean', ['0.5', '1-p'])
        max_res = score_row(builder, A, B, names, args, gt_cache, 'max', ['0.5'])
        results[row] = {
            'froc-0.5': mean_res['0.5']['froc'],
            'froc-1-p': mean_res['1-p']['froc'],
            'dice-0.5': mean_res['0.5']['dice'],
            'dice-1-p': mean_res['1-p']['dice'],
            'froc-0.5-mean-agg': mean_res['0.5']['froc'],
            'froc-0.5-max-agg': max_res['0.5']['froc'],
            'dice-0.5-mean-agg': mean_res['0.5']['dice'],
            'dice-0.5-max-agg': max_res['0.5']['dice'],
            'detail': {'mean': mean_res, 'max': max_res},
        }
        print(json.dumps({k: v for k, v in results[row].items() if k != 'detail'}, indent=1), flush=True)

    results['_meta'] = {'n_slides': len(names), 'partA_sanity': pa['sanity'], 'partB_sanity': pb['sanity']}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=1)
    print('saved', args.out, flush=True)


if __name__ == '__main__':
    main()
