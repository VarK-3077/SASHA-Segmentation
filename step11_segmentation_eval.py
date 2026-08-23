"""
Scores the attention dump from step10 against CAMELYON16 lesion annotations.

Heatmap -> mask follows the ASMIL protocol: per-slide attention is min-max rescaled to [0,1]
and binarised at a threshold. Two threshold rules are evaluated: a fixed 0.5, and a
confidence-driven one where slides SASHA calls normal get an empty mask and the rest use
thr = f(p_tumor). Reports Dice on tumor slides, tile specificity on normal slides and the
official lesion-level FROC, separately per split.

python step11_segmentation_eval.py --dump OUT/attention_dump.pkl --slide_dir WSI_DIR --xml_dir XML_DIR
"""

import argparse
import json
import os
import pickle

import numpy as np
import openslide
from tqdm import tqdm

from utils.segmentation_metrics import (dice, evaluation_mask, fp_tp_probs, froc, itc_labels, minmax,
                                        nms_candidates, predicted_mask, rasterize_mask, read_xml_polygons)

RULES = {
    'fixed_0.5': lambda p: 0.5,
    'conf_1-p': lambda p: None if p < 0.5 else 1.0 - p,
    'conf_p': lambda p: None if p < 0.5 else p,
}


def get_arguments():
    parser = argparse.ArgumentParser('SASHA segmentation eval')
    parser.add_argument('--dump', required=True)
    parser.add_argument('--slide_dir', required=True, help='dir with <slide>.tif (may be nested, searched recursively)')
    parser.add_argument('--xml_dir', required=True, help='dir with <slide>.xml lesion annotations (searched recursively)')
    parser.add_argument('--mask_dir', default=None, help='dir with <slide>_mask.tif; used for geometry when the slide is missing')
    parser.add_argument('--out', default=None, help='json path for results (default: next to dump)')
    parser.add_argument('--level', type=int, default=5, help='mask level (downsample 2^level)')
    parser.add_argument('--patch_level', type=int, default=3, help='level the LR patches were cut at')
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--nms_radius', type=int, default=1, help='NMS radius in LR tiles for FROC candidates')
    parser.add_argument('--sources', default='sasha,hafed')
    parser.add_argument('--score', default='softmax', choices=['softmax', 'log'],
                        help='softmax: mean-head attention; log: mean over heads of log-attention (= pre-softmax logits)')
    return parser.parse_args()


def find_file(root, name):
    for d, _, files in os.walk(root):
        if name in files:
            return os.path.join(d, name)
    return None


def slide_geometry(slide_path, level):
    s = openslide.OpenSlide(slide_path)
    mpp = float(s.properties.get('openslide.mpp-x', 0.243))
    dims = s.dimensions
    s.close()
    return dims, mpp, 2 ** level


def ground_truth(name, args, cache):
    """Returns dict(mask, eval_mask, itcs, downsample, mpp, shape); mask is None for normal slides."""
    if name in cache:
        return cache[name]
    slide_path = find_file(args.slide_dir, name + '.tif')
    if slide_path is None and args.mask_dir:
        slide_path = find_file(args.mask_dir, name + '_mask.tif')
    if slide_path is None:
        raise FileNotFoundError(f'no slide or mask for {name}')
    dims, mpp, ds = slide_geometry(slide_path, args.level)
    xml_path = find_file(args.xml_dir, name + '.xml')
    gt = {'downsample': ds, 'mpp': mpp, 'shape': (int(np.ceil(dims[1] / ds)), int(np.ceil(dims[0] / ds)))}
    if xml_path is None:
        gt.update(mask=None, eval_mask=None, itcs=[])
    else:
        mask = rasterize_mask(*read_xml_polygons(xml_path), dims, ds)
        em = evaluation_mask(mask, mpp, ds)
        gt.update(mask=mask, eval_mask=em, itcs=itc_labels(em, mpp, ds))
    cache[name] = gt
    return gt


def slide_scores(e, kind):
    if kind == 'log':
        return np.log(e['attention_heads'] + 1e-12).mean(0)
    return e['attention']


def evaluate_split(entries, coords, args, gt_cache, tile_l0):
    per_rule = {r: {'dice': [], 'spec': [], 'skipped_tumor': 0, 'empty_tumor': 0} for r in RULES}
    froc_data, n_tumor = [], 0
    for name, e in tqdm(entries.items(), desc='slides'):
        gt = ground_truth(name, args, gt_cache)
        is_tumor = gt['mask'] is not None
        n_tumor += is_tumor
        if is_tumor != bool(e['label']):
            print(f'warning: {name} label={e["label"]} but xml {"found" if is_tumor else "missing"}')
        scores = minmax(slide_scores(e, args.score))
        xy = coords[name]
        froc_data.append(fp_tp_probs(nms_candidates(scores, xy, tile_l0, args.nms_radius), is_tumor,
                                     gt['eval_mask'], gt['itcs'], gt['downsample']))
        for rule, f in RULES.items():
            thr = f(e['p_tumor'])
            pred_tiles = np.zeros(len(scores), dtype=bool) if thr is None else scores >= thr
            if is_tumor:
                if thr is None:
                    per_rule[rule]['skipped_tumor'] += 1
                    per_rule[rule]['dice'].append(0.0)
                    continue
                pred = predicted_mask(scores, thr, xy, tile_l0, gt['downsample'], gt['shape'])
                if pred.sum() == 0:
                    per_rule[rule]['empty_tumor'] += 1
                per_rule[rule]['dice'].append(dice(pred, gt['mask']))
            else:
                per_rule[rule]['spec'].append(1.0 - pred_tiles.mean())
    score, at, _, _ = froc(froc_data)
    res = {'n_slides': len(entries), 'n_tumor': n_tumor,
           'froc': score, 'froc_sens_at_fp': dict(zip(map(str, (0.25, 0.5, 1, 2, 4, 8)), map(float, at))),
           'n_lesions': int(sum(n for _, _, n in froc_data)), 'rules': {}}
    for rule, d in per_rule.items():
        res['rules'][rule] = {'dice_tumor_slides': float(np.mean(d['dice'])) if d['dice'] else None,
                              'specificity_normal_slides': float(np.mean(d['spec'])) if d['spec'] else None,
                              'tumor_slides_skipped_as_normal': d['skipped_tumor'],
                              'tumor_slides_empty_mask': d['empty_tumor']}
    return res


def main():
    args = get_arguments()
    with open(args.dump, 'rb') as f:
        dump = pickle.load(f)
    tile_l0 = args.patch_size * (2 ** args.patch_level)
    gt_cache = {}
    results = {}
    for source in args.sources.split(','):
        for split, entries in dump[source].items():
            print(f'== {source} / {split} ({len(entries)} slides)')
            results[f'{source}/{split}'] = evaluate_split(entries, dump['coords'], args, gt_cache, tile_l0)
            print(json.dumps(results[f'{source}/{split}'], indent=1))
    out = args.out or os.path.join(os.path.dirname(args.dump), f'segmentation_results_{args.score}.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=1)
    print('saved', out)


if __name__ == '__main__':
    main()
