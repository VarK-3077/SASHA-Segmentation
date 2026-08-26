# Scores the three patch-classifier probability heatmaps (model1-full-hr, sasha+model2,
# combination) against CAMELYON16 lesion annotations, reusing step11_segmentation_eval's
# ground-truth conventions (level-5 masks, official FROC) and utils/segmentation_metrics.py.
# Probabilities are calibrated already -- no min-max rescaling, unlike the attention-map
# eval in step11. Two threshold rules per row: a fixed 0.5, and a SASHA-gated one that
# empties the mask/candidates on slides the rollout's own p_tumor calls normal.
#
# python -m patch_cls.score_heatmaps

import argparse
import json
import os
import pickle
import sys
import time
from types import SimpleNamespace

import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from patch_cls.heatmaps import combination_row, model1_row, model2_row
from step11_segmentation_eval import ground_truth
from utils.segmentation_metrics import dice, fp_tp_probs, froc, nms_candidates, predicted_mask

HOME = os.path.expanduser('~')
H5_PATH = '/data2/tej/SASHA/camelyon_lr_features/patch_feats_pretrain_medical_ssl.h5'
XML_DIR = '/data1/namanm/dataset/camelyon16/annotations'
SLIDE_DIR = '/data1/namanm/dataset/camelyon16/images/all'
MASK_DIR = '/data1/namanm/dataset/camelyon16/masks'
SPLIT_JSON = f'{HOME}/SASHA-Segmentation/dataset_csv/camelyon16/splits/split_4.json'
MODEL1_PROBS = f'{HOME}/seg_outputs/patch_cls/model1/test_probs.pkl'
MODEL2_PROBS = f'{HOME}/seg_outputs/patch_cls/model2/test_probs.pkl'
ROLLOUT_STATES = f'{HOME}/seg_outputs/patch_cls/rollout_states.pkl'
OUT = f'{HOME}/seg_outputs/patch_cls/final_metrics.json'

LEVEL = 5
GATE_THR = 0.5
DICE_THR = 0.5
DICE_SWEEP = (0.1, 0.25, 0.5, 0.75, 0.9)
FROC_FPS = (0.25, 0.5, 1, 2, 4, 8)

ROWS = ('model1-full-hr', 'sasha+model2', 'combination')
TILE_SIZE = {'model1-full-hr': 512, 'sasha+model2': 2048, 'combination': 512}
NMS_RADIUS = {'model1-full-hr': 4, 'sasha+model2': 1, 'combination': 4}
HAS_SWEEP = {'model1-full-hr': True, 'sasha+model2': True, 'combination': False}


def get_arguments():
    p = argparse.ArgumentParser('score patch-classifier heatmaps on CAMELYON16 test')
    p.add_argument('--limit', type=int, default=None, help='only score the first N test slides (debugging)')
    p.add_argument('--out', default=OUT)
    return p.parse_args()


def build_row(row, coords, p1, p2, visited):
    if row == 'model1-full-hr':
        return model1_row(coords, p1)
    if row == 'sasha+model2':
        return model2_row(coords, p2)
    return combination_row(coords, p1, p2, visited)


def slide_dice(scores, coords, thr, tile_size, g):
    pred = predicted_mask(scores, thr, coords, tile_size, g['downsample'], g['shape'])
    return dice(pred, g['mask']), bool(pred.sum() == 0)


def main():
    args = get_arguments()
    with open(SPLIT_JSON) as f:
        names = sorted(json.load(f)['test_names'])
    if args.limit:
        names = names[:args.limit]

    with open(MODEL1_PROBS, 'rb') as f:
        probs1 = pickle.load(f)
    with open(MODEL2_PROBS, 'rb') as f:
        probs2 = pickle.load(f)
    with open(ROLLOUT_STATES, 'rb') as f:
        rollout = pickle.load(f)['slides']

    missing = [n for n in names if n not in probs1 or n not in probs2 or n not in rollout]
    assert not missing, f'missing slides in one of the inputs: {missing}'
    assert all(rollout[n]['split'] == 'test' for n in names), 'rollout entry not from the test split'

    with h5py.File(H5_PATH, 'r') as f:
        coords = {n: f[n]['coords'][:] for n in names}

    p_tumor = {n: rollout[n]['p_tumor'] for n in names}
    visited = {n: rollout[n]['visited'] for n in names}

    gt_args = SimpleNamespace(slide_dir=SLIDE_DIR, mask_dir=MASK_DIR, xml_dir=XML_DIR, level=LEVEL)
    gt_cache = {}
    gt = {n: ground_truth(n, gt_args, gt_cache) for n in tqdm(names, desc='ground truth')}
    tumor_names = [n for n in names if gt[n]['mask'] is not None]

    n_lesions = sum(int(gt[n]['eval_mask'].max()) - len(gt[n]['itcs']) for n in tumor_names)
    gated_skip = {n for n in names if p_tumor[n] < GATE_THR}
    gated_skip_tumor = sorted(gated_skip & set(tumor_names))

    results = {
        'n_test_slides': len(names),
        'n_tumor_slides': len(tumor_names),
        'n_lesions': n_lesions,
        'gate_threshold': GATE_THR,
        'gated_slides_skipped_total': len(gated_skip),
        'gated_tumor_slides_skipped': gated_skip_tumor,
        'rows': {},
    }
    print(f'n_test_slides={len(names)} n_tumor_slides={len(tumor_names)} n_lesions={n_lesions} '
          f'gated_tumor_slides_skipped={len(gated_skip_tumor)} {gated_skip_tumor}', flush=True)

    for row in ROWS:
        t0 = time.time()
        tile_size, radius = TILE_SIZE[row], NMS_RADIUS[row]
        scores, rc = {}, {}
        for n in names:
            s, c = build_row(row, coords[n], probs1[n], probs2[n], visited[n])
            scores[n], rc[n] = s, c

        # FROC candidates come from the raw (ungated) map; the gated rule just empties the
        # per-slide list before scoring, so we only run NMS once per slide.
        candidates = {n: nms_candidates(scores[n], rc[n], tile_size, radius) for n in names}

        def run_froc(gated):
            per_slide = []
            for n in names:
                cand = [] if (gated and n in gated_skip) else candidates[n]
                per_slide.append(fp_tp_probs(cand, n in tumor_names, gt[n]['eval_mask'], gt[n]['itcs'], gt[n]['downsample']))
            score, at, _, _ = froc(per_slide)
            return {'score': score, 'sens_at_fp': dict(zip(map(str, FROC_FPS), map(float, at)))}

        def run_dice(thr, gated):
            vals, empty = [], []
            for n in tumor_names:
                if gated and n in gated_skip:
                    vals.append(0.0)
                    continue
                d, is_empty = slide_dice(scores[n], rc[n], thr, tile_size, gt[n])
                vals.append(d)
                if is_empty:
                    empty.append(n)
            return {'mean': float(np.mean(vals)) if vals else 0.0, 'empty_tumor_slides': empty}

        row_res = {
            'tile_size_l0': tile_size,
            'nms_radius_tiles': radius,
            'dice_0.5': run_dice(DICE_THR, gated=False),
            'dice_gated': dict(run_dice(DICE_THR, gated=True), skipped_tumor_slides=len(gated_skip_tumor)),
            'froc_0.5': run_froc(gated=False),
            'froc_gated': run_froc(gated=True),
        }
        if HAS_SWEEP[row]:
            row_res['dice_sweep'] = {str(t): run_dice(t, gated=False)['mean'] for t in DICE_SWEEP}

        results['rows'][row] = row_res
        print(f'== {row} ({time.time() - t0:.0f}s) ==', flush=True)
        print(json.dumps(row_res, indent=1), flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=1)
    print('saved', args.out)


if __name__ == '__main__':
    main()
