"""
Builds per-slide ground-truth tumor-fraction label tables for CAMELYON16 patches.

For every slide in the LR feature h5, rasterizes the lesion-annotation xml at level 5
(downsample 32, following step11_segmentation_eval.py's convention) and computes the
tumor fraction covered by each 512px sub-patch (16 per 2048px tile) and by each
2048px tile itself. Slides without an xml (normal slides) get all-zero fractions.
Also cross-checks the csv slide label against xml presence and writes a small
per-slide summary csv alongside the pickle.

python patch_cls/build_labels.py
"""

import argparse
import csv
import os
import pickle
import time
from datetime import date

import h5py
import numpy as np
import openslide
from tqdm import tqdm

from patch_cls.ncrf_non_exhaustive import NON_EXHAUSTIVE_TUMOR_SLIDES, QUOTE, SOURCE_URL
from utils.segmentation_metrics import rasterize_mask, read_xml_polygons, tile_footprints

H5_PATH = '/data2/tej/SASHA/camelyon_lr_features/patch_feats_pretrain_medical_ssl.h5'
XML_DIR = '/data1/namanm/dataset/camelyon16/annotations'
SLIDE_DIR = '/data1/namanm/dataset/camelyon16/images/all'
MASK_DIR = '/data1/namanm/dataset/camelyon16/masks'
CSV_PATH = 'dataset_csv/camelyon16/camelyon16.csv'
OUT_DIR = os.path.expanduser('~/seg_outputs/patch_cls')

TILE_L0 = 2048
SUB_L0 = 512
N_SUB = 16
DOWNSAMPLE = 32
LEVEL = 5

SUB_OFFSETS = np.array([[(k // 4) * SUB_L0, (k % 4) * SUB_L0] for k in range(N_SUB)])


def get_arguments():
    parser = argparse.ArgumentParser('build CAMELYON16 patch ground-truth label tables')
    parser.add_argument('--h5', default=H5_PATH)
    parser.add_argument('--xml_dir', default=XML_DIR)
    parser.add_argument('--slide_dir', default=SLIDE_DIR)
    parser.add_argument('--mask_dir', default=MASK_DIR)
    parser.add_argument('--csv', default=CSV_PATH)
    parser.add_argument('--out_dir', default=OUT_DIR)
    return parser.parse_args()


def load_slide_labels(csv_path):
    with open(csv_path) as f:
        return {row['slide_id']: int(row['label']) for row in csv.DictReader(f)}


def slide_dims(name, slide_dir, mask_dir):
    path = os.path.join(slide_dir, name + '.tif')
    if not os.path.exists(path):
        path = os.path.join(mask_dir, name + '_mask.tif')
    slide = openslide.OpenSlide(path)
    dims = slide.dimensions
    slide.close()
    return dims


def footprint_fractions(coords, tile_l0, mask):
    fracs = np.zeros(len(coords), dtype=np.float32)
    for i, (r0, r1, c0, c1) in enumerate(tile_footprints(coords, tile_l0, DOWNSAMPLE, mask.shape)):
        if r1 > r0 and c1 > c0:
            fracs[i] = mask[r0:r1, c0:c1].mean()
    return fracs


def build_slide_entry(coords, xml_path, dims, slide_label):
    n = len(coords)
    if xml_path is None:
        return {'sub_frac': np.zeros((n, N_SUB), dtype=np.float16),
                'tile_frac': np.zeros(n, dtype=np.float16),
                'slide_label': slide_label}
    mask = rasterize_mask(*read_xml_polygons(xml_path), dims, DOWNSAMPLE)
    tile_frac = footprint_fractions(coords, TILE_L0, mask)
    sub_coords = (coords[:, None, :] + SUB_OFFSETS[None, :, :]).reshape(-1, 2)
    sub_frac = footprint_fractions(sub_coords, SUB_L0, mask).reshape(n, N_SUB)
    return {'sub_frac': sub_frac.astype(np.float16),
            'tile_frac': tile_frac.astype(np.float16),
            'slide_label': slide_label}


def sanity_report(labels, names):
    tumor_names = [n for n in names if labels[n]['slide_label'] == 1]
    all_sub = np.concatenate([labels[n]['sub_frac'].reshape(-1) for n in names])
    tumor_sub = np.concatenate([labels[n]['sub_frac'].reshape(-1) for n in tumor_names]) if tumor_names else np.array([])
    mismatches = 0
    for n in names:
        tf, sf = labels[n]['tile_frac'], labels[n]['sub_frac']
        mismatches += int(np.sum((tf > 0) != (sf.max(axis=1) > 0))) if len(tf) else 0
    print(f'total sub-patches: {all_sub.size}')
    print(f'fraction sub_frac>0 overall: {(all_sub > 0).mean():.5f}')
    if tumor_sub.size:
        print(f'fraction sub_frac>0 among tumor slides: {(tumor_sub > 0).mean():.5f}')
    print(f'tile_frac>0 vs sub_frac.max>0 mismatches: {mismatches}')
    for spot in ('test_021', 'tumor_009'):
        if spot in labels:
            tf = labels[spot]['tile_frac']
            print(f'{spot}: n_tiles={len(tf)}, pct_tumor_tiles={(tf > 0).mean():.3f}, max_tile_frac={tf.max():.3f}')


def main():
    args = get_arguments()
    t0 = time.time()
    slide_labels = load_slide_labels(args.csv)

    with h5py.File(args.h5, 'r') as f:
        names = list(f.keys())
        coords_by_name = {name: f[name]['coords'][:] for name in names}

    labels = {}
    mismatches = []
    for name in tqdm(names, desc='slides'):
        xml_path = os.path.join(args.xml_dir, name + '.xml')
        xml_path = xml_path if os.path.exists(xml_path) else None
        csv_label = slide_labels.get(name)
        has_xml = xml_path is not None
        if csv_label is not None and int(has_xml) != csv_label:
            mismatches.append(name)
            print(f'warning: {name} csv label={csv_label} but xml {"found" if has_xml else "missing"}')
        dims = slide_dims(name, args.slide_dir, args.mask_dir) if has_xml else None
        labels[name] = build_slide_entry(coords_by_name[name], xml_path, dims, csv_label)

    labels['_meta'] = {
        'downsample': DOWNSAMPLE,
        'level': LEVEL,
        'date': str(date.today()),
        'non_exhaustive_tumor_train_slides': NON_EXHAUSTIVE_TUMOR_SLIDES,
        'non_exhaustive_source_url': SOURCE_URL,
        'non_exhaustive_quote': QUOTE,
        'slide_label_xml_mismatches': mismatches,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'labels.pkl'), 'wb') as f:
        pickle.dump(labels, f)

    with open(os.path.join(args.out_dir, 'label_summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['slide_id', 'n_tiles', 'pct_tumor_tiles'])
        for name in names:
            tf = labels[name]['tile_frac']
            w.writerow([name, len(tf), float((tf > 0).mean()) if len(tf) else 0.0])

    print(f'label/xml mismatches: {mismatches}')
    sanity_report(labels, names)
    print(f'runtime: {time.time() - t0:.1f}s')
    print('saved', os.path.join(args.out_dir, 'labels.pkl'))
    print('saved', os.path.join(args.out_dir, 'label_summary.csv'))


if __name__ == '__main__':
    main()
