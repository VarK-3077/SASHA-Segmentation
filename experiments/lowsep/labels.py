# Per-slide unit grid with labels: for every 2048 px unit on the (0, 0)-anchored grid, whether it
# holds tissue (the segmenter's Otsu mask at level 5), its tumour fraction, its bucket (normal /
# core / boundary / small, same rules as experiments/sep) and whether its normal label can be
# trusted (not a non-exhaustively annotated training slide). Run once, all slides, CPU pool.

import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
import openslide

from experiments.patch_common.ncrf_non_exhaustive import NON_EXHAUSTIVE_TUMOR_SLIDES
from experiments.sep import buckets as B
from experiments.sep.build_labels import lesion_lookup
from experiments.lowsep import paths as P
from utils.segmentation_metrics import rasterize_mask, read_xml_polygons


def tissue_mask(slide):
    img = np.asarray(slide.read_region((0, 0), P.TISSUE_LEVEL, slide.level_dimensions[P.TISSUE_LEVEL]).convert('RGB'))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    _, m = cv2.threshold(hsv[..., 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (m > 0) & (hsv[..., 2] > 40)


def block_reduce(a, k, fn):
    h, w = a.shape
    return fn(a.reshape(h // k, k, w // k, k), axis=(1, 3))


def pad_to(a, h, w, value=0):
    return np.pad(a, ((0, h - a.shape[0]), (0, w - a.shape[1])), constant_values=value)


def build(name):
    s = openslide.OpenSlide(P.slide_path(name))
    w, h = s.dimensions
    mpp = float(s.properties.get('openslide.mpp-x', 0.243))
    assert s.level_count > max(P.LEVELS), (name, s.level_count)
    hu, wu = -(-h // P.UNIT), -(-w // P.UNIT)
    t = tissue_mask(s)
    k5 = P.UNIT // 2 ** P.TISSUE_LEVEL
    tissue = block_reduce(pad_to(t, hu * k5, wu * k5), k5, np.any)
    tum, exc = read_xml_polygons(P.xml_path(name)) if os.path.exists(P.xml_path(name)) else ([], [])
    km = P.UNIT // P.MASK_DS
    frac = np.zeros((hu, wu), np.float32)
    bucket = np.zeros((hu, wu), np.int8)
    if tum:
        mask = pad_to(rasterize_mask(tum, exc, (w, h), P.MASK_DS), hu * km, wu * km)
        labels, area_l0, _ = lesion_lookup(mask, mpp)
        frac = block_reduce(mask.astype(np.float32), km, np.mean)
        area = block_reduce(area_l0[labels], km, np.max)
        bucket = B.assign(frac, area, P.UNIT).astype(np.int8)
    trust = name not in NON_EXHAUSTIVE_TUMOR_SLIDES
    os.makedirs(os.path.dirname(P.label_npz(name)), exist_ok=True)
    np.savez(P.label_npz(name), tissue=tissue | (frac > 0), frac=frac, bucket=bucket, trust=trust, mpp=mpp, dims=(w, h))
    return name, int(tissue.sum()), int((frac > 0).sum())


def main(nproc):
    names = sorted(P.split_of())
    todo = [n for n in names if not os.path.exists(P.label_npz(n))]
    with Pool(nproc) as pool:
        for i, (n, nt, ntum) in enumerate(pool.imap_unordered(build, todo), 1):
            if i % 25 == 0:
                print(f'{i}/{len(todo)} {n}: {nt} tissue units, {ntum} tumour units', flush=True)
    print('labels done', flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
