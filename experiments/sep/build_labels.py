# Builds the per-crop label tables for the separability study. For every slide in the
# level-3 feature h5 it rasterises the lesion xml at level 4, labels connected lesions, and
# for each pyramid level 3..0 subdivides each 2048 px tile into 256 px crops, recording per
# crop the tumour fraction, the largest lesion it touches (area and major axis), whether it
# lies near a lesion, and the resulting bucket. Also dumps a per-lesion size table.

import csv
import os
import sys
from multiprocessing import Pool

import cv2
import h5py
import numpy as np
import openslide
from skimage import measure

from utils.segmentation_metrics import read_xml_polygons, rasterize_mask
from experiments.sep import buckets as B
from experiments.sep import paths as P

NEAR_FOOTPRINTS = 2  # a normal crop within this many footprints of a lesion is a hard negative


def slide_coords():
    with h5py.File(P.LR_H5, 'r') as f:
        return {k: f[k]['coords'][:].astype(np.int64) for k in f.keys()}


def lesion_lookup(mask, mpp):
    labels = measure.label(mask, connectivity=2)
    props = measure.regionprops(labels)
    area_l0 = np.zeros(len(props) + 1, np.float64)
    major_um = np.zeros(len(props) + 1, np.float64)
    for p in props:
        area_l0[p.label] = p.area * P.MASK_DS ** 2
        major_um[p.label] = p.axis_major_length * P.MASK_DS * mpp
    return labels, area_l0, major_um


def crop_offsets(k, fp_l0):
    return np.stack(np.meshgrid(np.arange(k), np.arange(k), indexing='ij'), -1).reshape(-1, 2) * fp_l0


def normal_tables(coords, L):
    """Slides without lesions: every crop is normal, no mask needed."""
    k = 2 ** (3 - L)
    offs = crop_offsets(k, P.footprint_l0(L))
    n = len(coords) * k * k
    x = (coords[:, None, 0] + offs[None, :, 1]).ravel().astype(np.int32)
    y = (coords[:, None, 1] + offs[None, :, 0]).ravel().astype(np.int32)
    return {f'L{L}_x': x, f'L{L}_y': y, f'L{L}_frac': np.zeros(n, np.float16),
            f'L{L}_area_l0': np.zeros(n, np.float32), f'L{L}_major_um': np.zeros(n, np.float32),
            f'L{L}_near': np.zeros(n, bool), f'L{L}_bucket': np.zeros(n, np.int8)}


def crop_stats(block_mask, block_lab, block_dist, k, area_l0, major_um):
    """Vectorised per-crop stats for one 2048 tile split into k x k crops."""
    fp = block_mask.shape[0] // k
    view = lambda a: a.reshape(k, fp, k, fp)
    frac = view(block_mask.astype(np.float32)).mean(axis=(1, 3))
    area = view(area_l0[block_lab]).max(axis=(1, 3))
    major = view(major_um[block_lab]).max(axis=(1, 3))
    dist = view(block_dist).min(axis=(1, 3))
    return frac.ravel(), area.ravel(), major.ravel(), dist.ravel()


def process(args):
    name, coords = args
    slide_path = os.path.join(P.SLIDE_DIR, name + '.tif')
    if not os.path.exists(slide_path):
        return name, 'missing', []
    s = openslide.OpenSlide(slide_path)
    mpp = float(s.properties.get('openslide.mpp-x', 0.243))
    dims = s.level_dimensions[0]
    s.close()
    xml = os.path.join(P.XML_DIR, name + '.xml')
    tum, exc = read_xml_polygons(xml) if os.path.exists(xml) else ([], [])
    out = {'mpp': mpp, 'n_tiles': len(coords)}
    if not tum:
        for L in P.LEVELS:
            out.update(normal_tables(coords, L))
        np.savez_compressed(os.path.join(P.LABEL_DIR, name + '.npz'), **out)
        return name, 'ok', []
    mask = rasterize_mask(tum, exc, dims, P.MASK_DS)
    tile_px = P.TILE_L0 // P.MASK_DS
    mask = np.pad(mask, ((0, tile_px), (0, tile_px)))
    labels, area_l0, major_um = lesion_lookup(mask, mpp)
    dist = cv2.distanceTransform((1 - mask).astype(np.uint8), cv2.DIST_L2, 5)
    for L in P.LEVELS:
        k = 2 ** (3 - L)
        fp_l0 = P.footprint_l0(L)
        fp_px = fp_l0 // P.MASK_DS
        offs = crop_offsets(k, fp_l0)
        xs, ys, fr, ar, mj, di = [], [], [], [], [], []
        for x, y in coords:
            r0, c0 = int(y // P.MASK_DS), int(x // P.MASK_DS)
            sl = (slice(r0, r0 + tile_px), slice(c0, c0 + tile_px))
            f, a, m, d = crop_stats(mask[sl], labels[sl], dist[sl], k, area_l0, major_um)
            xs.append(x + offs[:, 1]); ys.append(y + offs[:, 0])
            fr.append(f); ar.append(a); mj.append(m); di.append(d)
        frac = np.concatenate(fr); area = np.concatenate(ar); major = np.concatenate(mj)
        dist_px = np.concatenate(di)
        out[f'L{L}_x'] = np.concatenate(xs).astype(np.int32)
        out[f'L{L}_y'] = np.concatenate(ys).astype(np.int32)
        out[f'L{L}_frac'] = frac.astype(np.float16)
        out[f'L{L}_area_l0'] = area.astype(np.float32)
        out[f'L{L}_major_um'] = major.astype(np.float32)
        out[f'L{L}_near'] = dist_px <= NEAR_FOOTPRINTS * fp_px
        out[f'L{L}_bucket'] = B.assign(frac, area, fp_l0)
    np.savez_compressed(os.path.join(P.LABEL_DIR, name + '.npz'), **out)
    lesions = [(name, i, area_l0[i] * (mpp / 1000) ** 2, major_um[i]) for i in range(1, len(area_l0))]
    return name, 'ok', lesions


def main():
    coords = slide_coords()
    todo = sorted(coords.items())
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    rows, missing = [], []
    with Pool(nproc) as pool:
        for i, (name, status, lesions) in enumerate(pool.imap_unordered(process, todo), 1):
            if status == 'missing':
                missing.append(name)
            rows.extend(lesions)
            if i % 25 == 0:
                print(f'{i}/{len(todo)} slides', flush=True)
    with open(os.path.join(P.ROOT, 'lesions.csv'), 'w') as f:
        w = csv.writer(f)
        w.writerow(['slide', 'lesion', 'area_mm2', 'major_um'])
        w.writerows(rows)
    with open(os.path.join(P.ROOT, 'missing_slides.txt'), 'w') as f:
        f.write('\n'.join(missing))
    print(f'done: {len(todo) - len(missing)} slides, {len(missing)} missing, {len(rows)} lesions')


if __name__ == '__main__':
    main()
