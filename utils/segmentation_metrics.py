"""
Tumor-localization metrics for CAMELYON16 from patch-level attention heatmaps.

Ground truth comes from the lesion-annotation XMLs (rasterized at a coarse level), the
lesion-level FROC follows the official CAMELYON16 Evaluation_FROC.py (75 um evaluation
mask, ITC exclusion, mean sensitivity at 0.25..8 FP/WSI), and Dice is pixel-level at the
same coarse level between the union of predicted tile footprints and the tumor mask.
"""

import xml.etree.ElementTree as ET

import cv2
import numpy as np
from scipy import ndimage as nd
from skimage import measure

FROC_FPS = (0.25, 0.5, 1, 2, 4, 8)
EXCLUSION_GROUPS = {'_2', 'exclusion'}


def read_xml_polygons(xml_path):
    """Returns (tumor_polygons, exclusion_polygons) as lists of (K,2) level-0 int arrays."""
    root = ET.parse(xml_path).getroot()
    tumor, exclusion = [], []
    for anno in root.iter('Annotation'):
        pts = np.array([[float(c.get('X')), float(c.get('Y'))] for c in anno.iter('Coordinate')])
        if len(pts) < 3:
            continue
        group = (anno.get('PartOfGroup') or '').lower()
        (exclusion if group in EXCLUSION_GROUPS else tumor).append(pts)
    return tumor, exclusion


def rasterize_mask(tumor, exclusion, level0_dims, downsample):
    """Binary uint8 tumor mask at the given downsample (shape H x W)."""
    w, h = int(np.ceil(level0_dims[0] / downsample)), int(np.ceil(level0_dims[1] / downsample))
    mask = np.zeros((h, w), dtype=np.uint8)
    scale = lambda polys: [np.round(p / downsample).astype(np.int32).reshape(-1, 1, 2) for p in polys]
    if tumor:
        cv2.fillPoly(mask, scale(tumor), 1)
    if exclusion:
        cv2.fillPoly(mask, scale(exclusion), 0)
    return mask


def evaluation_mask(mask, mpp, downsample):
    """Official CAMELYON16 evaluation mask: tumor dilated by 75 um, holes filled, labelled."""
    distance = nd.distance_transform_edt(1 - mask)
    threshold = 75 / (mpp * downsample * 2)
    filled = nd.binary_fill_holes(distance < threshold)
    return measure.label(filled, connectivity=2)


def itc_labels(eval_mask, mpp, downsample):
    """Labels whose major axis is below 275 um (isolated tumor cells) are ignored."""
    threshold = 275 / (mpp * downsample)
    props = measure.regionprops(eval_mask)
    return [i + 1 for i, p in enumerate(props) if p.axis_major_length < threshold]


def tile_footprints(coords, tile_size_l0, downsample, shape):
    """Yields (row0, row1, col0, col1) slices of each tile at the mask's downsample."""
    s = tile_size_l0 / downsample
    for x, y in coords:
        c0, r0 = int(x / downsample), int(y / downsample)
        yield r0, min(int(np.ceil(r0 + s)), shape[0]), c0, min(int(np.ceil(c0 + s)), shape[1])


def predicted_mask(scores, thr, coords, tile_size_l0, downsample, shape):
    pred = np.zeros(shape, dtype=np.uint8)
    for keep, (r0, r1, c0, c1) in zip(scores >= thr, tile_footprints(coords, tile_size_l0, downsample, shape)):
        if keep:
            pred[r0:r1, c0:c1] = 1
    return pred


def dice(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return 1.0 if denom == 0 else 2.0 * inter / denom


def minmax(scores):
    lo, hi = scores.min(), scores.max()
    return np.zeros_like(scores) if hi == lo else (scores - lo) / (hi - lo)


def nms_candidates(scores, coords, tile_size_l0, radius_tiles=1, min_score=0.0):
    """Greedy non-max suppression on the tile grid -> list of (score, x_center, y_center).

    Works directly on the attention grid: repeatedly take the highest remaining tile and
    suppress every tile within `radius_tiles` (Chebyshev) of it.
    """
    grid = np.round(coords / tile_size_l0).astype(np.int64)
    alive = np.ones(len(scores), dtype=bool)
    order = np.argsort(-scores)
    out = []
    for i in order:
        if not alive[i] or scores[i] < min_score:
            continue
        out.append((float(scores[i]), float(coords[i, 0] + tile_size_l0 / 2), float(coords[i, 1] + tile_size_l0 / 2)))
        alive &= np.abs(grid - grid[i]).max(axis=1) > radius_tiles
    return out


def fp_tp_probs(candidates, is_tumor, eval_mask, itcs, downsample):
    """Port of compute_FP_TP_Probs from the official script, for one slide."""
    max_label = int(eval_mask.max()) if is_tumor else 0
    tp = np.zeros(max_label, dtype=np.float32)
    fp = []
    for prob, x, y in candidates:
        if not is_tumor:
            fp.append(prob)
            continue
        r, c = int(y / downsample), int(x / downsample)
        hit = eval_mask[min(r, eval_mask.shape[0] - 1), min(c, eval_mask.shape[1] - 1)]
        if hit == 0:
            fp.append(prob)
        elif hit not in itcs and prob > tp[hit - 1]:
            tp[hit - 1] = prob
    return fp, tp, max_label - len(itcs)


def froc(per_slide):
    """per_slide: list of (fp_probs, tp_probs, n_tumors). Returns (score, sens@fps, fps, sens)."""
    fps_all = np.array([p for fp, _, _ in per_slide for p in fp])
    tps_all = np.array([p for _, tp, _ in per_slide for p in tp])
    n_tumors = sum(n for _, _, n in per_slide)
    total_fp, total_tp = [], []
    for thr in sorted(set(fps_all.tolist() + tps_all.tolist()))[1:]:
        total_fp.append((fps_all >= thr).sum())
        total_tp.append((tps_all >= thr).sum())
    total_fp.append(0)
    total_tp.append(0)
    avg_fp = np.asarray(total_fp) / len(per_slide)
    sens = np.asarray(total_tp) / max(n_tumors, 1)
    at = np.interp(FROC_FPS, avg_fp[::-1], sens[::-1])
    return float(at.mean()), at, avg_fp, sens
