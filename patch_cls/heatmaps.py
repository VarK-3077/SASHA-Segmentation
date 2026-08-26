# Builds the score/coordinate arrays for the three probability heatmaps scored by
# score_heatmaps.py, at whatever tile granularity each one is defined on: model1-full-hr
# and combination live on the 512px sub-patch grid, sasha+model2 on the 2048px tile grid.
# Sub-patch k of tile (x, y) sits at (x + (k // 4) * 512, y + (k % 4) * 512), the same
# layout build_labels.py uses for the ground-truth tumor-fraction tables.

import numpy as np

from patch_cls.build_labels import N_SUB, SUB_OFFSETS


def sub_patch_coords(coords):
    """(N,2) tile top-lefts -> (N*16,2) sub-patch top-lefts, tile-major k order."""
    return (coords[:, None, :] + SUB_OFFSETS[None, :, :]).reshape(-1, 2)


def model1_row(coords, probs1):
    """model1-full-hr: (N,16) sub-patch probs -> flat (scores, coords) on the 512px grid."""
    return probs1.astype(np.float32).reshape(-1), sub_patch_coords(coords)


def model2_row(coords, probs2):
    """sasha+model2: (N,) tile probs -> (scores, coords) unchanged, on the 2048px grid."""
    return probs2.astype(np.float32), coords


def combination_row(coords, probs1, probs2, visited):
    """visited tiles keep their 16 model1 sub-patch probs; every other tile gets its model2
    tile prob replicated over the same 16 sub-patch positions. Output is on the 512px grid."""
    visited_mask = np.zeros(len(coords), dtype=bool)
    visited_mask[np.asarray(visited, dtype=np.int64)] = True
    p1, p2 = probs1.astype(np.float32), probs2.astype(np.float32)
    scores = np.where(visited_mask[:, None], p1, np.repeat(p2[:, None], N_SUB, axis=1))
    return scores.reshape(-1), sub_patch_coords(coords)
