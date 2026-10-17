# Scoring of a finished episode and of the label-free and label-using baselines, all with the
# same sub-patch Dice the environment rewards. Bucket rates say what the agent did with each
# kind of tile: zoomed it, called it core, or left it as normal.

import numpy as np

from experiments.rlbin.env import CORE, ZOOMED, entropy
from experiments.sep import buckets as B


def dice_of(pred, gt):
    d = pred.sum() + gt.sum()
    return 1.0 if d == 0 else 2.0 * (pred & gt).sum() / d


def compose(slide, status):
    pred = np.zeros_like(slide.gt)
    pred[status == ZOOMED] = slide.p_hr[status == ZOOMED] >= 0.5
    pred[status == CORE] = True
    return pred


def episode_metrics(slide, status, steps):
    pred = compose(slide, status)
    out = {'dice': dice_of(pred, slide.gt), 'tumour': slide.gt_pos > 0, 'zoom_frac': float((status == ZOOMED).mean()),
           'steps_frac': steps / slide.n, 'fp_sub': int((pred & ~slide.gt).sum()), 'n': slide.n}
    for b, name in enumerate(B.NAMES):
        m = slide.bucket == b
        if m.any():
            out[f'{name}_zoom'] = float((status[m] == ZOOMED).mean())
            out[f'{name}_core'] = float((status[m] == CORE).mean())
            out[f'{name}_n'] = int(m.sum())
    return out


def aggregate(rows):
    tum = [r for r in rows if r['tumour']]
    nor = [r for r in rows if not r['tumour']]
    out = {'dice_all': float(np.mean([r['dice'] for r in rows])),
           'dice_tumour': float(np.mean([r['dice'] for r in tum])) if tum else None,
           'normal_slides_clean': float(np.mean([r['fp_sub'] == 0 for r in nor])) if nor else None,
           'fp_sub_per_normal_slide': float(np.mean([r['fp_sub'] for r in nor])) if nor else None,
           'zoom_frac': float(np.mean([r['zoom_frac'] for r in rows])),
           'steps_frac': float(np.mean([r['steps_frac'] for r in rows]))}
    for name in B.NAMES:
        for what in ('zoom', 'core'):
            vals = [(r[f'{name}_{what}'], r[f'{name}_n']) for r in rows if f'{name}_n' in r]
            if vals:
                out[f'{name}_{what}'] = float(sum(v * n for v, n in vals) / sum(n for _, n in vals))
    return out


def baseline_status(slide, kind, zoom_frac=0.0):
    """Label-free routers and the label-using oracle, expressed as a status vector."""
    status = np.zeros(slide.n, np.int8)
    if kind == 'zoom_all':
        status[:] = ZOOMED
    elif kind == 'oracle_route':
        status[slide.bucket == B.CORE] = CORE
        status[np.isin(slide.bucket, (B.BOUNDARY, B.SMALL))] = ZOOMED
    elif kind == 'sup_router':
        pbar = slide.p_lr.mean(1)
        status[pbar >= 0.9] = CORE
        rest = np.flatnonzero(status == 0)
        k = min(int(round(zoom_frac * slide.n)), len(rest))
        status[rest[np.argsort(-entropy(slide.p_lr)[rest])[:k]]] = ZOOMED
    return status


def lr_only_dice(slide):
    return dice_of(slide.p_lr >= 0.5, slide.gt)
