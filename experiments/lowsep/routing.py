# Routing rules that turn per-unit bucket probabilities into zoom / select / leave, and their
# score under an ideal segmenter: a zoomed unit returns its ground truth, a selected unit is
# painted all tumour, a left unit stays empty. Unit tumour fractions give the pixel counts.

import numpy as np

ZOOM, SELECT, LEAVE = 0, 1, 2


def ideal(prob, bucket):
    return np.where(bucket == 0, LEAVE, np.where(bucket == 1, SELECT, ZOOM))


def argmax(prob, bucket):
    pred = prob.argmax(1)
    return np.where(pred == 0, LEAVE, np.where(pred == 1, SELECT, ZOOM))


def zoom_all_tumour(prob, bucket):
    """Never select: every unit not predicted normal is zoomed."""
    return np.where(prob.argmax(1) == 0, LEAVE, ZOOM)


def low_conf(tau):
    def rule(prob, bucket):
        act = argmax(prob, bucket)
        act[(prob.argmax(1) == 0) & (prob[:, 0] < tau)] = ZOOM
        return act
    return rule


def sure_select(tau):
    """Select only units the head is at least tau sure are core; other tumour predictions zoom."""
    def rule(prob, bucket):
        act = zoom_all_tumour(prob, bucket)
        act[(prob.argmax(1) == 1) & (prob[:, 1] >= tau)] = SELECT
        return act
    return rule


ROUTES = {'ideal': ideal, 'argmax': argmax, 'zoom_all_tumour': zoom_all_tumour,
          'argmax + zoom if p_normal < 0.9': low_conf(0.9), 'argmax + zoom if p_normal < 0.99': low_conf(0.99),
          'zoom_all_tumour, select if p_core >= 0.9': sure_select(0.9)}


BUDGETS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3)


def frontier(slides):
    """Zoom-only routing at fixed budgets: the top-b fraction of all test units by p(tumour) is
    zoomed (one global threshold, so calibration does not move the operating point), the rest
    left. Returns Dice per budget; the ideal rule's own zoom fraction is the label-based ceiling."""
    p_tum = np.concatenate([1 - s['prob'][:, 0] for s in slides])
    out = {}
    for b in BUDGETS:
        thr = np.quantile(p_tum, 1 - b)
        dice = [(slide_dice(np.where(1 - s['prob'][:, 0] >= thr, ZOOM, LEAVE), s['frac']), s['frac'].sum() > 0) for s in slides]
        out[str(b)] = {'dice_tumour': float(np.mean([d for d, t in dice if t])), 'dice_all': float(np.mean([d for d, _ in dice])),
                       'normal_clean': float(np.mean([d for d, t in dice if not t]))}
    return out


def slide_dice(act, frac):
    inter = frac[act == ZOOM].sum() + frac[act == SELECT].sum()
    pred = frac[act == ZOOM].sum() + (act == SELECT).sum()
    gt = frac.sum()
    return 1.0 if pred + gt == 0 else float(2 * inter / (pred + gt))


def route_scores(slides, name):
    rule = ROUTES[name]
    acts = [rule(s['prob'], s['bucket']) for s in slides]
    dice = [(slide_dice(a, s['frac']), s['frac'].sum() > 0) for a, s in zip(acts, slides)]
    allact = np.concatenate(acts)
    counts = np.bincount(allact, minlength=3)
    tab = np.zeros((3, 3), np.int64)
    for a, s in zip(acts, slides):
        np.add.at(tab, (ideal(None, s['bucket']), a), 1)
    return {'dice_tumour': float(np.mean([d for d, t in dice if t])), 'dice_all': float(np.mean([d for d, _ in dice])),
            'normal_clean': float(np.mean([d for d, t in dice if not t])), 'zoom_frac': float(counts[ZOOM] / len(allact)),
            'counts': {'zoom': int(counts[ZOOM]), 'select': int(counts[SELECT]), 'leave': int(counts[LEAVE])},
            'table_ideal_vs_routed': tab.tolist()}
