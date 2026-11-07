# The sequential env on real segmenter outcomes (seg_tables.py) with a lesion-aware quality. The policy sees
# what the level-2 head says, not the ground truth. J = 1/4 area recall + 1/2 share of non-ITC lesions found
# + 1/4 mean recall per lesion - fp / (fp + gt + 1), all from the head's own map, so a lesion counts the
# same however small it is and false positives cost quality on any slide, tumour or not.

import numpy as np

from experiments.rlscreen import paths as P
from experiments.rlscreen import stoprl as R


def attach(slides):
    """Replaces the ideal outcome with the segmenter's; slides without a map are dropped."""
    out = []
    for s in slides:
        f = f'{P.units_dir("kaiko_vitl14")}/seg/{s.name}.npz'
        try:
            z = np.load(f)
        except OSError:
            continue
        s.tp, s.fp, s.gt = z['tp'].astype(np.float64), z['fp'].astype(np.float64), z['gt'].astype(np.float64)
        s.gt_total = float(s.gt.sum())
        s.pos = z['seg_frac'] > 0
        s.edges = z['seg_edges'] > 0
        s.edge_frac = z['seg_edges']
        s.lu_unit_s, s.lu_lesion_s, s.lu_tp, s.lesion_px = z['lu_unit'], z['lu_lesion'], z['lu_tp'].astype(np.float64), z['lesion_px'].astype(np.float64)
        s.les = np.flatnonzero(s.lesion_px)
        s.tumour = s.gt_total > 0
        out.append(s)
    return out


class SegEp(R.Ep):
    def terms(self):
        s = self.s
        seg = self.zoomed & ~self.painted
        tp = s.tp[seg].sum() + s.gt[self.painted].sum()
        fp = s.fp[seg].sum() + (1 - s.gt[self.painted]).sum()
        found = recall = None
        if len(s.les):
            hit = self.zoomed.reshape(-1)[s.lu_unit_s]
            cov = np.bincount(s.lu_lesion_s[hit], weights=s.lu_tp[hit], minlength=len(s.lesion_px))[s.les]
            found, recall = float((cov > 0).mean()), float(np.minimum(cov / s.lesion_px[s.les], 1).mean())
        return tp, fp, found, recall

    def quality(self):
        s = self.s
        tp, fp, found, recall = self.terms()
        pen = R.CFG['fp'] * fp / (fp + s.gt_total + 1)
        if not s.tumour:
            return 1.0 - pen
        area = tp / s.gt_total
        if found is None:
            return area - pen
        return 0.25 * area + 0.5 * found + 0.25 * recall - pen

    def summary(self):
        s = self.s
        tp, fp, found, recall = self.terms()
        n = max(s.n, 1)
        seg = self.zoomed & ~self.painted
        return dict(tumour=s.tumour, dice=2 * tp / (tp + fp + s.gt_total) if s.tumour else None, found=found, screened=self.screened.sum() / n,
                    l2=seg.sum() / n, painted=self.painted.sum() / n, fp_units=float(fp), sec=R.GLOBAL_S[R.CFG['design']] + self.sec,
                    rounds=self.round, j=self.quality())
