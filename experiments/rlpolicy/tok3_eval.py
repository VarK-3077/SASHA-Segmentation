# Out-of-fold check of the token-level L3 head against the supervised unit head it would replace, on the
# trusted train slides only (each slide scored by heads that did not see it). Unit level: AUC and average
# precision for "holds tumour" over all tissue units and over the units the L6 screen passes. Routing level:
# with zoom = L6-screened and score >= t, the zooms per normal and per tumour slide against the share of
# non-ITC lesions with at least one zoomed unit and the share of small-bucket units zoomed; each new score is
# read at the supervised head's own operating point (same lesion recall, and same zooms).
# Usage: python -m experiments.rlpolicy.tok3_eval --tag t0

import argparse
import json

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.rlpolicy import where as W
from experiments.rlpolicy.tok3_data import train_names

TAU, TZ = 0.6, 0.5


def load(name, tag):
    z = np.load(f'{W.UNITS}/{name}.npz')
    oof = np.load(f'{W.UNITS}/oof/{name}.npz')
    p = np.load(f'{W.head_dir(tag)}/pred/{name}.npz')
    hu, wu = z['tissue'].shape
    rc = p['rc']
    ok = (rc[:, 0] < hu) & (rc[:, 1] < wu)
    r, c = rc[ok, 0], rc[ok, 1]
    tp = p['tokp'][ok].reshape(ok.sum(), -1).astype(np.float32) / 255
    g = lambda v: (lambda a: (a.__setitem__((r, c), v), a)[1])(np.zeros((hu, wu), np.float32))
    top = np.sort(tp, 1)[:, ::-1]
    scores = dict(base=1 - oof['p3_unit'][..., 0].astype(np.float32), unit=g(1 / (1 + np.exp(-p['unit'][ok, 0]))), tokmax=g(top[:, 0]),
                  top4=g(top[:, :4].mean(1)), top16=g(top[:, :16].mean(1)))
    scores['mix'] = 0.5 * (scores['unit'] + scores['top4'])
    has = np.zeros((hu, wu), bool)
    has[r, c] = True
    keep = ~z['lesion_itc'][z['lu_lesion']] if len(z['lu_lesion']) else np.zeros(0, bool)
    return dict(name=name, m=z['tissue'] & has, pos=z['frac'] > 0, small=z['bucket'] == 3, pg=1 - oof['p6'][..., 0].astype(np.float32),
                lu_unit=z['lu_unit'][keep], lu_lesion=z['lu_lesion'][keep], scores=scores, tumour=bool((z['frac'] > 0).any()))


def route(s, score, t):
    return s['m'] & (s['pg'] >= TAU) & (score >= t)


def point(slides, key, t):
    zn, zt, found, total, sm, smt, tu, tut = [], [], 0, 0, 0, 0, 0, 0
    for s in slides:
        zm = route(s, s['scores'][key], t)
        (zt if s['tumour'] else zn).append(zm.sum())
        if len(s['lu_unit']):
            hit = zm.reshape(-1)[s['lu_unit']]
            les = np.unique(s['lu_lesion'])
            found += len(np.unique(s['lu_lesion'][hit])); total += len(les)
        sm += (zm & s['small'] & s['m']).sum(); smt += (s['small'] & s['m']).sum()
        tu += (zm & s['pos'] & s['m']).sum(); tut += (s['pos'] & s['m']).sum()
    return dict(t=float(t), zoom_normal=float(np.mean(zn)), zoom_tumour=float(np.mean(zt)), zoom=float(np.mean(zn + zt)), lesion=found / max(total, 1),
                small=sm / max(smt, 1), units=tu / max(tut, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    slides = [load(n, a.tag) for n in train_names()]
    keys = list(slides[0]['scores'])
    out = {}
    print('unit level (out-of-fold train): AUC / AP all tissue units | AUC / AP L6-screened units | AP small-vs-normal screened')
    for k in keys:
        y = np.concatenate([s['pos'][s['m']] for s in slides]); v = np.concatenate([s['scores'][k][s['m']] for s in slides])
        sc = np.concatenate([(s['pg'] >= TAU)[s['m']] for s in slides]); sm = np.concatenate([s['small'][s['m']] for s in slides])
        sel = sc & (sm | ~y)
        out[k] = dict(auc=roc_auc_score(y, v), ap=average_precision_score(y, v), auc_s=roc_auc_score(y[sc], v[sc]), ap_s=average_precision_score(y[sc], v[sc]),
                      ap_small=average_precision_score(sm[sel], v[sel]))
        print(f"{k:8s} {out[k]['auc']:.4f} / {out[k]['ap']:.4f} | {out[k]['auc_s']:.4f} / {out[k]['ap_s']:.4f} | {out[k]['ap_small']:.4f}")
    base = point(slides, 'base', TZ)
    print('\nrouting, supervised head at 0.5:', json.dumps({k: round(v, 4) for k, v in base.items()}))
    grid = np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), [0.995, 0.999, 0.002, 0.005]]))
    for k in keys:
        pts = [point(slides, k, t) for t in grid]
        same_rec = min((p for p in pts if p['lesion'] >= base['lesion'] and p['small'] >= base['small']), key=lambda p: p['zoom'], default=None)
        same_zoom = max((p for p in pts if p['zoom'] <= base['zoom']), key=lambda p: (p['lesion'], p['small']), default=None)
        out[k].update(curve=pts, same_rec=same_rec, same_zoom=same_zoom)
        f = lambda p: 'none' if p is None else f"t {p['t']:.3f} zoom {p['zoom']:.1f} (normal {p['zoom_normal']:.1f}, tumour {p['zoom_tumour']:.1f}) lesion {p['lesion']:.4f} small {p['small']:.4f} units {p['units']:.4f}"
        print(f'{k:8s} same recall -> {f(same_rec)}\n{"":8s} same zooms  -> {f(same_zoom)}')
    json.dump(dict(base=base, scores=out), open(f'{W.head_dir(a.tag)}/oof_eval.json', 'w'))


if __name__ == '__main__':
    main()
