"""Slide-classification table for timed runs of time_infer.py (one json per slide with prob_tumour and times), with
every metric the SASHA paper reports for CAMELYON16: Accuracy, AUC, F1 (Table 1), inference time per slide with its
stage breakdown (Fig. 3c, Table 13), WSI compressibility = W x H x 3 / size of the representation the classifier
reads (A.4: N x d for a state of N tiles), and the expected calibration error (A.4: 10 equal bins of the predicted
class's probability, |accuracy - confidence| weighted by bin share). The predicted class uses the run's own
threshold when its jsons carry one (tile classifiers: chosen on validation), 0.5 otherwise (the softmax argmax);
ECE always uses the argmax, as the paper defines it. Differences to the first run: paired bootstrap over slides.
Usage: python -m experiments.sasha.cls_report --runs NAME=DIR ... --out FILE.md"""

import argparse
import csv
import glob
import json
import os

import numpy as np
import openslide
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE = '/data1/namanm/dataset/camelyon16/images/all'
N_BOOT = 2000


def ece(y, p, bins=10):
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def metrics(y, p, thr):
    hat = (p >= thr).astype(int)
    return dict(acc=accuracy_score(y, hat), auc=roc_auc_score(y, p), f1=f1_score(y, hat), ece=ece(y, p))


def load(d, names=None):
    recs = {json.load(open(f))['slide']: json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, 'test_*.json')))}
    return {n: recs[n] for n in (names or sorted(recs))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    labels = {r['slide_id']: int(r['label']) for r in csv.DictReader(open(os.path.join(ROOT, 'dataset_csv/camelyon16/camelyon16.csv')))}
    runs = dict(r.split('=', 1) for r in a.runs)
    first = next(iter(runs))
    names = sorted(set.intersection(*(set(load(d)) for d in runs.values())))
    data = {k: load(d, names) for k, d in runs.items()}
    dims = {n: openslide.OpenSlide(os.path.join(SOURCE, n + '.tif')).dimensions for n in names}
    y = np.array([labels[n] for n in names])
    L = [f'## Slide classification, {len(names)} CAMELYON16 test slides ({int(y.sum())} tumour); seconds = measured wall per slide\n',
         '| run | Accuracy | AUC | F1 | ECE | threshold | s/slide mean ± sd | median | tiles read at high res | WSI compressibility |', '|---|---|---|---|---|---|---|---|---|---|']
    P, T = {}, {}
    for k, recs in data.items():
        r = [recs[n] for n in names]
        P[k] = np.array([x['prob_tumour'] for x in r])
        T[k] = float(r[0].get('threshold', 0.5))
        m = metrics(y, P[k], T[k])
        w = np.array([x['times']['total_wall'] for x in r])
        d = 768 if r[0].get('kind') == 'juiced' else 384
        comp = np.mean([dims[n][0] * dims[n][1] * 3 / (x['n_tiles'] * d) for n, x in zip(names, r)])
        hr = np.mean([x.get('n_visited', 0) for x in r]) * 16
        L.append(f"| {k} | {m['acc']:.3f} | {m['auc']:.3f} | {m['f1']:.3f} | {m['ece']:.3f} | {T[k]:.3f} | {w.mean():.2f} ± {w.std(ddof=1):.2f} | "
                 f"{np.median(w):.2f} | {hr:.0f} | {comp:.2e} |")
    L += ['', '| run | stage means, s/slide |', '|---|---|']
    for k, recs in data.items():
        keys = [s for s in recs[names[0]]['times'] if s != 'total_wall']
        L.append(f"| {k} | " + ', '.join(f"{s} {np.mean([recs[n]['times'][s] for n in names]):.2f}" for s in keys) + ' |')
    L += ['', f'Paired bootstrap over slides ({N_BOOT} resamples), run minus {first}: mean [95 % interval]', '',
          '| run | ΔAccuracy | ΔAUC | ΔF1 |', '|---|---|---|---|']
    rng = np.random.default_rng(0)
    idx = [rng.integers(len(names), size=len(names)) for _ in range(N_BOOT)]
    idx = [i for i in idx if 0 < y[i].sum() < len(i)]
    for k in list(data)[1:]:
        d = {m: [] for m in ('acc', 'auc', 'f1')}
        for i in idx:
            a_, b_ = metrics(y[i], P[k][i], T[k]), metrics(y[i], P[first][i], T[first])
            for m in d:
                d[m].append(a_[m] - b_[m])
        L.append(f'| {k} | ' + ' | '.join(f'{np.mean(v):+.3f} [{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]' for v in d.values()) + ' |')
    text = '\n'.join(L) + '\n'
    open(a.out, 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
