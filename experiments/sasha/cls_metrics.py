"""Slide-level Accuracy / AUC / F1 on the CAMELYON16 test split from the prob_tumour field that time_infer.py
writes per slide, with the label csv of the SASHA repo. Prints one markdown table row per result directory."""

import csv
import glob
import json
import os
import sys

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(dirs):
    labels = {r['slide_id']: int(r['label']) for r in csv.DictReader(open(os.path.join(ROOT, 'dataset_csv/camelyon16/camelyon16.csv')))}
    print('| run | slides | Accuracy | AUC | F1 |\n|---|---|---|---|---|')
    for d in dirs:
        recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, 'test_*.json')))]
        y = [labels[r['slide']] for r in recs]
        p = [r['prob_tumour'] for r in recs]
        hat = [int(v > 0.5) for v in p]
        print(f'| {d} | {len(recs)} ({sum(y)} tumour) | {accuracy_score(y, hat):.3f} | {roc_auc_score(y, p):.3f} | {f1_score(y, hat):.3f} |')


if __name__ == '__main__':
    main(sys.argv[1:])
