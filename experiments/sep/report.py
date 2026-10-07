# Turns the probe jsons into the markdown tables of the separability study and prints
# them. Numbers are mean over seeds; the spread column is the std over seeds.

import csv
import glob
import json
import os

import numpy as np

from experiments.sep import buckets as B
from experiments.sep import paths as P


def agg(runs, get):
    v = [get(r) for r in runs]
    v = [x for x in v if x is not None]
    return (np.mean(v), np.std(v)) if v else (np.nan, 0)


def cell(ms):
    return f'{ms[0]:.3f} ± {ms[1]:.3f}'


def table(rows, header):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def main():
    res = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(P.PROBE_DIR, 'L*_*.json')))]
    bbs = sorted({r['backbone'] for r in res})
    lines = ['# Separability study', '',
             'Unit: 256 px crop at level L (footprint = 256·2^L px at level 0). Test = 129 CAMELYON16 test slides,',
             'stratified crop sample. Probe selected on val macro-recall, 3 seeds.', '']
    for kind in ('mlp', 'linear'):
        lines += [f'## {kind}: macro-recall over the four buckets (argmax)', '']
        rows = []
        for L in P.LEVELS:
            row = [f'L{L} ({P.footprint_l0(L)} px)']
            for bb in bbs:
                r = next((r for r in res if r['level'] == L and r['backbone'] == bb), None)
                row.append(cell(agg(r['models'][kind], lambda x: x['macro_recall'])) if r else '-')
            rows.append(row)
        lines += [table(rows, ['level'] + bbs), '']
        for name in ('small', 'boundary', 'core'):
            lines += [f'## {kind}: recall of **{name}** crops at 99% specificity on normals', '']
            rows = []
            for L in P.LEVELS:
                row = [f'L{L}']
                for bb in bbs:
                    r = next((r for r in res if r['level'] == L and r['backbone'] == bb), None)
                    row.append(cell(agg(r['models'][kind], lambda x: x['at_spec']['0.99']['recall'][name])) if r else '-')
                rows.append(row)
            lines += [table(rows, ['level'] + bbs), '']
        lines += [f'## {kind}: recall by lesion size at 99% specificity (ITC <0.2 mm, micro 0.2-2 mm, macro >2 mm)', '']
        rows = []
        for L in P.LEVELS:
            for bb in bbs:
                r = next((r for r in res if r['level'] == L and r['backbone'] == bb), None)
                if r:
                    rows.append([f'L{L}', bb] + [cell(agg(r['models'][kind], lambda x, s=s: x['at_spec']['0.99']['recall'][f'size_{s}']))
                                                 for s in B.SIZE_NAMES])
        lines += [table(rows, ['level', 'backbone'] + list(B.SIZE_NAMES)), '']
    lines += ['## test crop counts per level [normal, core, boundary, small]', '']
    lines += [f'- L{r["level"]}: {r["counts_test"]} (train n={r["n"]["train"]})' for r in res if r['backbone'] == bbs[0]]
    les = list(csv.DictReader(open(os.path.join(P.ROOT, 'lesions.csv'))))
    major = np.array([float(r['major_um']) for r in les])
    lines += ['', f'## lesions in the xmls: {len(les)} connected components over {len(set(r["slide"] for r in les))} slides', '',
              f'- ITC (<0.2 mm): {(major < 200).sum()}', f'- micro (0.2-2 mm): {((major >= 200) & (major < 2000)).sum()}',
              f'- macro (>2 mm): {(major >= 2000).sum()}']
    text = '\n'.join(lines)
    open(os.path.join(P.ROOT, 'report.md'), 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
