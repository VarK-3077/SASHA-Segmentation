# Draws the stratified crop sample each level is extracted and probed on. Per slide and
# bucket the counts are capped so that the finer levels (4x more crops per level) stay
# comparable in size, and half of the normals on tumour slides are drawn near lesions as
# hard negatives. Normals from non-exhaustively annotated training slides are discarded.

import json
import os

import numpy as np

from experiments.patch_common.ncrf_non_exhaustive import NON_EXHAUSTIVE_TUMOR_SLIDES
from experiments.sep import buckets as B
from experiments.sep import paths as P

CAPS = {B.NORMAL: 150, B.CORE: 150, B.BOUNDARY: 300, B.SMALL: 400}
NEAR_NORMAL_CAP = 150
SEED = 0


def split_of():
    d = json.load(open(P.SPLIT_JSON))
    return {s: k.replace('_names', '') for k in d for s in d[k]}


def pick(idx, cap, rng):
    return idx if len(idx) <= cap else rng.choice(idx, cap, replace=False)


def sample_slide(z, L, name, rng):
    bucket = z[f'L{L}_bucket']
    near = z[f'L{L}_near']
    keep = []
    for b, cap in CAPS.items():
        idx = np.flatnonzero(bucket == b)
        if b == B.NORMAL:
            if name in NON_EXHAUSTIVE_TUMOR_SLIDES:
                continue
            near_idx = idx[near[idx]]
            far_idx = idx[~near[idx]]
            keep.append(pick(near_idx, NEAR_NORMAL_CAP, rng))
            keep.append(pick(far_idx, cap, rng))
        else:
            keep.append(pick(idx, cap, rng))
    return np.sort(np.concatenate(keep)).astype(np.int64)


def main():
    split = split_of()
    names = sorted(f[:-4] for f in os.listdir(P.LABEL_DIR) if f.endswith('.npz'))
    skipped = [n for n in names if n not in split]
    names = [n for n in names if n in split]
    print(f'{len(names)} slides; not in split, skipped: {skipped}', flush=True)
    for L in P.LEVELS:
        rng = np.random.default_rng(SEED + L)
        cols = {k: [] for k in ('slide', 'split', 'x', 'y', 'frac', 'bucket', 'major_um', 'near')}
        for name in names:
            z = np.load(os.path.join(P.LABEL_DIR, name + '.npz'))
            idx = sample_slide(z, L, name, rng)
            cols['slide'].append(np.full(len(idx), name))
            cols['split'].append(np.full(len(idx), split[name]))
            for c in ('x', 'y', 'frac', 'bucket', 'major_um', 'near'):
                cols[c].append(z[f'L{L}_{c}'][idx])
        cols = {k: np.concatenate(v) for k, v in cols.items()}
        cols['size_bin'] = B.size_bin(cols['major_um'])
        np.savez(P.crops_npz(L), **cols)
        counts = {s: np.bincount(cols['bucket'][cols['split'] == s], minlength=4).tolist()
                  for s in ('train', 'val', 'test')}
        print(f'L{L}: {len(cols["x"])} crops; per split [normal, core, boundary, small] = {counts}', flush=True)


if __name__ == '__main__':
    main()
