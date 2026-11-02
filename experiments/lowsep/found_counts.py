# Lesions found by a post-processed mask, per size bin: a lesion counts as found when the mask
# kept by one clean_sweep variant (threshold, optional seed) overlaps its tumour. Uses
# clean_sweep's own per-slide scoring, which computes per-lesion Dice but saves only averages.
# Usage: python -m experiments.lowsep.found_counts --maps DIR --variant t0.4_seed0.9 --out FILE

import argparse
import glob
import json
import os
import re
from functools import partial
from multiprocessing import Pool

from experiments.pfa_scannet.clean_sweep import BINS, run_one


def parse(v):
    m = re.fullmatch(r't([\d.]+)(?:_seed([\d.]+))?', v)
    return float(m.group(1)), (float(m.group(2)) if m.group(2) else None), 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--maps', required=True)
    ap.add_argument('--variant', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    names = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(a.maps, 'test_*.npz')))
    with Pool(a.workers) as pool:
        res = dict(pool.imap_unordered(partial(run_one, maps=a.maps, var=[parse(a.variant)]), names))
    lesions = [l for n in names for l in res[n][a.variant].get('lesions', [])]
    out = {'variant': a.variant, 'maps': a.maps, 'slides': len(names)}
    for lo, hi in BINS:
        L = [d for u, d in lesions if lo <= u < hi]
        out[f'{lo}-{hi}'] = [sum(d > 0 for d in L), len(L)]
    out['total'] = [sum(d > 0 for _, d in lesions), len(lesions)]
    json.dump(out, open(a.out, 'w'), indent=1)
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
