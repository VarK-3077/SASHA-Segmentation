# How close the simulator is to the exact scorer: plays the supervised rule on every slide of a split in the
# simulator and prints its unit counts, Dice and FROC, to be read against the exact rows (train with
# out-of-fold maps: 404 L3, 113 L2, 5.0 painted, Dice 0.8072, FROC 0.8601; test: 462, 207, 13.0, 0.7962, 0.8673).
# Usage: python -m experiments.rlpolicy.check_sim --split train

import argparse

from experiments.rlpolicy import sim as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True)
    a = ap.parse_args()
    recs = [S.rule(s) for s in S.load(a.split)]
    print(a.split, len(recs), {k: round(v, 4) for k, v in S.summarise(recs).items()})


if __name__ == '__main__':
    main()
