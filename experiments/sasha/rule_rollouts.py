"""patch_cls/acq/run_acq.py with a different random seed for the rules only. run_acq's --seed also selects the data
split (split_<seed>.json) the released SASHA models belong to, so it stays 4; --rule_seed only moves the per-slide
random draws and tie-breaks (slide_seed). Usage: python -m experiments.sasha.rule_rollouts --rule_seed 1 -- <run_acq args>"""

import sys

from patch_cls.acq import run_acq


def main():
    i = sys.argv.index('--rule_seed')
    rs = int(sys.argv[i + 1])
    rest = sys.argv[i + 2:]
    if rest and rest[0] == '--':
        rest = rest[1:]
    orig = run_acq.slide_seed
    run_acq.slide_seed = lambda name, base: orig(name, base + 1000 * rs)
    sys.argv = [sys.argv[0]] + rest
    run_acq.main()


if __name__ == '__main__':
    main()
