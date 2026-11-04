# Where the RL screening study keeps its per-slide unit tables, runs and reports. The unit grid,
# labels and screen tokens come from experiments/lowsep (same 2048 px segmenter units).

import os

ROOT = '/data2/venkatavks/rlscreen'
G_LEVELS = (5, 6)            # cheap global views for the coarse-to-fine agent
R_MAX = 6                    # rounds per episode at most
NON_TRUSTED_SKIP = True      # drop non-exhaustively annotated training slides from RL episodes


def units_dir(backbone):
    return f'{ROOT}/units_{backbone}'


def run_dir(tag):
    return f'{ROOT}/runs/{tag}'
