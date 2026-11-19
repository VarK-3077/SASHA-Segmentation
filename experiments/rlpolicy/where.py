# Where the rlpolicy study reads and writes. Defaults are vivi's paths; RLPOLICY_ROOT moves the outputs
# (prince keeps them beside its rlcascade copy). Inputs are the rlcascade / rlscreen / lowsep caches.

import json
import os

from experiments.rlcascade import where as CW
from experiments.rlscreen import paths as SP

ROOT = os.environ.get('RLPOLICY_ROOT', '/var/tmp/venkatavks/rlpolicy')
FOLDS = os.environ.get('RLPOLICY_FOLDS', f'{ROOT}/folds5.json')
UNITS = SP.units_dir('kaiko_vitl14')
TOK3 = CW.TOK3
GT = f'{CW.ROOT}/gt'
SUB = f'{ROOT}/sub'
N_FOLDS = 5


def folds():
    return json.load(open(FOLDS))


def head_dir(tag):
    return f'{ROOT}/tok3/{tag}'
