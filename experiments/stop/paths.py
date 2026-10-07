# Locations for the stopping-ceiling study. Rollouts and heads live under /data2.

import os

ROOT = '/data2/venkatavks/stop'
ROLL_DIR = f'{ROOT}/rollouts'
HEADS = f'{ROOT}/heads.pt'
HR_DIRS = ('/data1/venkatavks/seg_outputs/hr_train_feats', '/data1/venkatavks/seg_outputs/hr_test_feats')

TILE_L0 = 2048
SUB_POS_THR = 0.5          # a 512 px sub-patch is tumour if at least half of it is
ORDERS = ('random', 'greedy', 'uncertain', 'adaptive', 'oracle')
TARGET_FRACS = (0.1, 0.2, 0.4, 0.6)

os.makedirs(ROLL_DIR, exist_ok=True)
