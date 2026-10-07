# Bucket rules for the separability study. A crop is named by what a selector would have to
# do with it: leave it (normal), mark it without looking closer (core), look closer because a
# big lesion's edge runs through it (boundary), or look closer because a lesion smaller than
# the crop sits inside it (small).

import numpy as np

NORMAL, CORE, BOUNDARY, SMALL = 0, 1, 2, 3
NAMES = ('normal', 'core', 'boundary', 'small')
CORE_THR = 0.9

# clinical lesion classes by major axis: ITC < 0.2 mm, micro 0.2-2 mm, macro > 2 mm
SIZE_NAMES = ('itc', 'micro', 'macro')
SIZE_EDGES_UM = (200.0, 2000.0)


def assign(frac, max_lesion_area_l0, footprint_l0):
    b = np.full(frac.shape, NORMAL, dtype=np.int8)
    touched = frac > 0
    b[touched & (frac >= CORE_THR)] = CORE
    partial = touched & (frac < CORE_THR)
    big = max_lesion_area_l0 > float(footprint_l0) ** 2
    b[partial & big] = BOUNDARY
    b[partial & ~big] = SMALL
    return b


def size_bin(major_um):
    s = np.full(major_um.shape, -1, dtype=np.int8)
    s[major_um > 0] = 0
    s[major_um >= SIZE_EDGES_UM[0]] = 1
    s[major_um >= SIZE_EDGES_UM[1]] = 2
    return s
