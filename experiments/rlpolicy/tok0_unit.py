# The token head at level 0 (frozen ViT-L + the PFE head trained on level-0 crops) as the segmenter of one
# 2048-px cascade unit. A unit is 4 x 4 of the head's own 512-px crops on the dense run's (0, 0)-anchored grid,
# so its map is the dense level-0 map on that unit: each crop with Otsu tissue is read as its four 256-px
# children (resized to 224), the head gives 32 x 32 tokens of 16 px, and the unit's 128 x 128 token map is
# averaged 2 x 2 into the 32-px cells every scoring tool reads (as tokhead/infer.py does at TOK_LEVEL 0). Units
# are read in a process pool; every call adds read-wait / GPU seconds to a timer dict (cascade_pfa's layout).

import collections
import time

import numpy as np
import torch
from PIL import Image

from experiments.lowsep.cascade_pipe import _slide
from experiments.pfa_scannet.data import read_rgb

UNIT, CROP0, CHILD0, INPUT, TOK = 2048, 512, 256, 224, 32
N = UNIT // CROP0            # crops per unit side
TILE = UNIT // 32            # unit map side in 32-px cells
C5 = CROP0 // 32             # crop side in level-5 px
THR = 0.7                    # the level-0 head's validation-chosen threshold (tokhead_l0 chain)
CKPT = '/data2/venkatavks/pfa_ckpt/named/tokhead_kaiko_vitl14_stageA_input_l0.pt'
IN_FLIGHT = 16


def read_unit0(task):
    """-> gy, gx, uint8 (n, 4, 3, 224, 224): the four children of each kept crop of the unit, crops row-major."""
    path, gy, gx, keep = task
    img = read_rgb(_slide(path), gx * UNIT, gy * UNIT, UNIT, level=0)
    out = np.empty((sum(keep), 4, 3, INPUT, INPUT), np.uint8)
    k = 0
    for i in range(N * N):
        if not keep[i]:
            continue
        r, c = divmod(i, N)
        for j in range(4):
            a, b = divmod(j, 2)
            child = img[r * CROP0 + a * CHILD0:r * CROP0 + (a + 1) * CHILD0, c * CROP0 + b * CHILD0:c * CROP0 + (b + 1) * CHILD0]
            out[k, j] = np.asarray(Image.fromarray(np.ascontiguousarray(child)).resize((INPUT, INPUT), Image.BILINEAR)).transpose(2, 0, 1)
        k += 1
    return gy, gx, out


class UnitSegmenter:
    def __init__(self, net, device):
        self.net, self.dev, self.t5 = net, device, None     # t5: the slide's level-5 tissue mask, set per slide
        self.t = collections.defaultdict(float)

    def warmup(self, n=3):
        with torch.no_grad():
            for _ in range(n):
                self.net(torch.zeros(N * N, 4, 3, INPUT, INPUT, dtype=torch.uint8, device=self.dev))
        torch.cuda.synchronize()

    def tissue_crops(self, gy, gx):
        b = self.t5[gy * TILE:(gy + 1) * TILE, gx * TILE:(gx + 1) * TILE]
        b = np.pad(b, ((0, TILE - b.shape[0]), (0, TILE - b.shape[1])))
        return tuple(bool(v) for v in b.reshape(N, C5, N, C5).any((1, 3)).ravel())

    def run(self, pool, path, units):
        """Segments units [(gy, gx)] in order; returns {(gy, gx): 64 x 64 float32 map of 32-px cells}."""
        out, todo, live = {}, list(units), collections.deque()
        t, ev = self.t, []
        while todo or live:
            while todo and len(live) < IN_FLIGHT:
                gy, gx = todo.pop(0)
                keep = self.tissue_crops(gy, gx)
                live.append((keep, pool.apply_async(read_unit0, ((path, gy, gx, keep),))))
            s0 = time.perf_counter()
            keep, fut = live.popleft()
            gy, gx, x = fut.get()
            s1 = time.perf_counter()
            m = np.zeros((N * TOK, N * TOK), np.float32)
            if len(x):
                e = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                e[0].record()
                with torch.no_grad():
                    p = torch.sigmoid(self.net(torch.from_numpy(x).to(self.dev)).float())
                e[1].record()
                ev.append(e)
                for q, i in zip(p.cpu().numpy(), np.flatnonzero(keep)):
                    r, c = divmod(int(i), N)
                    m[r * TOK:(r + 1) * TOK, c * TOK:(c + 1) * TOK] = q
            out[(gy, gx)] = m.reshape(TILE, 2, TILE, 2).mean((1, 3))
            s2 = time.perf_counter()
            t['l0_wait'] += s1 - s0
            t['pfa_loop'] += s2 - s1
            t['rois'] += 1
            t['crops'] += len(x)
        t['pfa_gpu'] += sum(a.elapsed_time(b) for a, b in ev) / 1e3
        return out
