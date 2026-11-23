"""Supervised tile-picking rules in place of SASHA's RL agent, for the timed pipeline (time_infer.py --policy).
The rules are patch_cls/acq's (random: uniform over unvisited tiles; greedy: highest Model-2 P(tumour); uncertain:
Model-2 P(tumour) closest to 0.5), Model-2 recomputed on the current state at every step, ties broken by a per-slide
seed. Only the choice of the next tile changes: reads, ViT, TSU, budget and the HAFED slide classifier stay SASHA's."""

import math

import numpy as np
import torch

from patch_cls.acq import m2 as M2
from patch_cls.acq.policies import POLICIES, SlideCtx, slide_seed


class Rule:
    def __init__(self, name, device, seed=0):
        self.select, self.needs_p = POLICIES[name]
        self.predict = M2.load(device=device) if self.needs_p else None
        self.seed, self.ctx = seed, None

    def start(self, slide, coords, frac_visit):
        n = len(coords)
        self.ctx = SlideCtx(coords=np.asarray(coords), budget=math.ceil(n * frac_visit), rng=np.random.default_rng(slide_seed(slide, self.seed)),
                            gt_frac=np.zeros(n, np.float32))

    def pick(self, state):
        """state (1, N, 384) on the GPU -> index of the next tile; records it as visited."""
        p = self.predict(state[0].to(torch.float16).cpu().numpy(), self.ctx.coords, self.ctx.visited) if self.needs_p else None
        a = self.select(self.ctx, p, state, None)
        self.ctx.mark(a)
        return a
