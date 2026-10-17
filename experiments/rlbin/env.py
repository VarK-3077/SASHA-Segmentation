# One slide as an episode. The agent either zooms a tile (its sub-patches then carry the
# high-res scorer's prediction), marks a tile as core (all sub-patches positive), marks it
# normal (all negative; only in mode 'lr'), or stops. Two modes decide what an untouched tile
# means at the end: 'neg' = negative (the agent must act on every tumour tile), 'lr' = it keeps
# the low-res head's sub-patch mask (deployment semantics; the agent only overrides). Reward is
# the change in Dice of the composed mask, so the return is final Dice minus the starting Dice.

import numpy as np
import torch

from experiments.rlbin import paths as P

UNASSIGNED, ZOOMED, CORE, NORMAL = 0, 1, 2, 3
N_EXTRA, N_GLOB = 8, 3             # per tile: 4 low-res score stats + 4 bucket dims
N_ACT = {'neg': 2, 'lr': 3}        # per-tile actions: zoom, core (+ normal in 'lr')
N_STATUS = {'neg': 3, 'lr': 4}     # status one-hot width; dyn features = one-hot + 2 hr stats (keeps 'neg' checkpoints loadable)


def entropy(p):
    p = np.clip(p.astype(np.float32), 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p)).mean(1)


class Slide:
    """Static, per-slide tensors shared by the environment and the policy."""

    def __init__(self, name, c, device, state='none', mode='neg'):
        """state: 'none' (embedding + low-res score stats), 'pred' (+ predicted bucket
        probabilities from bucket_head.py), 'oracle' (+ one-hot ground-truth bucket).
        mode: 'neg' or 'lr', see the module docstring."""
        p_lr, p_hr, gt = c['p_lr'].astype(np.float32), c['p_hr'].astype(np.float32), c['gt']
        self.name, self.mode, self.n_act = name, mode, N_ACT[mode]
        self.n_status, self.n_dyn = N_STATUS[mode], N_STATUS[mode] + 2
        self.n, self.n_sub = gt.shape
        self.d_emb = c['lr'].shape[1]
        self.cap = int(np.floor(P.ZOOM_CAP * self.n))
        self.max_steps = int(np.ceil(P.MAX_STEPS_FRAC * self.n)) + 1
        self.bucket = c['bucket']
        self.p_lr, self.p_hr, self.gt = p_lr, p_hr, gt
        hr_pred, lr_pred = p_hr >= 0.5, p_lr >= 0.5
        self.hr_inter = (hr_pred & gt).sum(1)
        self.hr_pos = hr_pred.sum(1)
        self.lr_inter = (lr_pred & gt).sum(1) if mode == 'lr' else np.zeros(self.n, np.int64)
        self.lr_pos = lr_pred.sum(1) if mode == 'lr' else np.zeros(self.n, np.int64)
        self.core_inter = gt.sum(1)
        self.gt_pos = int(gt.sum())
        if state == 'oracle':
            bucket_feat = np.eye(4, dtype=np.float32)[c['bucket'].astype(np.int64)]
        elif state == 'pred':
            bucket_feat = c['p_bucket'].astype(np.float32)
        else:
            bucket_feat = np.zeros((self.n, 4), np.float32)
        static = np.concatenate([c['lr'].astype(np.float32),
                                 np.stack([p_lr.mean(1), p_lr.max(1), entropy(p_lr), c['nbr_p']], 1), bucket_feat], 1)
        self.static = torch.tensor(static, device=device)
        self.hr_stats = torch.tensor(np.stack([p_hr.mean(1), p_hr.max(1)], 1), device=device)
        self.device = device

    def dyn_features(self, status):
        """status: (..., N) int tensor -> (..., N, n_dyn): status one-hot and gated high-res stats."""
        onehot = torch.nn.functional.one_hot(status.long(), self.n_status).float()
        hr = self.hr_stats * (status == ZOOMED).unsqueeze(-1).float()
        return torch.cat([onehot, hr], -1)


class Env:
    def __init__(self, slide):
        self.s = slide

    def dice(self, inter, pos):
        d = pos + self.s.gt_pos
        return 1.0 if d == 0 else 2.0 * inter / d

    def reset(self):
        self.status = np.zeros(self.s.n, np.int8)
        self.inter, self.pos = int(self.s.lr_inter.sum()), int(self.s.lr_pos.sum())
        self.zooms = self.steps = 0
        self.cur = self.dice(self.inter, self.pos)
        self.done = False
        return self.obs()

    def obs(self):
        glob = np.array([self.zooms / self.s.n, self.zooms / max(self.s.cap, 1), self.steps / self.s.max_steps], np.float32)
        return self.status.copy(), glob

    def masks(self):
        free = self.status == UNASSIGNED
        return free & (self.zooms < self.s.cap), free

    def step(self, a):
        n, s = self.s.n, self.s
        if a == s.n_act * n:
            self.done = True
            return 0.0
        i, kind = a % n, a // n
        assert self.status[i] == UNASSIGNED
        self.inter -= s.lr_inter[i]; self.pos -= s.lr_pos[i]     # the tile's low-res mask is replaced
        if kind == 0:
            self.status[i] = ZOOMED
            self.inter += s.hr_inter[i]; self.pos += s.hr_pos[i]
            self.zooms += 1
        elif kind == 1:
            self.status[i] = CORE
            self.inter += s.core_inter[i]; self.pos += s.n_sub
        else:
            self.status[i] = NORMAL
        self.steps += 1
        new = self.dice(self.inter, self.pos)
        r = new - self.cur
        self.cur = new
        if self.steps >= self.s.max_steps or not (self.status == UNASSIGNED).any():
            self.done = True
        return float(r)
