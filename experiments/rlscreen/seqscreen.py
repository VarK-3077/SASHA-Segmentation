# Sequential coarse screening with a learned stop. Nothing is embedded at level 3 for the whole slide:
# the tissue mask and the level-6 pass are the only global work. Every round the policy chooses to
# stop or to screen a batch of units at level 3 (batch size is its choice); the env takes them from a
# priority queue (level-6 p(tumour), with every unit next to a level-3 or level-2 tumour hit moved to
# the front), zooms the screened units the per-unit level-3 head calls tumour (p >= tz), and follows
# lesions across unit borders from the zoom outcomes, as the teacher does. The policy sees only
# slide-level evidence (the level-6 profile of what is left, the yield of its screens and zooms so far,
# how much tumour it has found, what it has spent, lambda), so its one decision per round is exactly
# "is more of this slide worth looking at". Ideal segmenter; reward phi(J) - lam * cost, phi the
# path-independent S^J potential, cost = seconds / seconds of segmenting the whole slide. Seconds
# use the per-unit costs measured in the real coarse runs.
# Compared in the same env with the fixed-threshold rule (screen every level-6 p >= ts), a non-learned
# evidence rule, and the hindsight-best stop (an upper bound).
# Usage: RLSCREEN_OOF=1 python -m experiments.rlscreen.seqscreen --tag NAME [--g_level 6]

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from experiments.rlscreen import paths as P
from experiments.rlscreen.rollout import load_tables

T0 = 0.56 + 0.12                          # tissue mask + level-6 pass (real coarse runs)
SCREEN_S, ZOOM_S, ROUND_S = 0.0022, 0.0097, 0.05
FULL_PER_UNIT = 8.44 / 1030
BATCHES = (0.01, 0.02, 0.04, 0.08)        # 'batch' actions: k > 0 screens BATCHES[k - 1] of the tissue units
THRESH = (0.95, 0.9, 0.8, 0.7, 0.5, 0.3)  # 'thresh' actions: 1 = follow lesions only, k > 1 screens every queued
                                          # unit with level-6 p >= THRESH[k - 2] (units next to hits always qualify)
ACTIONS = 'thresh'                        # set from --actions
R_MAX = 40
N_FEAT = 32
NB = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])


def phi(J, S):
    return J if S == 1 else (S ** J - 1) / np.log(S)


class Slide:
    """Static numpy view of one slide table for the sequential env."""

    def __init__(self, tab, metric):
        self.name, self.tumour = tab.name, bool(tab.has_tumour)
        self.tissue = tab.tissue.cpu().numpy()
        self.hu, self.wu = self.tissue.shape
        self.n = int(self.tissue.sum())
        self.pg = 1 - tab.p_g[..., 0]
        self.p3 = 1 - tab.p3[..., 0]
        self.p_core = np.where(tab.p3.argmax(-1) == 1, tab.p3[..., 1], 0.0)
        self.frac = tab.frac
        self.gt_px = float(tab.tumour_px)
        self.pos = tab.frac > 0
        self.edges = tab.edges > 0
        self.share = tab.share[metric].reshape(self.hu, self.wu)
        self.share_area = tab.share_area.reshape(self.hu, self.wu)
        self.lu_unit, self.lu_lesion, self.lesions = tab.lu_unit, tab.lu_lesion, tab.lesions


class SeqEpisode:
    def __init__(self, s, lam, tz=0.7, S=4.0, tc=0.99):
        self.s, self.lam, self.tz, self.S, self.tc = s, lam, tz, S, tc
        self.screened = np.zeros_like(s.tissue)
        self.zoomed = np.zeros_like(s.tissue)
        self.painted = np.zeros_like(s.tissue)    # routed units the head is sure are core: painted all tumour, not zoomed
        self.front = np.zeros_like(s.tissue)       # moved to the front of the screening queue
        self.follow = np.zeros_like(s.tissue)      # to zoom next round (border following)
        self.round, self.sec, self.done = 0, 0.0, False
        self.J = 0.0 if s.tumour else 1.0
        self.hist = []                              # per round: (screened, screen hits, zoomed, zoom positives)

    def full(self):
        return FULL_PER_UNIT * max(self.s.n, 1)

    def n_left(self):
        return int((self.s.tissue & ~self.screened & ~self.zoomed).sum())

    def queue(self):
        """Unscreened, unzoomed tissue units in screening order."""
        s = self.s
        free = s.tissue & ~self.screened & ~self.zoomed
        idx = np.flatnonzero(free)
        pri = s.pg.reshape(-1)[idx] + 2.0 * self.front.reshape(-1)[idx]
        return idx[np.argsort(-pri, kind='stable')]

    def _neighbours(self, m):
        out = np.zeros_like(m)
        for dr, dc in NB:
            sh = np.zeros_like(m)
            r0, r1 = max(dr, 0), self.s.hu + min(dr, 0)
            c0, c1 = max(dc, 0), self.s.wu + min(dc, 0)
            sh[r0:r1, c0:c1] = m[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
            out |= sh
        return out

    def _follow_from(self, zm):
        """Unzoomed units across a border where a zoomed unit's tumour touches it (N S W E strips)."""
        s, e = self.s, self.s.edges
        want = np.zeros_like(zm)
        want[:-1] |= (zm & e[..., 0])[1:]
        want[1:] |= (zm & e[..., 1])[:-1]
        want[:, :-1] |= (zm & e[..., 2])[:, 1:]
        want[:, 1:] |= (zm & e[..., 3])[:, :-1]
        return want & s.tissue & ~self.zoomed

    def step(self, n_screen):
        """Screens n_screen queued units (0 = only follow up). Returns the reward."""
        s = self.s
        sc = np.zeros_like(s.tissue)
        if n_screen > 0:
            q = self.queue()[:n_screen]
            sc.reshape(-1)[q] = True
        hits = sc & (s.p3 >= self.tz)
        paint = hits & (s.p_core >= self.tc) & ~self.zoomed
        zm = (hits | self.follow) & ~self.zoomed & ~paint
        ns, nz = int(sc.sum()), int(zm.sum())
        self.painted |= paint
        self.zoomed |= paint
        if ns + nz == 0:
            self.done = True
            return 0.0
        self.screened |= sc
        self.zoomed |= zm
        posz = zm & s.pos
        self.front |= self._neighbours(hits | posz | paint)
        self.follow = self._follow_from(posz)
        sec = ROUND_S + ns * SCREEN_S + nz * ZOOM_S
        self.sec += sec
        J0 = self.J
        if s.tumour:
            self.J = float(s.share[self.zoomed].sum())
        self.hist.append((ns, int(hits.sum()), nz, int(posz.sum())))
        self.round += 1
        if self.round >= R_MAX or (self.n_left() == 0 and not self.follow.any()):
            self.done = True
        return float(phi(self.J, self.S) - phi(J0, self.S) - self.lam * sec / self.full())

    def features(self):
        s = self.s
        n = max(s.n, 1)
        free = s.tissue & ~self.screened & ~self.zoomed
        pg = s.pg[free]
        hist = np.histogram(pg, bins=10, range=(0, 1))[0] / n
        tot = np.array(self.hist).sum(0) if self.hist else np.zeros(4)
        last = np.array(self.hist[-1]) if self.hist else np.zeros(4)
        since = 0
        for h in reversed(self.hist):
            if h[3] > 0:
                break
            since += 1
        nxt = np.sort(pg)[::-1][:max(2, int(0.02 * n))]
        scr_unz = self.screened & ~self.zoomed
        f = [np.log(self.lam) / 5, self.round / R_MAX, self.screened.sum() / n, self.zoomed.sum() / n,
             self.sec / self.full(), np.log1p(n) / 8, *hist,
             (pg >= 0.5).sum() / n, (pg >= 0.9).sum() / n, pg.max() if len(pg) else 0.0, nxt.mean() if len(nxt) else 0.0,
             tot[1] / max(tot[0], 1), last[1] / max(last[0], 1), tot[3] / max(tot[2], 1), last[3] / n,
             tot[3] / n, np.log1p(tot[3]) / 6, self.follow.sum() / n, np.log1p(self.follow.sum()) / 6,
             float(tot[3] > 0), since / R_MAX, s.p3[scr_unz].max() if scr_unz.any() else 0.0,
             (self.front & free).sum() / n]
        return np.asarray(f, np.float32)

    def summary(self):
        s = self.s
        area = float(s.share_area[self.zoomed].sum()) if s.tumour else 1.0
        seg = self.zoomed & ~self.painted   # ideal mask: exact on segmented units, all tumour on painted ones
        inter = s.frac[self.zoomed].sum()
        pred = s.frac[seg].sum() + self.painted.sum()
        dice = 2 * inter / (pred + s.gt_px) if s.gt_px > 0 else (1.0 if pred == 0 else 0.0)
        found = None
        if len(s.lesions):
            hit = self.zoomed.reshape(-1)[s.lu_unit]
            found = float(np.mean([hit[s.lu_lesion == l].any() for l in s.lesions]))
        return dict(tumour=s.tumour, dice=dice if s.tumour else 1.0, found=found, painted=self.painted.sum() / max(s.n, 1),
                    screened=self.screened.sum() / max(s.n, 1), zoomed=self.zoomed.sum() / max(s.n, 1),
                    sec=T0 + self.sec, rounds=self.round)


def n_actions():
    return 1 + (len(BATCHES) if ACTIONS == 'batch' else 1 + len(THRESH))


def n_above(ep, tau):
    s = ep.s
    free = s.tissue & ~ep.screened & ~ep.zoomed
    return int((free & ((s.pg >= tau) | ep.front)).sum())


def n_of(ep, action):
    """Units to screen for a (non-stop) action; 0 = follow lesions only."""
    if ACTIONS == 'batch':
        return min(ep.n_left(), max(2, int(round(BATCHES[action - 1] * ep.s.n))))
    return 0 if action == 1 else n_above(ep, THRESH[action - 2])


# ---------------------------------------------------------------------------------- actors

TC = 0.99   # core painting threshold (set from --tc)


def run(s, lam, actor, tz, S):
    ep = SeqEpisode(s, lam, tz, S, TC)
    while not ep.done:
        if not actor(ep):
            break
    return ep


def fixed_rule(ts):
    """Screen every unit with level-6 p >= ts in round 0, then follow lesions until done."""
    def act(ep):
        if ep.round == 0:
            n = int(((ep.s.pg >= ts) & ep.s.tissue).sum())
            if n == 0:
                return False
            ep.step(n)
        else:
            if not ep.follow.any():
                return False
            ep.step(0)
        return True
    return act


def evidence_rule(h, batch=0.03):
    """Screen in batches; after a batch whose zoom-worthy share is < h, finish following and stop."""
    state = {'stopping': False}
    def act(ep):
        if ep.round == 0:
            state['stopping'] = False
        if not state['stopping'] and ep.n_left() > 0:
            ep.step(min(ep.n_left(), max(2, int(round(batch * ep.s.n)))))
            ns, nh = ep.hist[-1][0], ep.hist[-1][1]
            if ns and nh / ns < h:
                state['stopping'] = True
            return True
        if ep.follow.any():
            ep.step(0)
            return True
        return False
    return act


def hindsight(s, lam, tz, S):
    """Best stop round for each constant batch size, knowing the outcome; the best over batch sizes."""
    best = None
    for k in range(1, len(BATCHES) + 1):
        ep = SeqEpisode(s, lam, tz, S, TC)
        obj, snaps = 0.0, [(0.0, ep.summary())]
        while not ep.done:
            obj += ep.step(min(ep.n_left(), max(2, int(round(BATCHES[k - 1] * ep.s.n)))) if ep.n_left() else 0)
            snaps.append((obj, ep.summary()))
        o, summ = max(snaps, key=lambda x: x[0])
        if best is None or o > best[0]:
            best = (o, summ)
    return best[1]


# ---------------------------------------------------------------------------------- policy

class StopPolicy(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(N_FEAT, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.pi = nn.Linear(h, n_actions())
        self.v = nn.Linear(h, 1)

    def forward(self, x):
        b = self.body(x)
        return self.pi(b), self.v(b)[..., 0]


def mask_of(ep):
    """Stop is always allowed; other actions only when they would do something."""
    m = np.ones(n_actions(), bool)
    if ACTIONS == 'batch':
        if ep.n_left() == 0 and not ep.follow.any():
            m[1:] = False
        return m
    m[1] = bool(ep.follow.any())
    for k, tau in enumerate(THRESH):
        m[2 + k] = n_above(ep, tau) > 0
    return m


def teacher_action(ep, ts):
    """The fixed rule as a 'thresh' action sequence: screen to ts, follow lesions, stop."""
    m = mask_of(ep)
    if ep.round == 0:
        k = 2 + THRESH.index(ts)
        return k if m[k] else 0
    return 1 if m[1] else 0


def teacher_ts(lam):
    """Warm-start teacher: the fixed rule, deeper when compute is cheap (lam is the compute price)."""
    return 0.5 if lam < 0.3 else 0.8 if lam < 3 else 0.9


def policy_actor(pol, greedy=True, rec=None, rng=None):
    def act(ep):
        x = torch.from_numpy(ep.features())[None]
        m = torch.from_numpy(mask_of(ep))[None]
        with torch.no_grad():
            logits, v = pol(x)
        logits = logits.masked_fill(~m, -1e9)
        d = torch.distributions.Categorical(logits=logits)
        a = int(logits.argmax(-1)) if greedy else int(d.sample())
        if a == 0:
            r = 0.0
            ep.done = True
        else:
            r = ep.step(n_of(ep, a) if (ACTIONS == 'thresh' or ep.n_left()) else 0)
        if rec is not None:
            rec.append(dict(x=x[0], m=m[0], a=a, lp=float(d.log_prob(torch.tensor([a]))), v=float(v), r=r))
        return a != 0
    return act


def gae(rs, vs, lam_gae=0.95):
    adv, last = np.zeros(len(rs)), 0.0
    for t in reversed(range(len(rs))):
        nv = vs[t + 1] if t + 1 < len(rs) else 0.0
        last = rs[t] + nv - vs[t] + lam_gae * last
        adv[t] = last
    return adv, adv + np.array(vs)


def bc(pol, slides, a, rng, log):
    """Clone the lambda-mapped fixed rule (cross-entropy on its actions, masked like the policy)."""
    X, M, A = [], [], []
    for i in range(a.bc_episodes):
        s = slides[rng.integers(len(slides))]
        lam = float(np.exp(rng.uniform(np.log(a.lam_min), np.log(a.lam_max))))
        ep = SeqEpisode(s, lam, a.tz, a.S, TC)
        ts = teacher_ts(lam)
        while not ep.done:
            k = teacher_action(ep, ts)
            X.append(torch.from_numpy(ep.features())); M.append(torch.from_numpy(mask_of(ep))); A.append(k)
            if k == 0:
                break
            ep.step(n_of(ep, k))
    X, M, A = torch.stack(X), torch.stack(M), torch.tensor(A)
    opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
    for e in range(a.bc_epochs):
        losses = []
        for idx in torch.randperm(len(A)).split(256):
            logits, _ = pol(X[idx])
            loss = nn.functional.cross_entropy(logits.masked_fill(~M[idx], -1e9), A[idx])
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss))
        log({'update': -1, 'bc_epoch': e, 'bc_loss': float(np.mean(losses)), 'decisions': len(A)})


def train(pol, slides, a, rng, log):
    opt = torch.optim.Adam(pol.parameters(), lr=a.lr)
    tum = [s for s in slides if s.tumour]
    nor = [s for s in slides if not s.tumour]
    for u in range(a.updates):
        t0 = time.time()
        batch, rets, stats = [], [], []
        picks = list(rng.choice(tum, a.episodes // 2)) + list(rng.choice(nor, a.episodes - a.episodes // 2))
        for s in picks:
            lam = float(np.exp(rng.uniform(np.log(a.lam_min), np.log(a.lam_max))))
            rec = []
            ep = run(s, lam, policy_actor(pol, greedy=False, rec=rec), a.tz, a.S)
            if not rec:
                continue
            adv, ret = gae([r['r'] for r in rec], [r['v'] for r in rec])
            for r, ad, rt in zip(rec, adv, ret):
                r['adv'], r['ret'] = ad, rt
            batch += rec
            rets.append(sum(r['r'] for r in rec))
            stats.append(ep.summary())
        X = torch.stack([b['x'] for b in batch]); M = torch.stack([b['m'] for b in batch])
        A = torch.tensor([b['a'] for b in batch]); LP = torch.tensor([b['lp'] for b in batch])
        ADV = torch.tensor([b['adv'] for b in batch], dtype=torch.float32); RET = torch.tensor([b['ret'] for b in batch], dtype=torch.float32)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        for _ in range(a.epochs):
            for idx in torch.randperm(len(batch)).split(256):
                logits, v = pol(X[idx])
                d = torch.distributions.Categorical(logits=logits.masked_fill(~M[idx], -1e9))
                ratio = torch.exp(d.log_prob(A[idx]) - LP[idx])
                loss = -torch.min(ratio * ADV[idx], ratio.clamp(1 - a.clip, 1 + a.clip) * ADV[idx]).mean() \
                    + 0.5 * ((v - RET[idx]) ** 2).mean() - a.ent * d.entropy().mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(pol.parameters(), 0.5); opt.step()
        tum_s = [x for x in stats if x['tumour']]
        log({'update': u, 'return': float(np.mean(rets)), 'sec': time.time() - t0, 'decisions': len(batch),
             'dice_tumour': float(np.mean([x['dice'] for x in tum_s])) if tum_s else None,
             'screened': float(np.mean([x['screened'] for x in stats])), 'zoomed': float(np.mean([x['zoomed'] for x in stats])),
             'est_s': float(np.mean([x['sec'] for x in stats]))})


# ---------------------------------------------------------------------------------- eval

def agg(res):
    tum = [r for r in res if r['tumour']]
    nor = [r for r in res if not r['tumour']]
    m = lambda k, rs: float(np.mean([r[k] for r in rs if r[k] is not None])) if rs else float('nan')
    return dict(dice=m('dice', tum), found=m('found', tum), screened=m('screened', res), zoomed=m('zoomed', res),
                sec=m('sec', res), sec_normal=m('sec', nor), sec_tumour=m('sec', tum), rounds=m('rounds', res))


def evaluate(slides, a, pol=None, lams=None):
    lams = lams or tuple(float(x) for x in np.geomspace(a.lam_min, a.lam_max, 6).round(3))
    rows = {}
    if pol is not None:
        for lam in lams:
            rows[f'RL stop, lam={lam:g}'] = agg([run(s, lam, policy_actor(pol), a.tz, a.S).summary() for s in slides])
    for ts in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        rows[f'fixed screen tau {ts:g}'] = agg([run(s, 1.0, fixed_rule(ts), a.tz, a.S).summary() for s in slides])
    for h in (0.02, 0.05, 0.1):
        rows[f'evidence rule h={h:g}'] = agg([run(s, 1.0, evidence_rule(h), a.tz, a.S).summary() for s in slides])
    for lam in lams:
        rows[f'hindsight stop, lam={lam:g}'] = agg([hindsight(s, lam, a.tz, a.S) for s in slides])
    return rows


def table(rows, title):
    keys = ('dice', 'found', 'screened', 'zoomed', 'sec', 'sec_normal', 'sec_tumour', 'rounds')
    lines = [f'# {title}', '', '| policy | ideal Dice (tumour) | lesions found | screened | zoomed | est. s | est. s normal | est. s tumour | rounds |',
             '|' + '---|' * (len(keys) + 1)]
    lines += [f'| {k} | ' + ' | '.join(f'{v[c]:.3f}' if c not in ('sec', 'sec_normal', 'sec_tumour') else f'{v[c]:.2f}' for c in keys) + ' |'
              for k, v in rows.items()]
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--g_level', type=int, default=6)
    ap.add_argument('--backbone', default='kaiko_vitl14')
    ap.add_argument('--metric', choices=('area', 'lesion', 'mix'), default='mix')
    ap.add_argument('--S', type=float, default=4.0)
    ap.add_argument('--tz', type=float, default=0.7)
    ap.add_argument('--tc', type=float, default=0.99, help='paint routed units with p(core) >= tc instead of zooming; > 1 = never')
    ap.add_argument('--lam_min', type=float, default=0.01)
    ap.add_argument('--lam_max', type=float, default=1.0)
    ap.add_argument('--updates', type=int, default=400)
    ap.add_argument('--episodes', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--ent', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--actions', choices=('thresh', 'batch'), default='thresh')
    ap.add_argument('--bc_episodes', type=int, default=4000, help='warm start by cloning the lambda-mapped fixed rule; 0 = none (thresh actions only)')
    ap.add_argument('--bc_epochs', type=int, default=10)
    ap.add_argument('--baselines_only', action='store_true')
    ap.add_argument('--split', default='test')
    ap.add_argument('--tag', default='seq')
    a = ap.parse_args()
    global TC, ACTIONS
    TC, ACTIONS = a.tc, a.actions
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    load = lambda sp, trusted=False: [Slide(t, a.metric) for t in load_tables(sp, 'coarse', a.g_level, a.backbone, 'cpu', trusted_only=trusted)]
    ev = load(a.split)
    name = f'{a.tag}_L{a.g_level}_{a.backbone}'
    if a.baselines_only:
        print(table(evaluate(ev, a), f'Sequential coarse screening baselines, {name}, {a.split}'), flush=True)
        return
    out = P.run_dir(a.tag)
    os.makedirs(out, exist_ok=True)
    json.dump({**vars(a), 'oof': os.environ.get('RLSCREEN_OOF') == '1'}, open(f'{out}/args.json', 'w'), indent=1)
    tr = load('train', trusted=P.NON_TRUSTED_SKIP)
    print(f'{a.tag}: {len(tr)} train slides ({sum(s.tumour for s in tr)} tumour), OOF heads: {os.environ.get("RLSCREEN_OOF") == "1"}', flush=True)
    logf = open(f'{out}/train.jsonl', 'w')

    def log(row):
        logf.write(json.dumps(row) + '\n'); logf.flush()
        if row['update'] % 10 == 0 or row['update'] < 0:
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

    pol = StopPolicy()
    if a.actions == 'thresh' and a.bc_episodes > 0:
        bc(pol, tr, a, rng, log)
        torch.save(pol.state_dict(), f'{out}/bc.pt')
        md = table(evaluate(ev, a, pol), f'Cloned fixed rule (before RL), {name}, {a.split}')
        open(f'{P.ROOT}/seq_{a.tag}_bc_{a.split}.md', 'w').write(md)
        print(md, flush=True)
    train(pol, tr, a, rng, log)
    torch.save(pol.state_dict(), f'{out}/final.pt')
    md = table(evaluate(ev, a, pol), f'Sequential coarse screening, {name}, {a.split} (ideal segmenter, est. A100 s)')
    open(f'{P.ROOT}/seq_{a.tag}_{a.split}.md', 'w').write(md)
    print(md, flush=True)


if __name__ == '__main__':
    main()
