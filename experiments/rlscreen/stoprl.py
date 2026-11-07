# Learned stopping and core selection, offline, for both supervised rankings.
#   full    the level-3 pass scores every unit; units are taken from the top of that ranking in batches
#   coarse  only the level-6 pass is global; units are screened at level 3 down the level-6 ranking and
#           routed if the per-unit level-3 head says p >= tz
# Each round the policy chooses: stop, follow lesions only, or how far to go (batch size on full,
# level-6 threshold on coarse) together with a core threshold tc. Routed units with p(core) >= tc are
# painted all tumour instead of zoomed; the rest are zoomed (ideal segmenter). Borders are followed from
# zoom outcomes; a painted unit sends its four neighbours to zoom, because no segmenter ran on it.
# Reward: phi(J) - lam * seconds / seconds of segmenting the slide. J is recall minus a false-positive
# term, fp / (fp + gt + 1) in unit areas, so painting non-tumour costs quality on any slide, including
# slides with no tumour. The policy starts as a clone of a rule (hit-rate on full, fixed threshold on
# coarse), then PPO, trained on out-of-fold head outputs.
# Settings are chosen on the out-of-fold train slides by a compute price (best Dice - price * seconds);
# test is read only at the chosen row.
# Usage: RLSCREEN_OOF=1 python -m experiments.rlscreen.stoprl --design full --tag NAME [--tcs 2 0.99 0.9]

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from experiments.rlscreen import paths as P
from experiments.rlscreen.rollout import load_tables
from experiments.rlscreen.seqscreen import FULL_PER_UNIT, NB, ROUND_S, SCREEN_S, ZOOM_S, Slide, gae, phi

GLOBAL_S = {'full': 0.56 + 1.52, 'coarse': 0.56 + 0.12}   # real runs: tissue + whole-slide pass
BATCHES = (0.01, 0.02, 0.04, 0.08)
THRESH = (0.95, 0.9, 0.8, 0.7, 0.5, 0.3)
R_MAX = 40
FLOOR = 0.02         # full: units below this p are never taken from the ranking
N_FEAT = 34
CFG = dict(design='full', tz=0.7, S=4.0, fp=1.0, tcs=(0.99,), acts=[])


def build_actions():
    levels = [('batch', b) for b in BATCHES] if CFG['design'] == 'full' else [('thresh', t) for t in THRESH]
    CFG['acts'] = [('stop', None, None), ('follow', None, None)] + [(k, v, tc) for k, v in levels for tc in CFG['tcs']]


class Ep:
    def __init__(self, s, lam):
        self.s, self.lam = s, lam
        self.full_design = CFG['design'] == 'full'
        self.screened = s.tissue.copy() if self.full_design else np.zeros_like(s.tissue)
        self.zoomed = np.zeros_like(s.tissue)     # recovered units, segmented or painted
        self.painted = np.zeros_like(s.tissue)
        self.front = np.zeros_like(s.tissue)
        self.follow = np.zeros_like(s.tissue)
        self.round, self.sec, self.done, self.stopping = 0, 0.0, False, False
        self.J = 0.0 if s.tumour else 1.0
        self.hist = []                             # per round: taken, routed, zoomed, positives, painted, hit rate

    def seconds_full(self):
        return FULL_PER_UNIT * max(self.s.n, 1)

    def free(self):
        return self.s.tissue & ~self.zoomed & (np.ones_like(self.zoomed) if self.full_design else ~self.screened)

    def queue(self):
        s = self.s
        idx = np.flatnonzero(self.free())
        pg, fr = s.pg.reshape(-1)[idx], self.front.reshape(-1)[idx]
        if self.full_design:
            keep = (pg >= FLOOR) | fr
            idx, pg, fr = idx[keep], pg[keep], fr[keep]
        return idx[np.argsort(-(pg + 2.0 * fr), kind='stable')]

    def count(self, kind, v):
        q = self.queue()
        if kind == 'batch':
            return min(len(q), max(2, int(round(v * self.s.n))))
        pri = self.s.pg.reshape(-1)[q] + 2.0 * self.front.reshape(-1)[q]
        return int((pri >= v).sum())

    def _shift(self, m, dr, dc):
        out = np.zeros_like(m)
        r0, r1, c0, c1 = max(dr, 0), self.s.hu + min(dr, 0), max(dc, 0), self.s.wu + min(dc, 0)
        out[r0:r1, c0:c1] = m[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
        return out

    def _ring(self, m):
        out = np.zeros_like(m)
        for dr, dc in NB:
            out |= self._shift(m, dr, dc)
        return out

    def _follow_from(self, posz, paint):
        s, e = self.s, self.s.edges
        want = np.zeros_like(posz)
        want[:-1] |= (posz & e[..., 0])[1:]
        want[1:] |= (posz & e[..., 1])[:-1]
        want[:, :-1] |= (posz & e[..., 2])[:, 1:]
        want[:, 1:] |= (posz & e[..., 3])[:, :-1]
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            want |= self._shift(paint, dr, dc)
        return want & s.tissue & ~self.zoomed

    def quality(self):
        s = self.s
        rec = float(s.share[self.zoomed].sum()) if s.tumour else 1.0
        fp = float((1 - s.frac[self.painted]).sum())
        return rec - CFG['fp'] * fp / (fp + s.gt_px + 1)

    def paint_mask(self, routed, cand, tc):
        return routed & (self.s.p_core >= tc)

    def step(self, n, tc):
        s = self.s
        sel = np.zeros_like(s.tissue)
        if n > 0:
            sel.reshape(-1)[self.queue()[:n]] = True
        routed = sel if self.full_design else sel & (s.p3 >= CFG['tz'])
        cand = (routed | self.follow) & ~self.zoomed
        paint = self.paint_mask(routed, cand, tc)
        zm = cand & ~paint
        ns, nz, np_ = (0 if self.full_design else n), int(zm.sum()), int(paint.sum())
        if n + nz + np_ == 0:
            self.done = True
            return 0.0
        self.screened |= sel
        self.zoomed |= zm | paint
        self.painted |= paint
        posz = zm & s.pos
        self.front |= self._ring(posz | paint | (np.zeros_like(sel) if self.full_design else routed))
        self.follow = self._follow_from(posz, paint)
        sec = ROUND_S + ns * SCREEN_S + nz * ZOOM_S
        self.sec += sec
        J0, self.J = self.J, self.quality()
        rate = (int((sel & posz).sum()) + int((sel & paint).sum())) / max(n, 1)
        self.hist.append((n, int(routed.sum()), nz, int(posz.sum()), np_, rate))
        self.round += 1
        if self.round >= R_MAX or (len(self.queue()) == 0 and not self.follow.any()):
            self.done = True
        return float(phi(self.J, CFG['S']) - phi(J0, CFG['S']) - self.lam * sec / self.seconds_full())

    def act(self, k):
        kind, v, tc = CFG['acts'][k]
        if kind == 'stop':
            self.done = True
            return 0.0
        if kind == 'follow':
            return self.step(0, 2.0)
        return self.step(self.count(kind, v), tc)

    def mask(self):
        m = np.zeros(len(CFG['acts']), bool)
        m[0] = True
        m[1] = bool(self.follow.any())
        for i, (kind, v, _) in enumerate(CFG['acts'][2:], 2):
            m[i] = self.count(kind, v) > 0
        return m

    def features(self):
        s = self.s
        n = max(s.n, 1)
        free = self.free()
        pg = s.pg[free]
        hist = np.histogram(pg, bins=10, range=(0, 1))[0] / n
        H = np.array(self.hist) if self.hist else np.zeros((1, 6))
        tot, last = H.sum(0), H[-1]
        since = 0
        for h in reversed(self.hist):
            if h[3] + h[4] > 0:
                break
            since += 1
        nxt = np.sort(pg)[::-1][:max(2, int(0.02 * n))]
        f = [np.log(self.lam) / 5, self.round / R_MAX, self.screened.sum() / n, self.zoomed.sum() / n, self.sec / self.seconds_full(),
             np.log1p(n) / 8, *hist, (pg >= 0.5).sum() / n, (pg >= 0.9).sum() / n, pg.max() if len(pg) else 0.0,
             nxt.mean() if len(nxt) else 0.0, (tot[3] + tot[4]) / max(tot[0], 1), last[5], (tot[3] + tot[4]) / max(tot[2] + tot[4], 1),
             last[3] / n, tot[3] / n, np.log1p(tot[3]) / 6, self.follow.sum() / n, np.log1p(self.follow.sum()) / 6, float(tot[3] + tot[4] > 0),
             since / R_MAX, last[1] / max(last[0], 1), (self.front & free).sum() / n, self.painted.sum() / n, (s.p_core[free] >= 0.9).sum() / n]
        return np.asarray(f, np.float32)

    def summary(self):
        s = self.s
        seg = self.zoomed & ~self.painted
        inter = s.frac[self.zoomed].sum()
        pred = s.frac[seg].sum() + self.painted.sum()
        found = None
        if len(s.lesions):
            hit = self.zoomed.reshape(-1)[s.lu_unit]
            found = float(np.mean([hit[s.lu_lesion == l].any() for l in s.lesions]))
        n = max(s.n, 1)
        return dict(tumour=s.tumour, dice=2 * inter / (pred + s.gt_px) if s.tumour else None, found=found, screened=self.screened.sum() / n,
                    l2=seg.sum() / n, painted=self.painted.sum() / n, fp_units=int((self.painted & (s.frac < 0.5)).sum()),
                    sec=GLOBAL_S[CFG['design']] + self.sec, rounds=self.round)


def run(s, lam, actor):
    ep = Ep(s, lam)
    while not ep.done:
        if not actor(ep):
            break
    return ep


def run_with(make, s, actor, lam=1.0):
    ep = make(s, lam)
    while not ep.done:
        if not actor(ep):
            break
    return ep


def fixed_rule(t, tc):
    """Take everything at or above t in one round (full: p >= t; coarse: level-6 p >= t), then follow."""
    def act(ep):
        if ep.round == 0:
            n = ep.count('thresh', t)
            if n == 0:
                return False
            ep.step(n, tc)
        elif ep.follow.any():
            ep.step(0, 2.0)
        else:
            return False
        return True
    return act


def hit_step(ep, h, tc, batch):
    """One round of the hit-rate rule; False when finished. Stops taking units after a batch below h."""
    if not ep.stopping and len(ep.queue()) > 0:
        ep.step(ep.count('batch', batch), tc)
        if ep.hist and ep.hist[-1][5] < h:
            ep.stopping = True
        return True
    if ep.follow.any():
        ep.step(0, 2.0)
        return True
    return False


def hit_rule(h, tc, batch=0.02):
    return lambda ep: hit_step(ep, h, tc, batch)


def hindsight(s, lam, tc):
    best = None
    for b in BATCHES:
        ep = Ep(s, lam)
        obj, snaps = 0.0, [(0.0, ep.summary())]
        while not ep.done:
            n = ep.count('batch', b)
            if n == 0 and not ep.follow.any():
                break
            obj += ep.step(n, tc)
            snaps.append((obj, ep.summary()))
        o, summ = max(snaps, key=lambda x: x[0])
        if best is None or o > best[0]:
            best = (o, summ)
    return best[1]


class Net(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(N_FEAT, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.pi = nn.Linear(h, len(CFG['acts']))
        self.v = nn.Linear(h, 1)

    def forward(self, x):
        b = self.body(x)
        return self.pi(b), self.v(b)[..., 0]


def policy_actor(pol, greedy=True, rec=None):
    def act(ep):
        x = torch.from_numpy(ep.features())[None]
        m = torch.from_numpy(ep.mask())[None]
        with torch.no_grad():
            logits, v = pol(x)
        logits = logits.masked_fill(~m, -1e9)
        d = torch.distributions.Categorical(logits=logits)
        a = int(logits.argmax(-1)) if greedy else int(d.sample())
        r = ep.act(a)
        if rec is not None:
            rec.append(dict(x=x[0], m=m[0], a=a, lp=float(d.log_prob(torch.tensor([a]))), v=float(v), r=r))
        return a != 0
    return act


def action_index(kind, v, tc):
    return CFG['acts'].index((kind, v, tc))


def teacher_h(lam):
    return 0.02 if lam < 0.5 else 0.1 if lam < 3 else 0.3 if lam < 10 else 0.5


def teacher_t(lam):
    return 0.5 if lam < 0.3 else 0.8 if lam < 3 else 0.9


def bc(pol, slides, a, rng, log):
    """Clone the rule: hit-rate (2 % batches) on full, the fixed threshold on coarse; lam picks its setting."""
    tc = 0.99 if 0.99 in CFG['tcs'] else CFG['tcs'][0]
    X, M, A = [], [], []
    for _ in range(a.bc_episodes):
        s = slides[rng.integers(len(slides))]
        lam = float(np.exp(rng.uniform(np.log(a.lam_min), np.log(a.lam_max))))
        ep = Ep(s, lam)
        while not ep.done:
            m = ep.mask()
            if CFG['design'] == 'full':
                if not ep.stopping and len(ep.queue()) > 0:
                    k = action_index('batch', 0.02, tc)
                else:
                    k = 1 if m[1] else 0
            elif ep.round == 0:
                k = action_index('thresh', teacher_t(lam), tc)
                k = k if m[k] else 0
            else:
                k = 1 if m[1] else 0
            X.append(torch.from_numpy(ep.features())); M.append(torch.from_numpy(m)); A.append(k)
            ep.act(k)
            if k == 0:
                break
            if CFG['design'] == 'full' and CFG['acts'][k][0] == 'batch' and ep.hist[-1][5] < teacher_h(lam):
                ep.stopping = True
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
        for s in list(rng.choice(tum, a.episodes // 2)) + list(rng.choice(nor, a.episodes - a.episodes // 2)):
            lam = float(np.exp(rng.uniform(np.log(a.lam_min), np.log(a.lam_max))))
            rec = []
            ep = run(s, lam, policy_actor(pol, greedy=False, rec=rec))
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
        log({'update': u, 'return': float(np.mean(rets)), 'sec': time.time() - t0, 'decisions': len(batch),
             'est_s': float(np.mean([x['sec'] for x in stats])), 'l2': float(np.mean([x['l2'] for x in stats]))})


def agg(res):
    tum = [r for r in res if r['tumour']]
    nor = [r for r in res if not r['tumour']]
    m = lambda k, rs: float(np.mean([r[k] for r in rs if r[k] is not None])) if rs else float('nan')
    return dict(dice=m('dice', tum), found=m('found', tum), screened=m('screened', res), l2=m('l2', res), painted=m('painted', res),
                sec=m('sec', res), sec_normal=m('sec', nor), l2_normal=m('l2', nor), fp_normal=m('fp_units', nor), rounds=m('rounds', res))


def rule_rows(slides):
    rows = {}
    ts = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9) if CFG['design'] == 'full' else (0.3, 0.5, 0.6, 0.7, 0.8, 0.9)
    for tc in (2.0, 0.99, 0.95, 0.9):
        lab = 'no paint' if tc > 1 else f'paint {tc:g}'
        for t in ts:
            rows[f'fixed t={t:g}, {lab}'] = agg([run(s, 1.0, fixed_rule(t, tc)).summary() for s in slides])
        for h in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
            rows[f'hit h={h:g}, {lab}'] = agg([run(s, 1.0, hit_rule(h, tc)).summary() for s in slides])
    return rows


def rl_rows(slides, pol, a, prefix='RL'):
    lams = tuple(float(x) for x in np.geomspace(a.lam_min, a.lam_max, 9).round(3))
    return {f'{prefix} lam={l:g}': agg([run(s, l, policy_actor(pol)).summary() for s in slides]) for l in lams}


def table(rows, title):
    keys = ('dice', 'found', 'screened', 'l2', 'painted', 'sec', 'sec_normal', 'l2_normal', 'fp_normal', 'rounds')
    lines = [f'# {title}', '', '| policy | ' + ' | '.join(keys) + ' |', '|' + '---|' * (len(keys) + 1)]
    lines += [f'| {k} | ' + ' | '.join(f'{v[c]:.3f}' for c in keys) + ' |' for k, v in rows.items()]
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--design', choices=('full', 'coarse'), required=True)
    ap.add_argument('--g_level', type=int, default=6)
    ap.add_argument('--backbone', default='kaiko_vitl14')
    ap.add_argument('--metric', choices=('area', 'lesion', 'mix'), default='mix')
    ap.add_argument('--S', type=float, default=4.0)
    ap.add_argument('--tz', type=float, default=0.7)
    ap.add_argument('--tcs', type=float, nargs='+', default=[0.99], help='core thresholds the policy may pick; 2 = never paint')
    ap.add_argument('--fp', type=float, default=1.0, help='weight of the false-positive term in J')
    ap.add_argument('--lam_min', type=float, default=0.1)
    ap.add_argument('--lam_max', type=float, default=30.0)
    ap.add_argument('--updates', type=int, default=400)
    ap.add_argument('--episodes', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--ent', type=float, default=0.01)
    ap.add_argument('--bc_episodes', type=int, default=4000)
    ap.add_argument('--bc_epochs', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--rules_only', action='store_true')
    ap.add_argument('--tag', default='stoprl')
    a = ap.parse_args()
    CFG.update(design=a.design, tz=a.tz, S=a.S, fp=a.fp, tcs=tuple(a.tcs))
    build_actions()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    load = lambda sp, trusted=False: [Slide(t, a.metric) for t in load_tables(sp, a.design, a.g_level, a.backbone, 'cpu', trusted_only=trusted)]
    tr, te = load('train', trusted=P.NON_TRUSTED_SKIP), load('test')
    print(f'{a.tag}: {a.design}, {len(tr)} train slides ({sum(s.tumour for s in tr)} tumour), OOF {os.environ.get("RLSCREEN_OOF") == "1"}, {len(CFG["acts"])} actions', flush=True)
    if a.rules_only:
        out = {'train': rule_rows(tr), 'test': rule_rows(te), 'hindsight_test': {f'hindsight lam={l:g}': agg([hindsight(s, l, 0.99) for s in te]) for l in (0.3, 1, 3)}}
        json.dump(out, open(f'{P.ROOT}/stop_rules_{a.design}.json', 'w'), indent=1)
        open(f'{P.ROOT}/stop_rules_{a.design}.md', 'w').write(table(out['test'], f'rules, {a.design}, test') + '\n\n' + table(out['hindsight_test'], 'hindsight, test'))
        print('rules done', flush=True)
        return
    run_dir = P.run_dir(a.tag)
    os.makedirs(run_dir, exist_ok=True)
    json.dump({**vars(a), 'oof': os.environ.get('RLSCREEN_OOF') == '1'}, open(f'{run_dir}/args.json', 'w'), indent=1)
    logf = open(f'{run_dir}/train.jsonl', 'w')

    def log(row):
        logf.write(json.dumps(row) + '\n'); logf.flush()
        if row['update'] % 50 == 0 or row['update'] < 0:
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

    pol = Net()
    out = {'train': {}, 'test': {}}
    if a.bc_episodes > 0:
        bc(pol, tr, a, rng, log)
        torch.save(pol.state_dict(), f'{run_dir}/bc.pt')
        out['train'].update(rl_rows(tr, pol, a, 'clone'))
        out['test'].update(rl_rows(te, pol, a, 'clone'))
    train(pol, tr, a, rng, log)
    torch.save(pol.state_dict(), f'{run_dir}/final.pt')
    out['train'].update(rl_rows(tr, pol, a))
    out['test'].update(rl_rows(te, pol, a))
    json.dump(out, open(f'{P.ROOT}/stop_{a.tag}.json', 'w'), indent=1)
    print(table(out['test'], f'{a.tag}, test (ideal segmenter, est. A100 s)'), flush=True)


if __name__ == '__main__':
    main()
