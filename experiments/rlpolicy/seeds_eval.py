# Tables for the seed study (rl3_seeds.py): the PPO policy and its behaviour-cloned start for every seed, against the
# teacher (the consensus veto rule, vsel) and the supervised rule, all scored with the exact scorer.
#   --split train   cross-fitted routes (every slide played by networks fitted without its fold)
#   --split test    the all-train networks of every seed played on the test slides (logged in TEST_READS.txt)
#   --real K=DIR    instead, the maps a timing window wrote (seconds = measured wall time)
# Per model: seconds, units, Dice, FROC, lesions lost; across seeds: mean and sd. Differences to the teacher come
# from a paired bootstrap over slides (2000 resamples, the same slides drawn for both models): per seed; for the
# seed-averaged model (each resample averages the seeds' scores on the drawn slides); and with seeds resampled too.
# Usage: python -m experiments.rlpolicy.seeds_eval --split test --seeds 0 1 2 3 4 --out FILE.md

import argparse
import glob
import json
import os
import pickle
import time
from functools import partial
from multiprocessing import get_context

import numpy as np

from experiments.pfa_scannet import clean_sweep as CS
from experiments.rlcascade import fast_score as FS
from experiments.rlpolicy import exact as X
from experiments.rlpolicy import policy as P
from experiments.rlpolicy import rl3
from experiments.rlpolicy import sim as S
from experiments.rlpolicy import where as W

TEACHER = ('vsel_t1', 't_top16:0.5')
N_BOOT = 2000


def sim_results(split, seeds, workers):
    slides = S.load(split, 't1')
    runs = {s: pickle.load(open(f'{W.ROOT}/search/seeds_s{s}/res.pkl', 'rb')) for s in seeds}
    t = pickle.load(open(f'{W.ROOT}/search/{TEACHER[0]}/res.pkl', 'rb'))
    teach = t['runs'][TEACHER[1]]
    routes = {s.name: {'rule': S.rule(s)} for s in slides}
    if split == 'train':
        assert t['names'] == [s.name for s in slides]
        for s, r in zip(slides, teach['cv']):
            routes[s.name]['teacher'] = r
        for seed, res in runs.items():
            assert res['names'] == [s.name for s in slides]
            for kind in ('ppo', 'clone'):
                for s, r in zip(slides, res['runs'][kind]['cv']):
                    routes[s.name][f'{kind}_s{seed}'] = r
    else:
        players = {f'{kind}_s{seed}': rl3.player(res['runs'][kind]) for seed, res in runs.items() for kind in ('ppo', 'clone')}
        for s in slides:
            routes[s.name]['teacher'] = P.play(s, teach['theta'], teach['key'])
            for k, play in players.items():
                routes[s.name][k] = play(s)
        open(f'{W.ROOT}/TEST_READS.txt', 'a').write(f"{time.strftime('%F %T')} seed study: ppo and clone networks of seeds {seeds} (one read per network)\n")
    return X.score_routes(split, routes, workers)


def score_real(name, dirs):
    rows, maps = {}, {}
    for k, d in dirs.items():
        t = json.load(open(f'{d}/{name}_time.json'))
        rows[k] = dict(n_l3=t['screened'], n_l2=t['zoomed'], n_paint=t['painted'], sec=t['wall'], stopped=False, map=k)
        maps[k] = np.load(f'{d}/{name}.npz')['probs'].astype(np.float32)
    return name, dict(rows=rows, scores=FS.score(name, maps))


def real_results(dirs, workers):
    first = next(iter(dirs.values()))
    names = sorted(os.path.basename(f)[:-4] for f in glob.glob(f'{first}/test_*.npz'))
    with get_context('fork').Pool(workers) as pool:
        return dict(pool.map(partial(score_real, dirs=dirs), names))


class Froc:
    """CAMELYON FROC (utils.segmentation_metrics.froc) of any bootstrap resample, from slide multiplicities instead of
    rebuilt candidate lists: every candidate carries its slide's count, so one sort serves every resample."""

    def __init__(self, recs):
        fp = [(p, i, 0) for i, (f, _, _) in enumerate(recs) for p in f]
        tp = [(p, i, 1) for i, (_, t, _) in enumerate(recs) for p in t]
        c = np.array(fp + tp, dtype=np.float64).reshape(-1, 3)
        self.p, self.slide, self.is_tp = c[:, 0], c[:, 1].astype(int), c[:, 2].astype(bool)
        self.n_les = np.array([n for _, _, n in recs], np.float64)
        self.vals = np.unique(self.p)
        self.thr = self.vals[1:]
        self.idx = np.searchsorted(self.thr, self.p, side='right') - 1      # candidates at or above thr[j] are those with idx >= j
        self.n = len(recs)

    def __call__(self, w):
        """w: (n_slides,) multiplicities -> FROC score of the resample."""
        from utils.segmentation_metrics import FROC_FPS
        cw = w[self.slide]
        ok = self.idx >= 0
        nt = len(self.thr)
        fp = np.bincount(self.idx[ok & ~self.is_tp], weights=cw[ok & ~self.is_tp], minlength=nt)[::-1].cumsum()[::-1]
        tp = np.bincount(self.idx[ok & self.is_tp], weights=cw[ok & self.is_tp], minlength=nt)[::-1].cumsum()[::-1]
        present = cw > 0
        if nt == 0 or not present.any():
            return 0.0
        keep = self.thr > self.p[present].min()      # the exact FROC drops the resample's own smallest value as a threshold
        avg_fp = np.r_[fp[keep], 0.0] / w.sum()
        sens = np.r_[tp[keep], 0.0] / max((w * self.n_les).sum(), 1)
        return float(np.interp(FROC_FPS, avg_fp[::-1], sens[::-1]).mean())


class Table:
    """Per-slide arrays of one results dict, so bootstraps only index arrays."""

    def __init__(self, res):
        self.names = sorted(res)
        self.res = res
        base = [res[n]['scores']['rule'] for n in self.names]
        self.tum = np.array(['dice' in r['clean'] for r in base])
        self._froc = {}

    def dice(self, k):
        return np.array([self.res[n]['scores'][k]['clean'].get('dice', 0.0) for n in self.names])

    def froc_recs(self, k):
        return [self.res[n]['scores'][k]['raw']['froc'] for n in self.names]

    def froc_of(self, k):
        if k not in self._froc:
            self._froc[k] = Froc(self.froc_recs(k))
        return self._froc[k]

    def row(self, k, ref='rule'):
        rr = [self.res[n] for n in self.names]
        lost = sum(a > 0 and b == 0 for r in rr if 'dice' in r['scores'][ref]['clean']
                   for (_, a), (_, b) in zip(r['scores'][ref]['clean']['lesions'], r['scores'][k]['clean']['lesions']))
        mean = lambda f: float(np.mean([f(r['rows'][k]) for r in rr]))
        return dict(sec=mean(lambda x: x['sec']), l3=mean(lambda x: x['n_l3']),
                    l2=mean(lambda x: x['n_l2']), paint=mean(lambda x: x['n_paint']), dice=float(self.dice(k)[self.tum].mean()),
                    froc=float(CS.froc(self.froc_recs(k))[0]), lost=int(lost))


def boot(tab, keys, ref, seed_resample=False, n=N_BOOT, rs=0):
    """Paired bootstrap over slides of (mean over `keys` of Dice / FROC) minus `ref`'s, optionally resampling keys too."""
    rng = np.random.default_rng(rs)
    d = {k: tab.dice(k) for k in keys + [ref]}
    f = {k: tab.froc_of(k) for k in keys + [ref]}
    out = {'dice': [], 'froc': []}
    for _ in range(n):
        i = rng.integers(len(tab.names), size=len(tab.names))
        w = np.bincount(i, minlength=len(tab.names)).astype(np.float64)
        t = tab.tum[i]
        ks = [keys[j] for j in rng.integers(len(keys), size=len(keys))] if seed_resample else keys
        out['dice'].append(np.mean([d[k][i][t].mean() for k in ks]) - d[ref][i][t].mean())
        out['froc'].append(np.mean([f[k](w) for k in ks]) - f[ref](w))
    return {m: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for m, v in out.items()}


def fmt_ci(b, m):
    return f'{b[m][0]:+.4f} [{b[m][1]:+.4f}, {b[m][2]:+.4f}]'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=('train', 'test'), default='test')
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--real', nargs='*', default=None, help='K=DIR timing-window outputs; keys rule, teacher, ppo_s<seed>, clone_s<seed>')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--from_pkl', action='store_true', help='reuse the scores saved next to --out (skip the exact scoring)')
    ap.add_argument('--check_froc', action='store_true', help='check the weighted FROC against the exact one first')
    a = ap.parse_args()
    if a.from_pkl and os.path.exists(a.out[:-3] + '.pkl'):
        res = pickle.load(open(a.out[:-3] + '.pkl', 'rb'))
    else:
        res = real_results(dict(r.split('=', 1) for r in a.real), a.workers) if a.real else sim_results(a.split, a.seeds, a.workers)
        pickle.dump(res, open(a.out[:-3] + '.pkl', 'wb'))
    if a.check_froc:
        tab = Table(res)
        rng = np.random.default_rng(1)
        for k in list(res[tab.names[0]]['scores'])[:3]:
            for rep in range(3):
                i = np.arange(len(tab.names)) if rep == 0 else rng.integers(len(tab.names), size=len(tab.names))
                slow = CS.froc([tab.froc_recs(k)[j] for j in i])[0]
                fast = tab.froc_of(k)(np.bincount(i, minlength=len(tab.names)).astype(np.float64))
                print(f'FROC check {k} resample {rep}: exact {slow:.6f}  weighted {fast:.6f}', flush=True)
                assert abs(slow - fast) < 1e-9, (k, rep, slow, fast)
    tab = Table(res)
    kinds = [k for k in ('ppo', 'clone') if f'{k}_s{a.seeds[0]}' in res[tab.names[0]]['scores']]
    secs = 'measured wall' if a.real else 'cost-model estimate'
    L = [f"## Seed study, {'timing window, real maps' if a.real else a.split}, {len(tab.names)} slides ({int(tab.tum.sum())} tumour); seconds = {secs}; "
         f"lesions lost = found by the supervised rule, not by the model\n",
         '| model | s/slide | L3 | L2 | painted | Dice | FROC | lesions lost |', '|---|---|---|---|---|---|---|---|']
    rows = {}
    for k in ['rule', 'teacher'] + [f'{kind}_s{s}' for kind in kinds for s in a.seeds]:
        rows[k] = r = tab.row(k)
        L.append(f"| {k} | {r['sec']:.3f} | {r['l3']:.0f} | {r['l2']:.0f} | {r['paint']:.1f} | {r['dice']:.4f} | {r['froc']:.4f} | {r['lost']} |")
    L += ['', f'| across {len(a.seeds)} seeds | s/slide mean ± sd | L2 mean ± sd | Dice mean ± sd | FROC mean ± sd | lesions lost mean ± sd |', '|---|---|---|---|---|---|']
    for kind in kinds:
        v = {m: np.array([rows[f'{kind}_s{s}'][m] for s in a.seeds]) for m in ('sec', 'l2', 'dice', 'froc', 'lost')}
        sd = lambda x: x.std(ddof=1) if len(x) > 1 else 0.0
        L.append(f"| {kind} | {v['sec'].mean():.3f} ± {sd(v['sec']):.3f} | {v['l2'].mean():.1f} ± {sd(v['l2']):.1f} | {v['dice'].mean():.4f} ± {sd(v['dice']):.4f} | "
                 f"{v['froc'].mean():.4f} ± {sd(v['froc']):.4f} | {v['lost'].mean():.1f} ± {sd(v['lost']):.1f} |")
    L += ['', 'Paired bootstrap over slides (2000 resamples), model minus reference: mean [95 % interval]', '',
          '| comparison | ΔDice | ΔFROC |', '|---|---|---|']
    for kind in kinds:
        for ref in ('teacher', 'rule'):
            for s in a.seeds:
                b = boot(tab, [f'{kind}_s{s}'], ref, rs=s)
                L.append(f'| {kind} seed {s} − {ref} | {fmt_ci(b, "dice")} | {fmt_ci(b, "froc")} |')
            keys = [f'{kind}_s{s}' for s in a.seeds]
            b = boot(tab, keys, ref, rs=99)
            L.append(f'| **{kind} averaged over seeds − {ref}** (slides resampled) | {fmt_ci(b, "dice")} | {fmt_ci(b, "froc")} |')
            b = boot(tab, keys, ref, seed_resample=True, rs=98)
            L.append(f'| **{kind} over seeds − {ref}** (slides and seeds resampled) | {fmt_ci(b, "dice")} | {fmt_ci(b, "froc")} |')
    if 'clone' in kinds:
        for s in a.seeds:
            b = boot(tab, [f'ppo_s{s}'], f'clone_s{s}', rs=s + 50)
            L.append(f'| ppo seed {s} − its own clone | {fmt_ci(b, "dice")} | {fmt_ci(b, "froc")} |')
    text = '\n'.join(L) + '\n'
    open(a.out, 'w').write(text)
    print(text)


if __name__ == '__main__':
    main()
