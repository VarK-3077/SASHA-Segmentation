# Seed study of one PPO recipe (rl3.py; defaults = the consensus-clone agent rl3_m4@lam0.3). The training run is
# repeated with several seeds, and for every seed and fold both networks are kept: the behaviour-cloned start (the
# teacher's network copy, before any PPO) and the PPO policy after fine-tuning. The seed drives the clone's
# minibatches, the networks' initialisation and every PPO rollout sample; readers, folds and the teacher are fixed.
# Seed 0 reproduces rl3.py's own seeding. Per seed: ROOT/search/seeds_s<seed>/res.pkl with runs 'ppo' and 'clone',
# each holding the cross-fitted routes ('cv'), the all-train networks ('nets') and the fold networks.
# Usage: python -m experiments.rlpolicy.rl3_seeds --seeds 0 1 2 3 4 --devices cuda:0 cuda:1 [--workers 30]

import argparse
import copy
import os
import pickle
from multiprocessing import get_context

import numpy as np
import torch

from experiments.rlpolicy import rl3
from experiments.rlpolicy import sim as S
from experiments.rlpolicy import where as W

CLONE = {}


def keep_clone(orig):
    """rl3.clone, also keeping a copy of the cloned weights (the PPO run then keeps training the same networks)."""
    def f(nets, *args, **kw):
        orig(nets, *args, **kw)
        CLONE['sd'] = {n: v.detach().cpu().clone() for n, v in nets.state_dict().items()}
    return f


def job(task):
    seed, k, dev, a = task
    a = copy.copy(a)
    a.device = dev
    fold = W.folds()
    torch.set_num_threads(1)
    on = lambda d: dict(s=d['s'], x6=d['x6'].to(dev), x3=d['x3'].to(dev))
    nets = rl3.ppo([on(d) for d in rl3.SLIDES if fold[d['s'].name] != k], a.lam, a, seed=1000 * seed + 10 * (k + 1), veto=a.vetoes[k])
    clone = rl3.Nets(*nets.pi6[0].weight.shape[1:], *nets.pi3[0].weight.shape[1:]).to(dev)
    clone.load_state_dict(CLONE['sd'])
    held = [(d['s'].name, rl3.greedy(nets, on(d)), rl3.greedy(clone, on(d))) for d in rl3.SLIDES if fold[d['s'].name] == k]
    cpu = lambda m: {n: v.detach().cpu().clone() for n, v in m.state_dict().items()}
    part = part_path(seed, k)
    torch.save(dict(ppo=cpu(nets), clone=cpu(clone), held=held), f'{part}.tmp')
    os.replace(f'{part}.tmp', part)
    return seed, k      # results travel through the file: tensors sent back through the pool's pipes ran out of descriptors


def part_path(seed, k):
    d = f'{W.ROOT}/search/seeds_parts'
    os.makedirs(d, exist_ok=True)
    return f'{d}/s{seed}_k{k}.pt'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--devices', nargs='+', default=['cuda:0'])
    ap.add_argument('--workers', type=int, default=30)
    ap.add_argument('--tok3', default='t1')
    ap.add_argument('--lam', type=float, default=0.3)
    ap.add_argument('--state', default='sup+tok')
    ap.add_argument('--init', default='consensus')
    ap.add_argument('--teacher', default='vsel_t1@t_top16:0.5')
    ap.add_argument('--kl', type=float, default=0.02)
    ap.add_argument('--w_les', type=float, default=8.0)
    ap.add_argument('--w_fp', type=float, default=rl3.W_FP)
    ap.add_argument('--episodes', type=int, default=128)
    ap.add_argument('--updates', type=int, default=800)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--critic_lr', type=float, default=1e-3)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--ent', type=float, default=0.003)
    ap.add_argument('--vscale', type=float, default=0.05)
    a = ap.parse_args()
    assert a.init == 'consensus', 'the clone kept here is the consensus clone'
    a.log_every, a.tok6 = 10 ** 9, None
    name, rk = a.teacher.split('@')
    t = pickle.load(open(f'{W.ROOT}/search/{name}/res.pkl', 'rb'))['runs'][rk]
    a.vetoes = [float(rk.split(':')[1]) * v for v in t['v_fold']] + [float(t['theta'].veto)]   # index -1 = all slides
    slides = S.load('train', a.tok3)
    rl3.SLIDES = [dict(s=s, **dict(zip(('x6', 'x3'), rl3.states(s, a.state, None)))) for s in slides]
    rl3.clone = keep_clone(rl3.clone)
    rule = [S.rule(s) for s in slides]
    ks = list(range(W.N_FOLDS)) + [-1]
    todo = [(s, k) for s in a.seeds for k in ks if not os.path.exists(part_path(s, k))]
    tasks = [(seed, k, a.devices[i % len(a.devices)], a) for i, (seed, k) in enumerate(todo)]
    print(f'{len(todo)} networks to train, {len(a.seeds) * len(ks) - len(todo)} already on disk', flush=True)
    if tasks:
        with get_context('fork').Pool(min(a.workers, len(tasks))) as pool:
            for seed, k in pool.imap_unordered(job, tasks):
                print(f'seed {seed} fold {k} done', flush=True)
    got = {}
    for seed in a.seeds:
        for j in ks:
            r = torch.load(part_path(seed, j), weights_only=False)
            got[(seed, j)] = (r['ppo'], r['clone'], r['held'])
    for seed in a.seeds:
            by_ppo, by_clone = {}, {}
            for j in range(W.N_FOLDS):
                for n, r_ppo, r_clone in got[(seed, j)][2]:
                    by_ppo[n], by_clone[n] = r_ppo, r_clone
            runs = {}
            for kind, by, i in (('ppo', by_ppo, 0), ('clone', by_clone, 1)):
                runs[kind] = dict(cv=[by[s.name] for s in slides], nets=got[(seed, -1)][i], fold_nets=[got[(seed, j)][i] for j in range(W.N_FOLDS)],
                                  state=a.state, tok6=None, seed=seed)
            out = f'{W.ROOT}/search/seeds_s{seed}'
            os.makedirs(out, exist_ok=True)
            pickle.dump(dict(names=[s.name for s in slides], rule=rule, runs=runs, args=vars(a)), open(f'{out}/res.pkl', 'wb'))
            for kind in ('ppo', 'clone'):
                m = S.summarise(runs[kind]['cv'])
                print(f"seed {seed} {kind}: cross-fitted {m['sec']:.3f} s  Dice {m['dice']:.4f}  FROC {m['froc']:.4f}  L2 {m['l2']:.0f}", flush=True)
    open(f'{W.ROOT}/search/SEEDS_DONE', 'w').close()


if __name__ == '__main__':
    main()
