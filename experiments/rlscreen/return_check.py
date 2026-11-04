# Diagnostic: mean episode return (the training objective) of the teacher, the PPO agent and the
# value-based agent at the same lambda on the test slides, to tell an estimation failure (agent
# returns below the teacher's) from a reward that prefers other operating points (agent above).
# Usage: python -m experiments.rlscreen.return_check --design full --ppo u_d2_light --q q_d2_light

import argparse
import json

import numpy as np
import torch

from experiments.rlscreen import paths as P
from experiments.rlscreen.evaluate import agg
from experiments.rlscreen.policy import ScreenPolicy
from experiments.rlscreen.qscreen import make_actor
from experiments.rlscreen.rollout import load_tables, play


def load(tag, dev):
    args = json.load(open(f'{P.run_dir(tag)}/args.json'))
    net = ScreenPolicy(args['state']).to(dev).eval()
    net.load_state_dict(torch.load(f'{P.run_dir(tag)}/final.pt', map_location=dev))
    return net, args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--design', required=True)
    ap.add_argument('--g_level', type=int, default=6)
    ap.add_argument('--backbone', default='kaiko_vitl14')
    ap.add_argument('--ppo', default=None)
    ap.add_argument('--q', default=None)
    ap.add_argument('--taus', type=float, nargs='*', default=[0.3, 0.5, 0.7, 0.9])
    ap.add_argument('--lams', type=float, nargs='*', default=[0.1, 0.3, 1.0, 3.0])
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    test = load_tables('test', a.design, a.g_level, a.backbone, a.device)
    actors = {f'teacher tau={t:g}': dict(actor='teacher', tau=t) for t in a.taus}
    if a.ppo:
        net, args = load(a.ppo, a.device)
        actors['ppo ' + a.ppo] = dict(actor='policy', policy=net, state=args['state'], greedy=True)
    if a.q:
        net, args = load(a.q, a.device)
        actors['q ' + a.q] = dict(actor=make_actor(net, args['state']))
    print('| actor | lam | return | dice | zoom | s |\n|---|---|---|---|---|---|')
    for lam in a.lams:
        for name, kw in actors.items():
            kw = dict(kw); actor = kw.pop('actor')
            res = [play(t, lam, actor, 'mix', 4.0, **kw) for t in test]
            ret = float(np.mean([sum(tr['r'] for tr in trans) for _, trans in res]))
            s = agg([r for r, _ in res])
            print(f'| {name} | {lam:g} | {ret:.3f} | {s["ideal_dice_tumour"]:.3f} | {s["zoom_frac"]:.3f} | {s["seconds"]:.2f} |', flush=True)


if __name__ == '__main__':
    main()
