"""
Part B of tests23: the SASHA-0.2 rollout exactly as step10 runs it, except the high-res
features handed to the environment are the phase-1 aggregates recomputed from the raw
sub-patch features in step13 (afeat_1) instead of the pre-baked level-1 h5 features.
Dumps the visited patch order and the final phase-2 attention (raw + softmax) per slide.

python experiments/sasha/step14_sasha_rollout_p1feat.py --config config/camelyon_seg_0.2.yml --seed 4 \
       --device cuda:1 --partA ~/seg_outputs/tests23/partA_dump.pkl \
       --out ~/seg_outputs/tests23/partB_dump.pkl
"""

import argparse
import math
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from envs.WSI_cosine_env import WSICosineObservationEnv  # noqa: E402
from envs.WSI_env import WSIObservationEnv  # noqa: E402
from step13_common import load_conf, load_labels, load_lr, load_models  # noqa: E402
from utils.utils import set_seed  # noqa: E402


def get_arguments():
    p = argparse.ArgumentParser('SASHA rollout on phase-1 aggregated HR features')
    p.add_argument('--config', default='config/camelyon_seg_0.2.yml')
    p.add_argument('--seed', type=int, default=4)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--partA', required=True)
    p.add_argument('--old_dump', default=os.path.expanduser('~/seg_outputs/sasha_0.2/attention_dump.pkl'))
    p.add_argument('--out', required=True)
    return p.parse_args()


@torch.no_grad()
def rollout_slide(agent, fglobal, classifier, state, hr, label, conf):
    """Deterministic SASHA rollout; returns (visited, final_state)."""
    env_cls = WSIObservationEnv if conf.fglobal == 'attn' else WSICosineObservationEnv
    env = env_cls(lr_features=state, hr_features=hr, label=label, conf=conf)
    visited, done = [], False
    while not done:
        action, _, _ = agent.get_action(state, visited, is_eval=True)
        state, _, done = env.step(action=action, state_update_net=fglobal, classifier_net=classifier,
                                  device=conf.device)
        visited.append(action.item())
    return visited, state


@torch.no_grad()
def phase2(classifier, state):
    a2_raw = classifier.attention_2(state[0])
    a2_soft = F.softmax(a2_raw, dim=1)
    bag_feat = torch.mm(a2_soft.mean(0, keepdim=True), state[0])
    p_tumor = torch.softmax(classifier.Slide_classifier(bag_feat), dim=-1)[0, 1]
    return a2_raw, a2_soft, p_tumor


def main():
    args = get_arguments()
    conf = load_conf(args.config, args.device, args.seed)
    set_seed(args.seed)
    with open(args.partA, 'rb') as f:
        partA = pickle.load(f)['slides']
    names = list(partA.keys())
    lr = load_lr(conf, names)
    labels = load_labels(names)
    classifier, fglobal, agent, rl_config = load_models(conf)
    print(f'{len(names)} slides, frac_visit={conf.frac_visit}, cos_thr={conf.cosine_threshold}', flush=True)

    dump, visit_ok = {}, []
    for i, name in enumerate(names):
        t0 = time.time()
        hr = torch.from_numpy(partA[name]['afeat_1']).to(conf.device, dtype=torch.float32)
        state = torch.from_numpy(lr[name]['feat']).unsqueeze(0).to(conf.device, dtype=torch.float32)
        label = torch.tensor([labels[name]], device=conf.device)
        assert hr.shape[0] == state.shape[1], f'{name}: hr {hr.shape} vs lr {state.shape}'
        visited, final_state = rollout_slide(agent, fglobal, classifier, state, hr, label, conf)
        a2_raw, a2_soft, p_tumor = phase2(classifier, final_state)
        expected = math.ceil(state.shape[1] * conf.frac_visit)
        visit_ok.append(len(visited) == expected)
        dump[name] = {'visited': visited, 'n_expected': expected,
                      'A2f_raw': a2_raw.cpu().numpy(), 'A2f_soft': a2_soft.cpu().numpy(),
                      'p_tumor': float(p_tumor), 'label': labels[name],
                      'coords': lr[name]['coords']}
        print(f'[{i + 1}/{len(names)}] {name} N={state.shape[1]} visited={len(visited)}/{expected} '
              f'p={float(p_tumor):.4f} label={labels[name]} {time.time() - t0:.1f}s', flush=True)

    y = np.array([dump[n]['label'] for n in names])
    p = np.array([dump[n]['p_tumor'] for n in names])
    auc = float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else None
    acc = float(((p >= 0.5).astype(int) == y).mean())

    rhos = {}
    if os.path.exists(args.old_dump):
        with open(args.old_dump, 'rb') as f:
            old = pickle.load(f)['sasha']['test']
        for n in names:
            if n in old:
                rhos[n] = float(spearmanr(dump[n]['A2f_soft'].mean(0), old[n]['attention']).statistic)
    sanity = {'visit_count_ok': int(sum(visit_ok)), 'n_slides': len(names), 'auc': auc, 'acc': acc,
              'spearman_vs_old_median': float(np.median(list(rhos.values()))) if rhos else None,
              'spearman_vs_old_min': float(np.min(list(rhos.values()))) if rhos else None,
              'spearman_n': len(rhos), 'spearman_identical': int(sum(r > 0.9999 for r in rhos.values()))}
    print('SANITY', sanity, flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump({'slides': dump, 'sanity': sanity, 'spearman': rhos, 'rl_config': rl_config}, f)
    print('saved', args.out, flush=True)


if __name__ == '__main__':
    main()
