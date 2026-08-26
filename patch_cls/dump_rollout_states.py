"""
Dumps the SASHA-0.2 rollout's final feature matrix for every CAMELYON16 slide in
train/val/test of split seed 4. Reuses step10's model loading and dataset builder so the
rollout is byte-for-byte the deployed-pipeline one (WSICosineObservationEnv, hr features
taken from the level-1 h5 exactly as step10 feeds them in). For train/val slides, also runs
two extra stochastic rollouts (actions sampled instead of argmax) seeded deterministically
per slide, as a cheap augmentation for downstream training of a classifier on these states.

python patch_cls/dump_rollout_states.py --config config/camelyon_seg_0.2.yml --seed 4 \
       --device cuda:1 --out ~/seg_outputs/patch_cls/rollout_states.pkl
"""

import argparse
import hashlib
import math
import os
import pickle
import sys
import time

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.datasets import build_HDF5_feat_dataset_2
from envs.WSI_cosine_env import WSICosineObservationEnv
from envs.WSI_env import WSIObservationEnv
from step10_sasha_attention_dump import load_models
from utils.utils import Struct, set_seed

MAX_PICKLE_BYTES = 2 * 1024 ** 3


def get_arguments():
    p = argparse.ArgumentParser('SASHA-0.2 rollout state dump')
    p.add_argument('--config', default='config/camelyon_seg_0.2.yml')
    p.add_argument('--seed', type=int, default=4)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--out', default=os.path.expanduser('~/seg_outputs/patch_cls/rollout_states.pkl'))
    p.add_argument('--n_aug', type=int, default=2, help='stochastic rollouts per train/val slide')
    return p.parse_args()


def load_conf(config_path, device, seed):
    with open(config_path, 'r') as f:
        c = yaml.load(f, Loader=yaml.FullLoader)
    c.update({'device': torch.device(device), 'seed': seed})
    return Struct(**c)


def slide_seed(name, k, base_seed):
    """Stable per-slide, per-rollout seed (python's hash() is salted per-process, so use md5)."""
    h = int(hashlib.md5(name.encode()).hexdigest(), 16)
    return (base_seed * 1_000_003 + h + k) % (2 ** 31 - 1)


@torch.no_grad()
def rollout(agent, fglobal, classifier, state, hr, label, conf, is_eval):
    """SASHA rollout exactly as step10's rollout(): deterministic when is_eval, sampled otherwise."""
    env_cls = WSIObservationEnv if conf.fglobal == 'attn' else WSICosineObservationEnv
    env = env_cls(lr_features=state, hr_features=hr, label=label, conf=conf)
    visited, done = [], False
    while not done:
        action, _, _ = agent.get_action(state, visited, is_eval=is_eval)
        state, _, done = env.step(action=action, state_update_net=fglobal, classifier_net=classifier, device=conf.device)
        visited.append(action.item())
    return visited, state


def main():
    args = get_arguments()
    conf = load_conf(args.config, args.device, args.seed)
    set_seed(args.seed)

    datasets = dict(zip(['train', 'val', 'test'], build_HDF5_feat_dataset_2(conf.level1_path, conf.level3_path, conf)))
    classifier, fglobal, agent, rl_config = load_models(conf)

    dump = {}
    visit_ok, n_total, failed = [], 0, []
    t_start = time.time()

    for split in ['train', 'val', 'test']:
        loader = DataLoader(datasets[split], batch_size=1, shuffle=False, num_workers=0)
        for i, data in enumerate(loader):
            name = data['slide_name'][0]
            t0 = time.time()
            try:
                hr = data['hr'][0].to(conf.device, dtype=torch.float32)
                lr = data['lr'].to(conf.device, dtype=torch.float32)
                label = data['label'].to(conf.device)
                n_tiles = lr.shape[1]

                visited, final_state = rollout(agent, fglobal, classifier, lr, hr, label, conf, is_eval=True)
                expected = math.ceil(n_tiles * conf.frac_visit)
                visit_ok.append(len(visited) == expected)
                n_total += 1

                logits, _ = classifier.classify(final_state)
                p_tumor = torch.softmax(logits, dim=-1)[0, 1].item()

                entry = {
                    'state': final_state[0].to(torch.float16).cpu().numpy(),
                    'visited': visited,
                    'p_tumor': p_tumor,
                    'label': int(label.item()),
                    'split': split,
                }

                if split in ('train', 'val'):
                    aug = []
                    for k in range(args.n_aug):
                        set_seed(slide_seed(name, k, args.seed))
                        v_k, s_k = rollout(agent, fglobal, classifier, lr, hr, label, conf, is_eval=False)
                        aug.append({'state': s_k[0].to(torch.float16).cpu().numpy(), 'visited': v_k})
                    entry['aug'] = aug

                dump[name] = entry
                print(f'[{split} {i + 1}/{len(datasets[split])}] {name} N={n_tiles} '
                      f'visited={len(visited)}/{expected} p={p_tumor:.4f} label={int(label.item())} '
                      f'{time.time() - t0:.1f}s', flush=True)
            except Exception as e:
                failed.append(name)
                print(f'[{split}] {name} FAILED: {e}', flush=True)

    runtime = time.time() - t_start

    test_names = [n for n, e in dump.items() if e['split'] == 'test']
    y = np.array([dump[n]['label'] for n in test_names])
    p = np.array([dump[n]['p_tumor'] for n in test_names])
    auc = float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else None
    acc = float(accuracy_score(y, (p >= 0.5).astype(int)))

    sanity = {
        'n_slides': n_total,
        'visit_count_ok': int(sum(visit_ok)),
        'n_test': len(test_names),
        'test_auc': auc,
        'test_acc': acc,
        'runtime_sec': runtime,
        'failed': failed,
    }
    print('SANITY', sanity, flush=True)

    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump({'slides': dump, 'sanity': sanity, 'rl_config': rl_config}, f)

    size = os.path.getsize(args.out)
    print(f'saved {args.out} ({size / 1024 ** 2:.1f} MB)', flush=True)
    if size > MAX_PICKLE_BYTES:
        print(f'WARNING: pickle exceeds {MAX_PICKLE_BYTES / 1024 ** 3:.0f}GB, consider per-split files', flush=True)


if __name__ == '__main__':
    main()
