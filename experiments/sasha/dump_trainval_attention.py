"""
Attention-score dump for the CAMELYON16 train/val slides (split_4), the train/val counterpart
of step13_hafed_full_hr.py's test dump so pseudo-labeling has the same A1/A2 conventions on
every split. Two independent passes per slide, reusing step13's model loading and maths:
  - full-HR pass, only for slides with a raw HR feature file in ~/seg_outputs/hr_train_feats
    (15 normal slides are missing there and are skipped for this half only): A1_raw and
    A2_raw/A2_soft on the true phase-1 aggregate afeat_1, plus p_tumor.
  - rollout pass, for every train/val slide, using the SASHA-0.2 rollout's final state from
    ~/seg_outputs/patch_cls/rollout_states.pkl: A2f_raw/A2f_soft = attention_2 on that state
    (step14's phase2()), plus p_tumor_rollout via classifier.classify().

python experiments/sasha/dump_trainval_attention.py --config config/camelyon_seg_0.2.yml --seed 4 \
       --device cuda:0 --out ~/seg_outputs/patch_cls/trainval_attention.pkl
"""

import argparse
import json
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

from step13_common import load_conf, load_hr, load_labels, load_models

HR_TRAIN_DIR = os.path.expanduser('~/seg_outputs/hr_train_feats')
ROLLOUT_PATH = os.path.expanduser('~/seg_outputs/patch_cls/rollout_states.pkl')
LABELS_PATH = os.path.expanduser('~/seg_outputs/patch_cls/labels.pkl')


def get_arguments():
    p = argparse.ArgumentParser('HAFED attention dump for train/val slides')
    p.add_argument('--config', default='config/camelyon_seg_0.2.yml')
    p.add_argument('--seed', type=int, default=4)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', default=os.path.expanduser('~/seg_outputs/patch_cls/trainval_attention.pkl'))
    return p.parse_args()


def trainval_names(seed):
    with open(f'./dataset_csv/camelyon16/splits/split_{seed}.json', 'r') as f:
        d = json.load(f)
    return d['train_names'] + d['val_names']


@torch.no_grad()
def hafed_full_hr(classifier, x):
    """Same maths as step13_hafed_full_hr.hafed_full_hr, trimmed to what pseudo-labeling needs."""
    a1_raw = classifier.attention_1(x).transpose(0, 2).transpose(0, 1)      # N x 1 x 16, pre-softmax
    a1_soft = F.softmax(a1_raw, dim=-1)
    bag_a1 = a1_soft.mean(dim=1, keepdim=True)
    afeat_1 = torch.bmm(bag_a1, x).squeeze(1)                                # N x 384
    a2_raw = classifier.attention_2(afeat_1)                                 # 5 x N, pre-softmax
    a2_soft = F.softmax(a2_raw, dim=1)
    bag_feat = torch.mm(a2_soft.mean(0, keepdim=True), afeat_1)
    p_tumor = torch.softmax(classifier.Slide_classifier(bag_feat), dim=-1)[0, 1]
    return a1_raw, a2_raw, a2_soft, p_tumor


@torch.no_grad()
def rollout_attention(classifier, state):
    """A2f_raw/A2f_soft = attention_2 on the rollout's final state, same as step14's phase2()."""
    a2f_raw = classifier.attention_2(state)
    a2f_soft = F.softmax(a2f_raw, dim=1)
    logits, _ = classifier.classify(state.unsqueeze(0))
    p_tumor_rollout = torch.softmax(logits, dim=-1)[0, 1]
    return a2f_raw, a2f_soft, p_tumor_rollout


def main():
    args = get_arguments()
    conf = load_conf(args.config, args.device, args.seed)
    names = trainval_names(args.seed)
    print(f'{len(names)} train/val slides, device {conf.device}', flush=True)

    classifier, _, _, _ = load_models(conf)
    labels = load_labels(names)
    with open(ROLLOUT_PATH, 'rb') as f:
        rollout = pickle.load(f)['slides']

    dump, n_hr, n_rollout, missing_hr, missing_rollout = {}, 0, 0, [], []
    for i, name in enumerate(names):
        t0 = time.time()
        entry = {'label': labels[name]}
        hr_path = os.path.join(HR_TRAIN_DIR, name + '.pt')
        n_hr_tiles = None
        if os.path.exists(hr_path):
            hr = load_hr(name, hr_dir=HR_TRAIN_DIR)
            x = hr['feat'].to(conf.device, dtype=torch.float32)
            n_hr_tiles = x.shape[0]
            a1_raw, a2_raw, a2_soft, p_tumor = hafed_full_hr(classifier, x)
            entry.update({
                'A1_raw': a1_raw.squeeze(1).to(torch.float16).cpu().numpy(),
                'A2_raw': a2_raw.to(torch.float16).cpu().numpy(),
                'A2_soft': a2_soft.to(torch.float16).cpu().numpy(),
                'p_tumor': float(p_tumor),
                'coords': hr['coords'],
            })
            n_hr += 1
            del x, a1_raw, a2_raw, a2_soft
        else:
            missing_hr.append(name)

        if name in rollout:
            state_np = rollout[name]['state']
            if n_hr_tiles is not None and state_np.shape[0] != n_hr_tiles:
                print(f'WARNING {name}: hr N={n_hr_tiles} but rollout state N={state_np.shape[0]}', flush=True)
            state = torch.from_numpy(state_np).to(conf.device, dtype=torch.float32)
            a2f_raw, a2f_soft, p_tumor_rollout = rollout_attention(classifier, state)
            entry.update({
                'A2f_raw': a2f_raw.to(torch.float16).cpu().numpy(),
                'A2f_soft': a2f_soft.to(torch.float16).cpu().numpy(),
                'p_tumor_rollout': float(p_tumor_rollout),
            })
            n_rollout += 1
            del state, a2f_raw, a2f_soft
        else:
            missing_rollout.append(name)

        dump[name] = entry
        p_str = f'{entry["p_tumor"]:.4f}' if 'p_tumor' in entry else 'n/a'
        pr_str = f'{entry["p_tumor_rollout"]:.4f}' if 'p_tumor_rollout' in entry else 'n/a'
        print(f'[{i + 1}/{len(names)}] {name} label={labels[name]} p_hr={p_str} p_roll={pr_str} '
              f'{time.time() - t0:.1f}s', flush=True)

    hr_names = [n for n in names if 'p_tumor' in dump[n]]
    roll_names = [n for n in names if 'p_tumor_rollout' in dump[n]]
    y_hr = np.array([labels[n] for n in hr_names])
    p_hr = np.array([dump[n]['p_tumor'] for n in hr_names])
    auc_hr = float(roc_auc_score(y_hr, p_hr)) if len(set(y_hr.tolist())) > 1 else None
    y_roll = np.array([labels[n] for n in roll_names])
    p_roll = np.array([dump[n]['p_tumor_rollout'] for n in roll_names])
    auc_roll = float(roc_auc_score(y_roll, p_roll)) if len(set(y_roll.tolist())) > 1 else None

    spearman = {}
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, 'rb') as f:
            tile_labels = pickle.load(f)
        tumor_hr_names = [n for n in hr_names if labels[n] == 1 and n in tile_labels][:3]
        for n in tumor_hr_names:
            a1_max = dump[n]['A1_raw'].astype(np.float32).max(axis=1)
            tile_frac = tile_labels[n]['tile_frac'].astype(np.float32)
            if len(a1_max) == len(tile_frac):
                spearman[n] = float(spearmanr(a1_max, tile_frac).statistic)
            else:
                spearman[n] = f'length mismatch {len(a1_max)} vs {len(tile_frac)}'

    sanity = {
        'n_slides': len(names), 'n_with_hr': n_hr, 'n_with_rollout': n_rollout,
        'missing_hr': missing_hr, 'missing_rollout': missing_rollout,
        'auc_p_tumor_hr': auc_hr, 'auc_p_tumor_rollout': auc_roll,
        'spearman_A1max_vs_tilefrac': spearman,
    }
    print('SANITY', sanity, flush=True)

    dump['_meta'] = {
        'description': 'per-slide HAFED attention dump for CAMELYON16 train/val (split_4) slides',
        'keys_from_hr_pass': {
            'A1_raw': '(N,16) fp16, pre-softmax phase-1 attention over the 16 sub-patches',
            'A2_raw': '(5,N) fp16, pre-softmax phase-2 attention on afeat_1',
            'A2_soft': '(5,N) fp16, softmax(A2_raw, dim=1)',
            'p_tumor': 'float, slide tumor probability from the full-HR forward',
            'coords': '(N,2), tile coords',
        },
        'keys_from_rollout_pass': {
            'A2f_raw': '(5,N) fp16, attention_2 on the SASHA-0.2 rollout final state',
            'A2f_soft': '(5,N) fp16, softmax(A2f_raw, dim=1)',
            'p_tumor_rollout': 'float, slide tumor probability via classifier.classify() on the rollout state',
        },
        'other_keys': {'label': 'int, slide-level ground truth'},
        'missing_hr': missing_hr,
        'missing_rollout': missing_rollout,
        'config': str(args.config), 'seed': args.seed,
        'sanity': sanity,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump(dump, f)
    size_mb = os.path.getsize(args.out) / 1024 ** 2
    print(f'saved {args.out} ({size_mb:.1f} MB)', flush=True)


if __name__ == '__main__':
    main()
