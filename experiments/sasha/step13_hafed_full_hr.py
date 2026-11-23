"""
Part A of tests23: runs the released HAFED classifier on the *raw* high-res sub-patch
features (N x 16 x 384) of every CAMELYON16 test slide and dumps both attention blocks
before and after their softmax, plus the phase-1 aggregated patch features (afeat_1) that
the phase-2 rollout in step14 needs. The maths is the eval-mode path of HAFED.forward,
written out explicitly so the intermediate tensors can be kept.

python experiments/sasha/step13_hafed_full_hr.py --config config/camelyon_seg_0.2.yml --seed 4 --device cuda:1 \
       --out ~/seg_outputs/tests23/partA_dump.pkl
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from step13_common import available_hr, has_xml, load_conf, load_hr, load_labels, load_models, test_names  # noqa: E402


def get_arguments():
    p = argparse.ArgumentParser('HAFED on full high-res features')
    p.add_argument('--config', default='config/camelyon_seg_0.2.yml')
    p.add_argument('--seed', type=int, default=4)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--out', required=True)
    p.add_argument('--limit_available', action='store_true', help='debug: only slides whose .pt exists')
    return p.parse_args()


@torch.no_grad()
def hafed_full_hr(classifier, x):
    """x: (N,16,384) float32 on device. Returns the explicit eval-mode forward intermediates."""
    a1_raw = classifier.attention_1(x).transpose(0, 2).transpose(0, 1)      # N x 1 x 16, pre-softmax
    a1_soft = F.softmax(a1_raw, dim=-1)
    bag_a1 = a1_soft.mean(dim=1, keepdim=True)
    afeat_1 = torch.bmm(bag_a1, x).squeeze(1)                                # N x 384
    a2_raw = classifier.attention_2(afeat_1)                                 # 5 x N, pre-softmax
    a2_soft = F.softmax(a2_raw, dim=1)
    bag_feat = torch.mm(a2_soft.mean(0, keepdim=True), afeat_1)
    logits = classifier.Slide_classifier(bag_feat)
    p_tumor = torch.softmax(logits, dim=-1)[0, 1]
    return a1_raw, a1_soft, a2_raw, a2_soft, afeat_1, p_tumor


@torch.no_grad()
def check_against_forward(classifier, x, ours):
    """Compare our explicit maths against HAFED.forward on the same input."""
    a1_raw, a1_soft, a2_raw, a2_soft, afeat_1, p_tumor = ours
    _, slide_logits, f_a1, f_a2, f_afeat1, f_attn_raw = classifier.forward(x.unsqueeze(0))
    f_p = torch.softmax(slide_logits, dim=-1)[0, 1]
    return {
        'a1_soft_maxdiff': float((a1_soft - f_a1).abs().max()),
        'a2_raw_maxdiff': float((a2_raw - f_attn_raw).abs().max()),
        'a2_soft_maxdiff': float((a2_soft - f_a2).abs().max()),
        'afeat1_maxdiff': float((afeat_1 - f_afeat1).abs().max()),
        'p_tumor_diff': float((p_tumor - f_p).abs()),
    }


def main():
    args = get_arguments()
    conf = load_conf(args.config, args.device, args.seed)
    names = test_names(args.seed)
    if args.limit_available:
        have = set(available_hr())
        names = [n for n in names if n in have]
    print(f'{len(names)} slides, device {conf.device}', flush=True)

    classifier, _, _, _ = load_models(conf)
    labels = load_labels(names)

    dump, checks, warnings = {}, {}, []
    for i, name in enumerate(names):
        t0 = time.time()
        hr = load_hr(name)
        x = hr['feat'].to(conf.device, dtype=torch.float32)
        out = hafed_full_hr(classifier, x)
        a1_raw, a1_soft, a2_raw, a2_soft, afeat_1, p_tumor = out
        if i < 3:
            checks[name] = check_against_forward(classifier, x, out)
        xml = has_xml(name)
        if xml != bool(labels[name]):
            warnings.append(f'{name}: h5 label={labels[name]} but xml {"found" if xml else "missing"}')
        dump[name] = {
            'A1_raw': a1_raw.squeeze(1).cpu().numpy(),
            'A1_soft': a1_soft.squeeze(1).cpu().numpy(),
            'A2_raw': a2_raw.cpu().numpy(),
            'A2_soft': a2_soft.cpu().numpy(),
            'afeat_1': afeat_1.to(torch.float16).cpu().numpy(),
            'p_tumor': float(p_tumor),
            'label': labels[name],
            'has_xml': xml,
            'coords': hr['coords'],
        }
        print(f'[{i + 1}/{len(names)}] {name} N={x.shape[0]} p={float(p_tumor):.4f} '
              f'label={labels[name]} {time.time() - t0:.1f}s', flush=True)
        del x, a1_raw, a1_soft, a2_raw, a2_soft, afeat_1

    y = np.array([dump[n]['label'] for n in names])
    p = np.array([dump[n]['p_tumor'] for n in names])
    auc = float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else None
    acc = float(((p >= 0.5).astype(int) == y).mean())
    rowsum = {n: float(np.abs(dump[n]['A1_soft'].sum(1) - 1).max()) for n in names}
    branchsum = {n: float(np.abs(dump[n]['A2_soft'].sum(1) - 1).max()) for n in names}
    sanity = {'forward_match': checks, 'auc': auc, 'acc': acc,
              'A1_soft_rowsum_maxdev': max(rowsum.values()), 'A2_soft_branchsum_maxdev': max(branchsum.values()),
              'label_xml_warnings': warnings, 'n_slides': len(names)}
    print('SANITY', sanity, flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump({'slides': dump, 'sanity': sanity, 'config': str(args.config), 'seed': args.seed}, f)
    print('saved', args.out, flush=True)


if __name__ == '__main__':
    main()
