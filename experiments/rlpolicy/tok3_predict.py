# Runs the token-level L3 heads over every tissue unit of every slide that has cached level-3 tokens: a train
# slide gets the head that did not see its fold, any other slide the all-train head. Per slide: token
# probabilities (n, 16, 16) as uint8 (p * 255), unit logits (n, 2) and the units' grid coordinates.
# Usage: python -m experiments.rlpolicy.tok3_predict --tag t0

import argparse
import json
import os

import numpy as np
import torch

from experiments.rlpolicy import where as W
from experiments.rlpolicy.tok3_model import TokHead


def load_head(tag, fold, device):
    d = f'{W.head_dir(tag)}/f{fold}'
    cfg = json.load(open(f'{d}/run.json'))
    net = TokHead(d=cfg['d'], ctx=cfg['ctx'], p_drop=cfg['drop']).to(device)
    net.load_state_dict(torch.load(f'{d}/head.pt', map_location=device))
    return net.eval()


@torch.no_grad()
def run(net, tok, device, bs=1024):
    tp, ul = [], []
    for i in range(0, len(tok), bs):
        t, u = net(torch.from_numpy(np.ascontiguousarray(tok[i:i + bs])).to(device).float())
        tp.append((torch.sigmoid(t) * 255).round().byte().cpu().numpy())
        ul.append(u.float().cpu().numpy())
    return np.concatenate(tp), np.concatenate(ul)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    out = f'{W.head_dir(a.tag)}/pred'
    os.makedirs(out, exist_ok=True)
    folds = W.folds()
    heads = {k: load_head(a.tag, k, a.device) for k in list(range(W.N_FOLDS)) + [-1]}
    names = sorted(f[:-7] for f in os.listdir(W.TOK3) if f.endswith('_rc.npy'))
    for i, name in enumerate(names):
        f = f'{out}/{name}.npz'
        if os.path.exists(f):
            continue
        z = np.load(f'{W.UNITS}/{name}.npz')
        k = folds[name] if str(z['split']) == 'train' else -1
        tp, ul = run(heads[k], np.load(f'{W.TOK3}/{name}.npy', mmap_mode='r'), a.device)
        np.savez(f'{f}.tmp.npz', tokp=tp, unit=ul, rc=np.load(f'{W.TOK3}/{name}_rc.npy'), fold=k)
        os.replace(f'{f}.tmp.npz', f)
        if i % 25 == 0:
            print(i, name, k, flush=True)
    open(f'{out}/DONE', 'w').close()


if __name__ == '__main__':
    main()
