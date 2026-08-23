"""
Runs SASHA inference (same models and rollout as step7_inference.py) over the train/val/test
splits and dumps, per slide, what segmentation evaluation needs: the final low-res attention
over patches, the patch coordinates, the tumor probability and the visited patch ids.
Metrics are computed separately by step11_segmentation_eval.py so thresholding rules can be
changed without re-running the policy.

python step10_sasha_attention_dump.py --config config/camelyon_sasha_inference.yml --seed 4 --out_dir OUT
"""

import argparse
import os
import pickle
from pprint import pprint
from types import SimpleNamespace

import h5py
import torch
import yaml
from torch.utils.data import DataLoader

from architecture.transformer import HAFED
from datasets.datasets import build_HDF5_feat_dataset_2
from envs.WSI_cosine_env import WSICosineObservationEnv
from envs.WSI_env import WSIObservationEnv
from modules.fglobal_mlp import FGlobal
from rl_algorithms.ppo import Agent, Actor, Critic
from step4_extract_intermediate_features import load_model
from step7_inference import load_policy_model
from utils.gpu_utils import check_gpu_availability
from utils.utils import MetricLogger, Struct, set_seed


def get_arguments():
    parser = argparse.ArgumentParser('SASHA attention dump', add_help=False)
    parser.add_argument('--config', default=None, help='path to config file')
    parser.add_argument('--seed', type=int, default=4, help='split / random seed')
    parser.add_argument('--classifier_arch', default='hafed', choices=['hafed'])
    parser.add_argument('--out_dir', required=True, help='where to write attention_dump.pkl')
    parser.add_argument('--splits', default='train,val,test', help='comma separated subset of train,val,test')
    args = parser.parse_args()
    gpus = check_gpu_availability(3, 1, [])
    args.device = torch.device(f"cuda:{gpus[0]}")
    return args


def load_models(conf):
    classifier_dict, _, config, _ = load_model(conf.classifier_ckpt_path, conf)
    cc = SimpleNamespace(**config)
    classifier = HAFED(cc, n_token_1=cc.n_token_1, n_token_2=cc.n_token_2, n_masked_patch_1=cc.n_masked_patch_1,
                       n_masked_patch_2=cc.n_masked_patch_2, mask_drop=cc.mask_drop).to(conf.device)
    classifier.load_state_dict(classifier_dict)
    classifier.eval()

    fglobal = FGlobal(ip_dim=384 * 3, op_dim=384).to(conf.device)
    fglobal.load_state_dict(torch.load(conf.mlp_fglobal_ckpt, map_location=conf.device)['model'])
    fglobal.eval()

    actor, critic = Actor(conf=conf), Critic(conf=conf)
    agent = Agent(actor, critic, conf).to(conf.device)
    a_opt = torch.optim.AdamW(actor.parameters(), lr=0.001)
    c_opt = torch.optim.AdamW(critic.parameters(), lr=0.001)
    agent, _, _, _, rl_config = load_policy_model(agent, a_opt, c_opt, conf.rl_ckpt_path, conf.device)
    agent.eval()
    return classifier, fglobal, agent, rl_config


@torch.no_grad()
def rollout(agent, fglobal, classifier, loader, header, conf):
    """Deterministic (pick-max) SASHA rollout; returns {slide: {...}} with attention over LR patches."""
    out = {}
    logger = MetricLogger(delimiter=" ")
    for data in logger.log_every(loader, 50, header):
        hr = data['hr'][0].to(conf.device, dtype=torch.float32)
        state = data['lr'].to(conf.device, dtype=torch.float32)
        label = data['label'].to(conf.device)
        name = data['slide_name'][0]
        env_cls = WSIObservationEnv if conf.fglobal == 'attn' else WSICosineObservationEnv
        env = env_cls(lr_features=state, hr_features=hr, label=label, conf=conf)

        visited, done = [], False
        while not done:
            action, _, _ = agent.get_action(state, visited, is_eval=True)
            state, _, done = env.step(action=action, state_update_net=fglobal, classifier_net=classifier, device=conf.device)
            visited.append(action.item())

        logits, attn = classifier.classify(state)
        prob = torch.softmax(logits, dim=-1)[0]
        out[name] = {
            'attention': attn.mean(0).cpu().numpy(),
            'attention_heads': attn.cpu().numpy(),
            'p_tumor': prob[1].item(),
            'label': int(label.item()),
            'visited': visited,
        }
    return out


@torch.no_grad()
def hafed_full(classifier, loader, header, conf):
    """HAFED with every patch in high resolution (no sampling): reference upper bound."""
    out = {}
    logger = MetricLogger(delimiter=" ")
    for data in logger.log_every(loader, 50, header):
        hr = data['hr'].to(conf.device, dtype=torch.float32)
        logits, attn = classifier.classify(hr)
        prob = torch.softmax(logits, dim=-1)[0]
        out[data['slide_name'][0]] = {'attention': attn.mean(0).cpu().numpy(), 'attention_heads': attn.cpu().numpy(),
                                      'p_tumor': prob[1].item(), 'label': int(data['label'].item())}
    return out


def read_coords(level3_path, pretrain, names):
    with h5py.File(os.path.join(level3_path, f'patch_feats_pretrain_{pretrain}.h5'), 'r') as f:
        return {n: f[n]['coords'][:] for n in names}


def main():
    args = get_arguments()
    with open(args.config, 'r') as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        c.update(vars(args))
        conf = Struct(**c)
    pprint(vars(conf))
    set_seed(args.seed)

    datasets = dict(zip(['train', 'val', 'test'], build_HDF5_feat_dataset_2(conf.level1_path, conf.level3_path, conf)))
    classifier, fglobal, agent, rl_config = load_models(conf)

    dump = {'config': vars(conf), 'rl_config': rl_config, 'sasha': {}, 'hafed': {}, 'coords': {}, 'split': {}}
    for split in args.splits.split(','):
        loader = DataLoader(datasets[split], batch_size=1, shuffle=False, num_workers=conf.n_worker)
        dump['split'][split] = list(datasets[split].data_names)
        dump['sasha'][split] = rollout(agent, fglobal, classifier, loader, f'SASHA {split}', conf)
        dump['hafed'][split] = hafed_full(classifier, loader, f'HAFED {split}', conf)
        dump['coords'].update(read_coords(conf.level3_path, conf.pretrain, dump['split'][split]))

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'attention_dump.pkl'), 'wb') as f:
        pickle.dump(dump, f)
    print('saved', os.path.join(args.out_dir, 'attention_dump.pkl'))


if __name__ == '__main__':
    main()
