"""Stage-by-stage inference timing on the CAMELYON16 test slides for SASHA and for Model-1, from the raw slide.

sasha:  the authors' on-the-fly pipeline of step7_inference_with_fe.py, unchanged, with timers around its stages:
        tissue segmentation + patching, low-res (level 3) features for every tile, then the sequential RL loop
        (agent picks a tile, its 16 high-res sub-patches are read and embedded, TSU updates the state) and the
        final slide classification.
hafed:  the full high-res baseline as the authors run it: same patching, then their step2 loader reading the 16
        high-res sub-patches AND the low-res tile of every tile (batch 32, 8 workers), all through the ViT, then
        the HAFED hierarchical-attention classifier on the N x 16 high-res features.
model1: same patching, then the 16 high-res sub-patches of EVERY tile through the ViT (the step12 extraction),
        the exported Model-1 ensemble on those features, and a thresholded heat map with connected components.
        --policy random | greedy | uncertain replaces the agent's choice of the next tile by a supervised rule
        (rules.py; Model-2 on the current state); everything else is the authors' pipeline.
tilemlp: same patching and the same low-res (level 3) pass of every tile, then a supervised tile classifier
        (tile_mlp.py: linear / MLP on the ViT's CLS feature, or an MLP on the mean and max of its patch tokens) and
        a pooled slide score. No high-res reads.

Timing contract: every value under `times` is wall-clock. `total_wall` is a plain stopwatch over the slide (patching
start to slide label) with the authors' loader settings (8 workers, reads overlapped with the GPU). The wall-clock
cost of reading in such a pipeline is the time the GPU loop waited for the loader: `read_wall` = `loader_startup`
(workers spawned, first batch) + `wait_io` (later batches not ready); reading hidden under the ViT costs nothing.
CPU-seconds summed over the workers go to a separate `diagnostics` block (`read_cpu_sum_*`), never into `times`.
Do not change the hafed paths to num_workers=0: that measures a serialized pipeline nobody ships.

Run from the repo root. One json per slide in <save_dir>/<mode>/, `--summarise` prints the averages."""

import argparse
import glob
import json
import os
import sys
import time
from multiprocessing.pool import ThreadPool
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import yaml
from scipy import ndimage
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def now():
    torch.cuda.synchronize()
    return time.perf_counter()


def drop_page_cache(path):
    """Every slide starts cold, whatever ran before it."""
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)


def warm_up(vit, device, batch_sizes):
    with torch.no_grad():
        for b in batch_sizes:
            vit(torch.zeros(b, 3, 224, 224, device=device))
    torch.cuda.synchronize()


def threaded_crops(helper, pool, xy, wsi):
    """Same 16 crops, same order as Helper.get_patches_for_selected_action, read by 16 threads instead of one."""
    pts = [(int(xy[0]) + i * 512, int(xy[1]) + j * 512) for i in range(4) for j in range(4)]
    return pool.map(lambda pt: helper.img_transforms(wsi.read_region(pt, 1, (256, 256)).convert('RGB')), pts)


def fresh_coords(helper, slide):
    with h5py.File(os.path.join(helper.step1_save_dir, 'patches', slide + '.h5'), 'r') as f:
        return f['coords'][:]


def time_sasha(conf, names, out_dir):
    from envs.WSI_cosine_env_inference import WSICosineObservationEnv_inference
    from rl_algorithms.ppo import Actor, Agent, Critic
    from step7_inference_with_fe import get_classifier, get_tsu, load_agent
    from utils.inference_utils import Helper

    agent_dict, agent_cfg = load_agent(conf)
    agent_conf = SimpleNamespace(**agent_cfg)
    agent = Agent(Actor(conf=agent_conf), Critic(conf=agent_conf), agent_conf).to(conf.device)
    agent.load_state_dict(agent_dict)
    agent.eval()
    classifier, tsu = get_classifier(conf), get_tsu(conf)
    helper = Helper(conf, classifier)
    a = (conf.patch_level_low_res, conf.patch_level_high_res, conf.step_size, conf.patch_size)
    pool = ThreadPool(16) if conf.parallel_reads else None
    rule = None
    if conf.policy != 'rl':
        from experiments.sasha.rules import Rule
        rule = Rule(conf.policy, conf.device, conf.rule_seed)
    warm_up(helper.feature_extractor, conf.device, (16, conf.batch_size_lr))

    for slide in names:
        path = os.path.join(out_dir, slide + '.json')
        if os.path.exists(path):
            continue
        t = dict(agent=0.0, hr_read=0.0, hr_embed=0.0, tsu_step=0.0)
        drop_page_cache(os.path.join(conf.source, slide + '.tif'))
        t0 = now()
        helper.create_patches(slide, conf.patch_level_low_res, conf.step_size, conf.patch_size)
        t1 = now()
        feature, coords, wsi = helper.extract_lr_features(slide, 0, conf.slide_ext, *a[:2])
        t2 = now()
        with torch.no_grad():
            state = torch.tensor(feature, dtype=torch.float32, device=conf.device).unsqueeze(0)
            env = WSICosineObservationEnv_inference(lr_features=state, device=conf.device, frac_visit=conf.frac_visit,
                                                    cosine_threshold=conf.cosine_threshold)
            visited, done = [], False
            if rule is not None:
                rule.start(slide, coords, conf.frac_visit)
            while not done:
                s0 = now()
                if rule is None:
                    action, _, _ = agent.get_action(state, visited, is_eval=True)
                    act = action.item()
                else:
                    act = rule.pick(state)
                visited.append(act)
                s1 = now()
                if pool is None:
                    crops = helper.get_patches_for_selected_action(coords[act], wsi, *a)
                else:
                    crops = threaded_crops(helper, pool, coords[act], wsi)
                s2 = now()
                hr = helper.feature_extractor(torch.stack(crops).to(conf.device)).unsqueeze(0)
                v_at = classifier.get_hr_fa(hr)
                s3 = now()
                state, done = env.step(action=act, v_at=v_at, state_update_net=tsu,
                                       classifier_net=classifier, device=conf.device)
                s4 = now()
                for k, d in zip(t, (s1 - s0, s2 - s1, s3 - s2, s4 - s3)):
                    t[k] += d
            t3 = now()
            slide_preds, _ = classifier.classify(state)
        t4 = now()
        prob = torch.softmax(slide_preds.float(), -1).reshape(-1)[1].item()
        t.update(patching=t1 - t0, lr_features=t2 - t1, rl_loop_wall=t3 - t2, classify=t4 - t3, total_wall=t4 - t0)
        json.dump(dict(load1=os.getloadavg()[0], slide=slide, n_tiles=len(coords), n_visited=len(visited), prob_tumour=prob, policy=conf.policy, times=t), open(path, 'w'), indent=1)
        print(f"{slide}: {len(coords)} tiles, {len(visited)} visited, {t['total_wall']:.1f}s", flush=True)


def time_hafed(conf, names, out_dir):
    import openslide
    from datasets.dataset_h5 import Whole_Slide_Bag_FP
    from step7_inference_with_fe import get_classifier
    from utils.inference_utils import Helper

    classifier = get_classifier(conf)
    helper = Helper(conf, classifier)
    vit = helper.feature_extractor
    warm_up(vit, conf.device, (16 * conf.batch_size_hafed, conf.batch_size_hafed))
    if conf.hr_only:
        return time_hafed_hr_only(conf, names, out_dir, helper, classifier)
    for slide in names:
        path = os.path.join(out_dir, slide + '.json')
        if os.path.exists(path):
            continue
        t = dict(loader_startup=0.0, wait_io=0.0, hr_h2d_vit_d2h=0.0, lr_h2d_vit_d2h=0.0)
        diag = dict(read_cpu_sum_hr=0.0, read_cpu_sum_lr=0.0)
        slide_path = os.path.join(conf.source, slide + '.tif')
        drop_page_cache(slide_path)
        t0 = now()
        helper.create_patches(slide, conf.patch_level_low_res, conf.step_size, conf.patch_size)
        t1 = now()
        ds = Whole_Slide_Bag_FP(file_path=os.path.join(helper.step1_save_dir, 'patches', slide + '.h5'),
                                wsi=openslide.open_slide(slide_path), img_transforms=helper.img_transforms,
                                extract_high_res_features=True, patch_level_high_res=conf.patch_level_high_res,
                                patch_level_low_res=conf.patch_level_low_res, dataset_name=conf.dataset)
        loader = DataLoader(dataset=ds, batch_size=conf.batch_size_hafed, num_workers=8, pin_memory=True)
        hr_feats, it = [], iter(loader)
        with torch.inference_mode():
            for k in range(len(loader)):
                s0 = time.perf_counter()
                data = next(it)
                s1 = now()
                hr = data['hr_img']
                f = vit(hr.reshape(-1, *hr.shape[-3:]).to(conf.device, non_blocking=True))
                hr_feats.append(f.reshape(-1, hr.shape[1], f.shape[-1]).cpu())
                s2 = now()
                vit(data['lr_img'].to(conf.device, non_blocking=True)).cpu()
                s3 = now()
                t['loader_startup' if k == 0 else 'wait_io'] += s1 - s0
                diag['read_cpu_sum_hr'] += data['hr_time'].sum().item()
                diag['read_cpu_sum_lr'] += data['lr_time'].sum().item()
                t['hr_h2d_vit_d2h'] += s2 - s1
                t['lr_h2d_vit_d2h'] += s3 - s2
            t2 = now()
            x = torch.cat(hr_feats).unsqueeze(0).to(conf.device)
            out = classifier(x)
        t3 = now()
        prob = torch.softmax(out[1].float(), -1).reshape(-1)[1].item()
        t.update(read_wall=t['loader_startup'] + t['wait_io'], patching=t1 - t0, extract_wall=t2 - t1, hafed_classifier=t3 - t2, total_wall=t3 - t0)
        json.dump(dict(load1=os.getloadavg()[0], slide=slide, n_tiles=len(ds), prob_tumour=prob, times=t, diagnostics=diag), open(path, 'w'), indent=1)
        print(f"{slide}: {len(ds)} tiles, {t['total_wall']:.1f}s", flush=True)


def time_hafed_hr_only(conf, names, out_dir, helper, classifier):
    """HAFED needs only the high-res features (Table 13 of the paper lists no low-res extraction for it), so this
    variant reads and embeds the 16 sub-patches per tile and nothing else, with the authors' transforms."""
    import step12_extract_hr_test_features as S12

    class TimedHR(S12.HRSubPatches):
        def __getitem__(self, i):
            t0 = time.perf_counter()
            if self.wsi is None:
                self.wsi = S12.openslide.OpenSlide(self.wsi_path)
            x, y = (int(v) for v in self.coords[i])
            crops = [helper.img_transforms(self.wsi.read_region((x + xi * 512, y + yi * 512), 1, (256, 256)).convert('RGB'))
                     for xi in range(4) for yi in range(4)]
            return torch.stack(crops), time.perf_counter() - t0

    for slide in names:
        path = os.path.join(out_dir, slide + '.json')
        if os.path.exists(path):
            continue
        t = dict(loader_startup=0.0, wait_io=0.0, hr_h2d_vit_d2h=0.0)
        diag = dict(read_cpu_sum_hr=0.0)
        slide_path = os.path.join(conf.source, slide + '.tif')
        drop_page_cache(slide_path)
        t0 = now()
        helper.create_patches(slide, conf.patch_level_low_res, conf.step_size, conf.patch_size)
        coords = fresh_coords(helper, slide)
        t1 = now()
        loader = DataLoader(TimedHR(slide_path, coords), batch_size=conf.batch_size_hafed, num_workers=8, pin_memory=True)
        hr_feats, it = [], iter(loader)
        with torch.inference_mode():
            for k in range(len(loader)):
                s0 = time.perf_counter()
                hr, tr = next(it)
                s1 = now()
                f = helper.feature_extractor(hr.reshape(-1, *hr.shape[-3:]).to(conf.device, non_blocking=True))
                hr_feats.append(f.reshape(-1, 16, f.shape[-1]).cpu())
                s2 = now()
                t['loader_startup' if k == 0 else 'wait_io'] += s1 - s0
                diag['read_cpu_sum_hr'] += tr.sum().item()
                t['hr_h2d_vit_d2h'] += s2 - s1
            t2 = now()
            out = classifier(torch.cat(hr_feats).unsqueeze(0).to(conf.device))
        t3 = now()
        prob = torch.softmax(out[1].float(), -1).reshape(-1)[1].item()
        t.update(read_wall=t['loader_startup'] + t['wait_io'], patching=t1 - t0, extract_wall=t2 - t1, hafed_classifier=t3 - t2, total_wall=t3 - t0)
        json.dump(dict(load1=os.getloadavg()[0], slide=slide, n_tiles=len(coords), prob_tumour=prob, times=t, diagnostics=diag), open(path, 'w'), indent=1)
        print(f"{slide}: {len(coords)} tiles, {t['total_wall']:.1f}s", flush=True)


def time_model1(conf, names, out_dir):
    import step12_extract_hr_test_features as S12
    from models_features_extraction.builder import vit_small
    from utils.inference_utils import Helper
    sys.path.insert(0, conf.model1_code)
    import predict as P

    class TimedHR(S12.HRSubPatches):
        def __getitem__(self, i):
            t0 = time.perf_counter()
            idx, crops = super().__getitem__(i)
            return idx, crops, time.perf_counter() - t0

    helper = Helper(conf, None)
    helper.feature_extractor.cpu()
    vit = vit_small(pretrained=True, progress=False, key='DINO_p16', patch_size=16).to(conf.device).eval()
    head = P.Model1.from_dir(conf.model1_bundle, str(conf.device))
    warm_up(vit, conf.device, (256,))
    stored = h5py.File(conf.level3_h5, 'r')

    for slide in names:
        path = os.path.join(out_dir, slide + '.json')
        if os.path.exists(path):
            continue
        t = dict(loader_startup=0.0, wait_io=0.0, h2d=0.0, vit=0.0, d2h=0.0)
        diag = dict(read_cpu_sum=0.0)
        drop_page_cache(os.path.join(conf.source, slide + '.tif'))
        t0 = now()
        helper.create_patches(slide, conf.patch_level_low_res, conf.step_size, conf.patch_size)
        coords = fresh_coords(helper, slide)
        t1 = now()
        loader = DataLoader(TimedHR(os.path.join(conf.source, slide + '.tif'), coords), batch_size=16, num_workers=8,
                            pin_memory=True)
        feat = torch.empty(len(coords), 16, 384, dtype=torch.float16)
        it = iter(loader)
        with torch.no_grad():
            for k in range(len(loader)):
                s0 = time.perf_counter()
                idx, imgs, tr = next(it)
                s1 = time.perf_counter()
                x = imgs.reshape(-1, 3, 224, 224).to(conf.device, non_blocking=True)
                s2 = now()
                f = vit(x)
                s3 = now()
                feat[idx] = f.reshape(len(idx), 16, -1).to(torch.float16).cpu()
                s4 = now()
                t['loader_startup' if k == 0 else 'wait_io'] += s1 - s0
                diag['read_cpu_sum'] += tr.sum().item()
                for key, d in zip(('h2d', 'vit', 'd2h'), (s2 - s1, s3 - s2, s4 - s3)):
                    t[key] += d
        t2 = now()
        probs = head.predict_slide(feat.numpy(), coords)
        t3 = now()
        g = (coords // 512).astype(int)
        g -= g.min(0)
        canvas = np.zeros((g[:, 1].max() + 4, g[:, 0].max() + 4), np.float32)
        for k in range(16):
            canvas[g[:, 1] + k % 4, g[:, 0] + k // 4] = probs[:, k]
        lab, n = ndimage.label(canvas >= 0.5)
        if n:
            ndimage.maximum(canvas, lab, np.arange(1, n + 1))
        t4 = now()
        t.update(read_wall=t['loader_startup'] + t['wait_io'], patching=t1 - t0, hr_extract_wall=t2 - t1, model1_head=t3 - t2, postprocess=t4 - t3, total_wall=t4 - t0)
        same = slide in stored and np.array_equal(stored[slide]['coords'][:], coords)
        json.dump(dict(load1=os.getloadavg()[0], slide=slide, n_tiles=len(coords), coords_match_stored=bool(same), times=t, diagnostics=diag), open(path, 'w'), indent=1)
        print(f"{slide}: {len(coords)} tiles, {t['total_wall']:.1f}s, coords match {same}", flush=True)


def summarise(out_dir):
    recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out_dir, 'test_*.json')))]
    total = np.array([r['times']['total_wall'] for r in recs])
    tiles = np.array([r['n_tiles'] for r in recs], float)
    print(f'{out_dir}: {len(recs)} slides, tiles/slide mean {tiles.mean():.0f} median {np.median(tiles):.0f}')
    print(f'total wall/slide: mean {total.mean():.1f}s median {np.median(total):.1f}s min {total.min():.1f} max {total.max():.1f}')
    for k in recs[0]['times']:
        v = np.array([r['times'][k] for r in recs])
        if k.startswith('read_cpu'):
            continue  # legacy jsons: worker-summed CPU-seconds, not wall time
        print(f'{k:16s} mean {v.mean():8.2f}s  median {np.median(v):8.2f}s  {100 * v.sum() / total.sum():6.1f} %')
    if 'diagnostics' in recs[0]:
        for k in recs[0]['diagnostics']:
            print(f'diagnostic {k}: mean {np.mean([r["diagnostics"][k] for r in recs]):.2f} CPU-seconds summed over workers (not wall time)')
    for k in ('n_visited', 'coords_match_stored'):
        if k in recs[0]:
            print(k, float(np.mean([r[k] for r in recs])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('sasha', 'model1', 'hafed', 'tilemlp'), required=True)
    p.add_argument('--config', default='config/camelyon_seg_0.2.yml')
    p.add_argument('--frac_visit', type=float, default=None)
    p.add_argument('--rl_ckpt_path', default=None)
    p.add_argument('--source', default='/data1/namanm/dataset/camelyon16/images/all')
    p.add_argument('--save_dir', required=True)
    p.add_argument('--slides', nargs='*', default=None)
    p.add_argument('--model1_code', default=os.path.join(ROOT, 'patch_cls/model1'))
    p.add_argument('--model1_bundle', default=os.path.expanduser('~/seg_outputs/patch_cls/model1/best'))
    p.add_argument('--level3_h5', default='/data2/tej/SASHA/camelyon_lr_features/patch_feats_pretrain_medical_ssl.h5')
    p.add_argument('--batch_size_hafed', type=int, default=32)
    p.add_argument('--parallel_reads', action='store_true', help='sasha: read the 16 crops of a tile with 16 threads')
    p.add_argument('--policy', default='rl', choices=('rl', 'random', 'greedy', 'uncertain'), help='sasha: who picks the next tile (rl = the authors\' agent, unchanged)')
    p.add_argument('--rule_seed', type=int, default=0, help='sasha with a rule: seed of the per-slide tie-breaking / random draws')
    p.add_argument('--mlp_bundle', default=None, help='tilemlp: directory written by experiments/sasha/tile_mlp.py')
    p.add_argument('--hr_only', action='store_true', help='hafed: skip the low-res reads and features')
    p.add_argument('--summarise', action='store_true')
    a = p.parse_args()
    out_dir = os.path.join(a.save_dir, a.mode if a.mode != 'sasha' or a.policy == 'rl' else f'sasha_{a.policy}')
    if a.summarise:
        return summarise(out_dir)
    cfg = yaml.safe_load(open(a.config))
    cfg.update({k: v for k, v in vars(a).items() if v is not None})
    conf = SimpleNamespace(**cfg, device=torch.device('cuda:0'), step_size=256, patch_size=256, slide_ext='tif',
                           patch_level_low_res=3, patch_level_high_res=1, batch_size_lr=512, batch_size_hr=1,
                           classifier_arch='hafed')
    conf.save_dir = os.path.join(a.save_dir, a.mode + '_work')
    os.makedirs(out_dir, exist_ok=True)
    names = a.slides or sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(a.source, 'test_*.tif')))
    if a.mode == 'tilemlp':
        from experiments.sasha.tile_mlp import time_tilemlp
        time_tilemlp(conf, names, out_dir, now, drop_page_cache, warm_up)
    else:
        dict(sasha=time_sasha, model1=time_model1, hafed=time_hafed)[a.mode](conf, names, out_dir)
    open(os.path.join(out_dir, 'DONE'), 'w').write(f'{len(names)} slides\n')


if __name__ == '__main__':
    main()
