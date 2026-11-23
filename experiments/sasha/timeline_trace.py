# Event-level trace of one slide's inference for SASHA (authors' sequential loop), HAFED (authors' 8-worker
# loader) and Model-1, with a system-wide monotonic clock so worker-process and main-process events share one
# time axis. Writes <save_dir>/<mode>_<slide>.json: events = [process, kind, start, end], seconds from slide start.
import argparse, json, os, sys, time
from types import SimpleNamespace
import numpy as np, torch, yaml
from torch.utils.data import DataLoader
ROOT = os.path.expanduser('~/SASHA-Segmentation'); sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'experiments/sasha'))
from time_infer import drop_page_cache, fresh_coords, warm_up  # noqa

clock = time.monotonic


def sync():
    torch.cuda.synchronize(); return clock()


def trace_sasha(conf, slide, out, threads=0):
    from multiprocessing.pool import ThreadPool
    import threading
    pool = ThreadPool(threads) if threads else None
    tid_of = {}
    from envs.WSI_cosine_env_inference import WSICosineObservationEnv_inference
    from rl_algorithms.ppo import Actor, Agent, Critic
    from step7_inference_with_fe import get_classifier, get_tsu, load_agent
    from utils.inference_utils import Helper
    agent_dict, agent_cfg = load_agent(conf); ac = SimpleNamespace(**agent_cfg)
    agent = Agent(Actor(conf=ac), Critic(conf=ac), ac).to(conf.device); agent.load_state_dict(agent_dict); agent.eval()
    classifier, tsu = get_classifier(conf), get_tsu(conf); helper = Helper(conf, classifier)
    warm_up(helper.feature_extractor, conf.device, (16, conf.batch_size_lr))
    with torch.no_grad():  # first agent call pays CUDA initialisation (~3 s): not part of the slide
        agent.get_action(torch.zeros(1, 64, 384, device=conf.device), [], is_eval=True)
    drop_page_cache(os.path.join(conf.source, slide + '.tif'))
    ev = []; T0 = sync()
    helper.create_patches(slide, 3, 256, 256); t = sync(); ev.append(['main', 'patching', 0.0, t - T0])
    feature, coords, wsi = helper.extract_lr_features(slide, 0, 'tif', 3, 1); t2 = sync(); ev.append(['main', 'lowres_features', t - T0, t2 - T0])
    with torch.no_grad():
        state = torch.tensor(feature, dtype=torch.float32, device=conf.device).unsqueeze(0)
        env = WSICosineObservationEnv_inference(lr_features=state, device=conf.device, frac_visit=conf.frac_visit, cosine_threshold=conf.cosine_threshold)
        visited, done = [], False
        while not done:
            a0 = sync(); action, _, _ = agent.get_action(state, visited, is_eval=True); visited.append(action.item()); a1 = sync()
            ev.append(['main', 'agent', a0 - T0, a1 - T0])
            x, y = (int(v) for v in coords[action.item()]); crops = []
            pts = [(x + xi * 512, y + yi * 512) for xi in range(4) for yi in range(4)]
            if pool is None:
                for pt in pts:
                    r0 = clock(); patch = wsi.read_region(pt, 1, (256, 256)).convert('RGB'); r1 = clock()
                    crops.append(helper.img_transforms(patch)); r2 = clock()
                    ev.append(['main', 'read_crop', r0 - T0, r1 - T0]); ev.append(['main', 'transform_crop', r1 - T0, r2 - T0])
            else:  # 16 threads, each crop stamped inside its thread (start, after read, after transform)
                def one(pt):
                    r0 = clock(); patch = wsi.read_region(pt, 1, (256, 256)).convert('RGB'); r1 = clock()
                    return helper.img_transforms(patch), threading.get_ident(), r0, r1, clock()
                res = pool.map(one, pts)
                for c, tid, r0, r1, r2 in res:
                    lane = 'thread_%d' % tid_of.setdefault(tid, len(tid_of)); crops.append(c)
                    ev.append([lane, 'read_crop', r0 - T0, r1 - T0]); ev.append([lane, 'transform_crop', r1 - T0, r2 - T0])
            e0 = sync(); hr = helper.feature_extractor(torch.stack(crops).to(conf.device)).unsqueeze(0); v_at = classifier.get_hr_fa(hr); e1 = sync()
            ev.append(['main', 'gpu_vit', e0 - T0, e1 - T0])
            state, done = env.step(action=action.item(), v_at=v_at, state_update_net=tsu, classifier_net=classifier, device=conf.device); s1 = sync()
            ev.append(['main', 'tsu', e1 - T0, s1 - T0])
        c0 = sync(); classifier.classify(state); c1 = sync(); ev.append(['main', 'classify', c0 - T0, c1 - T0])
    json.dump(dict(mode='sasha_par' if threads else 'sasha', slide=slide, n_tiles=len(coords), n_visited=len(visited), total=c1 - T0, events=ev), open(out, 'w'))


class TracedHR(torch.utils.data.Dataset):
    """16 high-res crops per tile, each read timed inside the worker with the shared monotonic clock."""
    def __init__(self, path, coords, transform, T0):
        self.path, self.coords, self.transform, self.T0, self.wsi = path, coords, transform, T0, None

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        import openslide
        if self.wsi is None:
            self.wsi = openslide.OpenSlide(self.path)
        x, y = (int(v) for v in self.coords[i]); crops, stamps = [], []
        for xi in range(4):
            for yi in range(4):
                r0 = clock(); p = self.wsi.read_region((x + xi * 512, y + yi * 512), 1, (256, 256)).convert('RGB'); r1 = clock()
                crops.append(self.transform(p)); stamps.append((os.getpid(), r0 - self.T0, r1 - self.T0, clock() - self.T0))
        return torch.stack(crops), torch.tensor(stamps, dtype=torch.float64)


def trace_loader(conf, slide, out, mode):
    from step7_inference_with_fe import get_classifier
    from utils.inference_utils import Helper
    classifier = get_classifier(conf); helper = Helper(conf, classifier); vit = helper.feature_extractor
    head = None
    if mode == 'model1':
        sys.path.insert(0, os.path.join(ROOT, 'patch_cls/model1')); import predict as P
        head = P.Model1.from_dir(os.path.expanduser('~/seg_outputs/patch_cls/model1/best'), str(conf.device))
    bs = 32 if mode == 'hafed' else 16
    warm_up(vit, conf.device, (16 * bs,))
    path = os.path.join(conf.source, slide + '.tif'); drop_page_cache(path)
    ev = []; T0 = sync()
    helper.create_patches(slide, 3, 256, 256); coords = fresh_coords(helper, slide); t = sync(); ev.append(['main', 'patching', 0.0, t - T0])
    loader = DataLoader(TracedHR(path, coords, helper.img_transforms, T0), batch_size=bs, num_workers=8, pin_memory=True)
    feats = []; it = iter(loader); l0 = clock(); ev.append(['main', 'loader_spawn', t - T0, clock() - T0])
    with torch.inference_mode():
        for k in range(len(loader)):
            w0 = clock(); hr, stamps = next(it); w1 = clock(); ev.append(['main', 'wait_batch', w0 - T0, w1 - T0])
            for row in stamps.reshape(-1, 4).tolist():
                ev.append([f'worker_{int(row[0])}', 'read_crop', row[1], row[2]]); ev.append([f'worker_{int(row[0])}', 'transform_crop', row[2], row[3]])
            g0 = sync(); f = vit(hr.reshape(-1, *hr.shape[-3:]).to(conf.device, non_blocking=True)); feats.append(f.reshape(-1, 16, f.shape[-1]).cpu()); g1 = sync()
            ev.append(['main', 'gpu_vit', g0 - T0, g1 - T0])
        h0 = sync()
        if mode == 'hafed':
            classifier(torch.cat(feats).unsqueeze(0).to(conf.device))
        else:
            head.predict_slide(torch.cat(feats).numpy(), coords)
        h1 = sync(); ev.append(['main', 'head', h0 - T0, h1 - T0])
    pids = sorted({e[0] for e in ev if e[0].startswith('worker')})
    for e in ev:  # worker pids -> worker_0..7 in first-seen order
        if e[0].startswith('worker'):
            e[0] = f'worker_{pids.index(e[0])}'
    json.dump(dict(mode=mode, slide=slide, n_tiles=len(coords), total=h1 - T0, events=ev), open(out, 'w'))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--mode', choices=('sasha', 'sasha_par', 'hafed', 'model1'), required=True)
    p.add_argument('--slide', default='test_001'); p.add_argument('--save_dir', required=True)
    p.add_argument('--config', default='config/camelyon_seg_0.2.yml'); a = p.parse_args()
    cfg = yaml.safe_load(open(a.config))
    conf = SimpleNamespace(**cfg, device=torch.device('cuda:0'), source='/data1/namanm/dataset/camelyon16/images/all', batch_size_lr=512, batch_size_hr=1,
                           slide_ext='tif', step_size=256, patch_size=256, patch_level_low_res=3, patch_level_high_res=1, classifier_arch='hafed',
                           save_dir=os.path.join(a.save_dir, a.mode + '_work'))
    os.makedirs(a.save_dir, exist_ok=True); out = os.path.join(a.save_dir, f'{a.mode}_{a.slide}.json')
    if a.mode == 'sasha':
        trace_sasha(conf, a.slide, out)
    elif a.mode == 'sasha_par':
        trace_sasha(conf, a.slide, out, threads=16)
    else:
        trace_loader(conf, a.slide, out, a.mode)
    print('wrote', out, flush=True)


if __name__ == '__main__':
    main()
