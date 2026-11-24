# Real timed run of the staged policy on the test slides, in the rule's own harness (rlscreen.run_real: same
# reader pools, ViT-L screen, level-2 token head, cold page cache per slide, one slide at a time), so its wall
# time compares with the rule's r4_trace run made by rlcascade.run_real_rl --model rule. The L3 screen of a unit
# feeds both the supervised unit head and the token-level head (the same tokens, one ViT forward); with --tok6 the
# whole-slide L6 token map likewise feeds the second L6 head. Stage 1,
# border tracing, the evidence check and stage 2 are policy.play's, with every read and segmentation done for
# real. Each slide's time json also gets the stage it ended in, the 1-min load and other processes' CPU seconds.
# With --policy the decisions are a PPO policy's (rl3.py) instead of the staged policy's thresholds.
# Usage: python -m experiments.rlpolicy.run_staged (--theta FILE.json | --policy SEARCH@RUN) --tok3 TAG --out_dir DIR [--slides ...]

import argparse
import json
import os
from multiprocessing import get_context
from types import SimpleNamespace

import numpy as np
import torch

from experiments.lowsep.cascade_pipe import read_screen
from experiments.lowsep.extract import tokens
from experiments.lowsep.train import names_of
from experiments.rlcascade.run_real_rl import cpu_of, cpu_total, slide_state
from experiments.rlpolicy import policy as P
from experiments.rlpolicy import sim as S
from experiments.rlpolicy.tok3_predict import load_head
from experiments.rlscreen import run_real as RR
from experiments.rlscreen import stoprl as R

G = 32            # level-2 map cells per unit side
TOPK = ('t_top4', 't_top16', 't_max')
STASH = {}


def on_screen(self, sel):
    """RealEp.on_screen plus the token-level head on the same tokens (its score as the simulator stores it)."""
    pools, m, path, probs, tab = self.real
    units = [tuple(map(int, u)) for u in np.argwhere(sel)]
    if not units:
        return
    items = pools['screen'].map(read_screen, [(path, 3, r, c) for r, c in units])
    s = self.s
    for i in range(0, len(items), 128):
        tk = m.embed([it[2] for it in items[i:i + 128]], m.pca_3)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                v = torch.softmax(m.l3_unit_head(tk).float(), 1)[:, :, 0, 0].cpu().numpy()
            tl, ul = m.tok3(tk.float())
            top = ((torch.sigmoid(tl) * 255).round() / 255).flatten(1).sort(1, descending=True)[0]
            st = torch.stack([top[:, :4].mean(1), top[:, :16].mean(1), top[:, 0], top.mean(1), torch.sigmoid(ul[:, 0]), torch.sigmoid(ul[:, 1])], 1).cpu().numpy()
        for k, (r, c, _) in enumerate(items[i:i + 128]):
            s.p3[r, c], s.probs[r, c] = 1 - v[k, 0], v[k]
            s.p_core[r, c] = v[k, 1] if v[k].argmax() == 1 else 0.0
            s.t_top4[r, c], s.t_top16[r, c], s.t_max[r, c], s.t_mean[r, c], s.t_unit[r, c], s.t_core[r, c] = st[k]
        if getattr(m, 'seg3', None) is not None:      # a second token-level head on the same tokens (rl3 --state ...+seg)
            with torch.no_grad():
                tl, ul = m.seg3(tk.float())
                top = ((torch.sigmoid(tl) * 255).round() / 255).flatten(1).sort(1, descending=True)[0]
                st = torch.stack([top[:, :4].mean(1), top[:, :16].mean(1), top[:, 0], top.mean(1), torch.sigmoid(ul[:, 0]), torch.sigmoid(ul[:, 1])], 1).cpu().numpy()
            for k, (r, c, _) in enumerate(items[i:i + 128]):
                s.s_top4[r, c], s.s_top16[r, c], s.s_max[r, c], s.s_mean[r, c], s.s_unit[r, c], s.s_core[r, c] = st[k]


def wrap_global(orig):
    def f(m, path, level, t5, hu, wu, pool):
        t = np.pad(t5, ((0, t5.shape[0] % 2), (0, t5.shape[1] % 2)))
        tt = t.reshape(t.shape[0] // 2, 2, t.shape[1] // 2, 2).any((1, 3))       # tissue on the level-2 map grid
        STASH['tissue_tok'] = np.pad(tt, ((0, max(0, hu * G - tt.shape[0])), (0, max(0, wu * G - tt.shape[1]))))[:hu * G, :wu * G]
        STASH['t5'] = t5
        return orig(m, path, level, t5, hu, wu, pool)
    return f


def with_verify(fn):
    """Level-1 verification of the routed map (experiments.segfp.verify, the segmenter study's fixed operating
    point) as the last step of a slide's sequence, inside the timed region: the routed level-2 map, zero outside
    tissue, is replaced in place by the verified one. The crops read and the seconds spent go into the time record."""
    def f(a, m, pools, path, tissue, p_g, probs, tab):
        out = fn(a, m, pools, path, tissue, p_g, probs, tab)
        name = os.path.splitext(os.path.basename(path))[0]
        probs *= STASH['tissue_tok']
        v, n, info = m.verify(name, probs, paint=out[0], tissue5=STASH['t5'])
        probs[...] = v
        STASH['verify'] = dict(l1_crops=int(n), verify_s=float(info['seconds']), parts=int(info['parts']), verified=int(info['verified']))
        return out
    return f


class Tap(torch.nn.Module):
    """The supervised L6 head, keeping the token map it is given so the second L6 head can read the same one."""

    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, x):
        STASH['canvas'] = x
        return self.head(x)


def second_l6(m, hu, wu):
    """Per-unit maximum and mean of the second L6 head's token probabilities on the stashed token map (as tok6.unit_scores)."""
    with torch.no_grad():
        p = torch.sigmoid(m.tok6(STASH['canvas'].float()))[0]
    p = torch.nn.functional.pad(p, (0, max(0, 2 * wu - p.shape[1]), 0, max(0, 2 * hu - p.shape[0])))[:2 * hu, :2 * wu].reshape(hu, 2, wu, 2)
    return p.amax((1, 3)).cpu().numpy(), p.mean((1, 3)).cpu().numpy()


def unit_max(probs, units):
    tt = STASH['tissue_tok']
    return max((float((probs[r * G:(r + 1) * G, c * G:(c + 1) * G] * tt[r * G:(r + 1) * G, c * G:(c + 1) * G]).max()) for r, c in np.argwhere(units)), default=0.0)


def segment_traced(io, s, zoom, paint, done):
    """Segments `zoom`, then follows borders as sim.trace does; returns every unit segmented by this call."""
    zoom = zoom & ~paint & ~done
    io.on_zoom(zoom)
    done, got = done | zoom | paint, zoom.copy()
    add = S.follow_from(s, (zoom | (done & ~paint)) & s.pos, paint, done)
    while add.any():
        io.on_zoom(add)
        done, got = done | add, got | add
        add = S.follow_from(s, add & s.pos, np.zeros_like(add), done)
    return got


def new_state(m, tissue, p_g):
    s = slide_state(tissue, p_g)
    for k in ('t_top4', 't_top16', 't_max', 't_mean', 't_unit', 't_core') + (('s_top4', 's_top16', 's_max', 's_mean', 's_unit', 's_core') if getattr(m, 'seg3', None) is not None else ()):
        setattr(s, k, np.zeros(tissue.shape, np.float32))
    if m.tok6 is not None:
        s.g_max, s.g_mean = second_l6(m, *tissue.shape)
        s.g_key = 'g_max'
    return s


def attach_policy(m, spec, tok3, device):
    """Loads a PPO policy (SEARCH@RUN, saved by rl3.py) and the token-level L3 head onto run_real's models and makes
    every L3 screen feed both L3 heads. Returns the saved run."""
    import pickle
    from experiments.rlpolicy import rl3
    from experiments.rlpolicy import where as W
    name, rk = spec.split('@')
    run = pickle.load(open(f'{W.ROOT}/search/{name}/res.pkl', 'rb'))['runs'][rk]
    assert not run['tok6'], 'this loader does not set up the second L6 head'
    m.tok3, m.key, m.tok6, m.run = load_head(tok3, -1, device), 't_top16', None, run
    m.seg3 = load_head(run['seg3'], -1, device) if run.get('seg3') else None
    m.nets = rl3.nets_of(run['nets'])
    RR.RealEp.on_screen = on_screen
    return run


def run_ppo(a, m, pools, path, tissue, p_g, probs, tab):
    """The PPO policy in run_real's harness (level-2 token head behind every zoom)."""
    s = new_state(m, tissue, p_g)
    io = RR.RealEp(s, 1.0)
    io.real = (pools, m, path, probs, tab)
    STASH['stage'] = 1
    return ppo_play(io, s, m)


def ppo_play(io, s, m):
    """The PPO policy's greedy play on any env whose on_screen / on_zoom do the reads and fill the slide state (so
    another segmenter can sit behind on_zoom): the L6 policy picks the L3 reads, the L3 policy leaves / segments /
    paints, then border tracing from what the segmenter returned. -> painted, units segmented, rounds, units screened."""
    from experiments.rlpolicy import rl3
    tissue = s.tissue
    idx = np.flatnonzero(tissue.ravel())
    grid = lambda v: (lambda g: (g.__setitem__(idx, v), g.reshape(tissue.shape))[1])(np.zeros(tissue.size, bool))
    with torch.no_grad():
        scr_t = (m.nets.pi6(rl3.states(s, m.run['state'], m.run['tok6'])[0])[:, 0] >= 0).numpy()
    scr = grid(scr_t)
    io.on_screen(scr)
    with torch.no_grad():
        a3 = m.nets.pi3(rl3.states(s, m.run['state'], m.run['tok6'])[1]).argmax(1).numpy()
    paint = grid(scr_t & (a3 == rl3.PAINT))
    if getattr(m, 'probe', None):
        z, rounds = probe_traced(io, s, grid(scr_t & (a3 == rl3.ZOOM)), paint, *m.probe)
        return paint, int(z.sum()), rounds, int(scr.sum())
    z = segment_traced(io, s, grid(scr_t & (a3 == rl3.ZOOM)), paint, np.zeros_like(tissue))
    return paint, int(z.sum()), 1, int(scr.sum())


def probe_traced(io, s, cand, paint, B, rescue):
    """Probe-and-stop as probe_eval.play does it in the simulator: the zoom candidates in connected clumps, best
    token-head score first, B clumps per round; a round segments the best unit of each of its clumps (the probe),
    their units scoring >= rescue, and the rest of the clumps whose probe returned something in the round before;
    it stops after a round in which no probe returned anything. Then border tracing. -> units segmented, rounds."""
    from scipy import ndimage as nd
    grp, k = nd.label(cand, structure=np.ones((3, 3), bool))
    done = np.zeros(cand.size, bool)
    rounds = 0
    if k:
        flat, sc = grp.ravel(), s.t_top16.ravel()
        idx = np.arange(1, k + 1)
        best = nd.maximum(s.t_top16, grp, idx)
        probe = np.array([p[0] * cand.shape[1] + p[1] for p in nd.maximum_position(s.t_top16, grp, idx)])
        order = np.argsort(-best, kind='stable')
        fill = np.zeros(cand.size, bool)
        j = 0
        while j < k or fill.any():
            b = order[j:j + B] if j < k else order[:0]
            j += B
            now = fill.copy()
            now[probe[b]] = True
            for c in b:
                now |= (flat == c + 1) & (sc >= rescue)
            now &= ~done
            io.on_zoom(now.reshape(cand.shape))
            done |= now
            rounds += 1
            hit = s.pos.ravel()[probe[b]] if len(b) else np.zeros(0, bool)
            fill = np.zeros(cand.size, bool)
            for c in b[hit]:
                fill |= (flat == c + 1) & ~done
            if len(b) and not hit.any():
                break
    done = done.reshape(cand.shape)
    got = segment_traced(io, s, np.zeros_like(cand), paint, done | paint)
    return done | got, max(rounds, 1)


def run_staged(a, m, pools, path, tissue, p_g, probs, tab):
    th = a.theta
    s = new_state(m, tissue, p_g)
    s.t = getattr(s, m.key)
    io = RR.RealEp(s, 1.0)
    io.real = (pools, m, path, probs, tab)
    t3b, p3b, t6b = min(th.t3a, th.t3b), min(th.p3a, th.p3b), min(th.t6a, th.t6b)
    wanted = lambda tt, tp: ((s.t >= tt) | (s.p3 >= tp)) & (s.t >= th.veto)
    none = np.zeros_like(tissue)
    scr = P.screen(s, th.t6a, th)
    io.on_screen(scr)
    paint = scr & (s.p_core >= th.tc) & wanted(t3b, p3b)
    zoom = P.top_k(scr & wanted(th.t3a, th.p3a) & ~paint, np.maximum(s.t / th.t3a, s.p3 / th.p3a), th.k1)
    z1 = segment_traced(io, s, zoom, paint, none)
    STASH['stage'] = 1
    if not (paint.any() or unit_max(probs, z1) >= th.ev):
        return paint, int(z1.sum()), 1, int(scr.sum())
    scr2 = P.screen(s, t6b, th)
    io.on_screen(scr2 & ~scr)
    paint2 = scr2 & (s.p_core >= th.tc) & wanted(t3b, p3b)
    z2 = segment_traced(io, s, scr2 & wanted(t3b, p3b), paint2, z1)
    STASH['stage'] = 2
    return paint2, int((z1 | z2).sum()), 2, int(scr2.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--theta', default=None, help='json with the numbers of policy.Theta and "key"')
    ap.add_argument('--policy', default=None, help='SEARCH@RUN: a PPO policy saved by rl3.py (instead of --theta)')
    ap.add_argument('--tok3', required=True)
    ap.add_argument('--tok6', default=None, help='tag of the second L6 head (needed when theta has veto6 or r6)')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--seg_ckpt', default='/data2/venkatavks/pfa_ckpt/named/tokhead_kaiko_vitl14_stageA_input_l2.pt')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--slides', nargs='*', default=None)
    ap.add_argument('--verify', action='store_true', help='verify the routed map at level 1 at the end of each slide (segfp.verify.Verifier)')
    ap.add_argument('--probe', default=None, help='B:RESCUE with --policy: probe-and-stop on the policy\'s zoom candidates (probe_eval.py)')
    ap.add_argument('--seg_thr', type=float, default=None, help="the L2 head's threshold in the border-tracing feedback (default run_real's 0.4, the stage-A head's)")
    a = ap.parse_args()
    if a.seg_thr is not None:
        RR.SEG_THR = a.seg_thr
    run = None
    if a.policy:
        import pickle
        from experiments.rlpolicy import where as W
        name, rk = a.policy.split('@')
        run = pickle.load(open(f'{W.ROOT}/search/{name}/res.pkl', 'rb'))['runs'][rk]
        cfg, theta, a.tok6 = dict(policy=a.policy, key='t_top16', state=run['state']), P.RULE, run['tok6']
    else:
        cfg = json.load(open(a.theta))
        theta = P.Theta(**{k: cfg[k] for k in P.Theta._fields})
    assert cfg['key'] in TOPK
    ra = SimpleNamespace(design='coarse', g_level=6, backbone='kaiko_vitl14', actor='seq', tag=None, ckpt='final', lam=1.0, tau=0.6, tau_zoom=0.5, follow=1, paint=0.99,
                         h=0.02, stages=None, cluster=0, rule='fixed', minpos=1, minfrac=0.0, mode='last', critic=None, eps=0.02, seg_ckpt=a.seg_ckpt,
                         out_dir=a.out_dir, device=a.device, screen_workers=8, seg_workers=8, model='staged', theta=theta)
    R.CFG.update(design='coarse', tz=0.5, S=4.0, fp=1.0, tcs=(0.99,), cluster=False, follow=True)
    R.build_actions()
    os.makedirs(a.out_dir, exist_ok=True)
    ctx = get_context('fork')   # pools before CUDA; readers never touch the GPU
    pools = {'screen': ctx.Pool(8), 'seg': ctx.Pool(8)}
    mine = [os.getpid()] + [p.pid for pl in pools.values() for p in pl._pool]
    m = RR.Models(ra)
    m.tok3, m.key, m.tok6 = load_head(a.tok3, -1, a.device), cfg['key'], None
    if a.tok6:
        from experiments.rlpolicy import where as W
        from experiments.rlpolicy.tok6 import Tok6
        tc = json.load(open(f'{W.ROOT}/tok6/{a.tok6}/run.json'))
        m.tok6 = Tok6(tc['d'], tc['drop']).to(a.device).eval()
        m.tok6.load_state_dict(torch.load(f'{W.ROOT}/tok6/{a.tok6}/f-1.pt', map_location=a.device))
        m.g_head = Tap(m.g_head)
    assert m.tok6 is not None or (theta.veto6 <= 0 and theta.r6 > 1)
    RR.RealEp.on_screen = on_screen
    RR.global_pass = wrap_global(RR.global_pass)
    RR.run_seq = run_staged
    if run is not None:
        from experiments.rlpolicy import rl3
        m.run = run
        m.probe = (int(a.probe.split(':')[0]), float(a.probe.split(':')[1])) if a.probe else None
        m.seg3 = load_head(run['seg3'], -1, a.device) if run.get('seg3') else None
        m.nets = rl3.nets_of(run['nets'])
        RR.run_seq = run_ppo
    m.verify = None
    if a.verify:
        from experiments.segfp.verify import Verifier
        m.verify = Verifier(device=a.device)
        RR.run_seq = with_verify(RR.run_seq)
        with torch.no_grad():
            m.verify.head(m.verify.net.tokens(torch.zeros(2, 4, 3, 224, 224, dtype=torch.uint8, device=a.device)))
    with torch.no_grad():
        m.seg(torch.zeros(2, 4, 3, 224, 224, dtype=torch.uint8, device=a.device))
        m.tok3(m.embed([np.zeros((3, 224, 224), np.uint8)] * 2, m.pca_3).float())
        if getattr(m, 'seg3', None) is not None:
            m.seg3(m.embed([np.zeros((3, 224, 224), np.uint8)] * 2, m.pca_3).float())
        tokens(m.enc, torch.zeros(2, 3, 224, 224, dtype=torch.uint8, device=a.device))
    torch.cuda.synchronize()
    for n in a.slides or names_of('test', 6):
        out = os.path.join(a.out_dir, n + '.npz')
        if os.path.exists(out):
            continue
        c0, o0 = cpu_total(), cpu_of(mine)
        RR.run_slide(n, ra, m, pools, out)
        tj = out[:-4] + '_time.json'
        rec = json.load(open(tj))
        rec.update(load1=os.getloadavg()[0], foreign_cpu_s=(cpu_total() - c0) - (cpu_of(mine) - o0), model='staged', stage=STASH.get('stage'))
        if a.verify:
            rec.update(STASH.get('verify', {}))
        json.dump(rec, open(tj, 'w'))
    json.dump(dict(cfg, tok3=a.tok3, seg_ckpt=a.seg_ckpt, probe=a.probe, verify=a.verify), open(os.path.join(a.out_dir, 'run.json'), 'w'), indent=1)
    open(os.path.join(a.out_dir, 'DONE'), 'w').write('done\n')


if __name__ == '__main__':
    main()
