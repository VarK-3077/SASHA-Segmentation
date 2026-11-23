# SASHA (sequential reads), SASHA (16-thread reads) and HAFED: per-process timelines of one slide, a 1-s zoom of
# the high-res phase, and a red range bar with the wall time of the high-res reads + feature extraction.
import json, sys, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
D, OUT = sys.argv[1], sys.argv[2]
COL = dict(patching='#999999', lowres_features='#7fb3d5', loader_spawn='#bbbbbb', wait_batch='#f0e6a0', agent='#c39bd3', read_crop='#e67e22',
           read_16_threads='#e67e22', transform_crop='#f5b041', gpu_vit='#1f77b4', tsu='#8e44ad', classify='#2ca02c', head='#2ca02c')
LABEL = dict(patching='patching', lowres_features='low-res features (all tiles)', loader_spawn='loader start-up', wait_batch='GPU idle, waiting for a batch',
             agent='RL agent', read_crop='disk read (one crop)', transform_crop='resize + normalise', gpu_vit='ViT on GPU',
             tsu='TSU update', classify='classifier')
runs = {m: json.load(open(os.path.join(D, f'{m}_test_001.json'))) for m in ('sasha', 'sasha_par', 'hafed')}
title = dict(sasha="SASHA-0.2 (authors' code): one process, everything in sequence", sasha_par='SASHA-0.2 with 16 read threads: reads in parallel, but the GPU still waits for them',
             hafed='HAFED: 8 loader workers read while the GPU computes')
READ = ('read_crop', 'read_16_threads', 'transform_crop')
XMAX = None


def gantt(ax, r, window=None):
    procs = ['main'] + sorted({e[0] for e in r['events'] if e[0] != 'main'}, key=lambda s: int(s.split('_')[1]))
    for e in r['events']:
        if window and (e[3] < window[0] or e[2] > window[1]):
            continue
        ax.barh(procs.index(e[0]), e[3] - e[2], left=e[2], height=0.72, color=COL[e[1]], linewidth=0)
    ax.set_yticks(range(len(procs))); ax.set_yticklabels(['GPU / main' if p == 'main' else p.replace('_', ' ') for p in procs], fontsize=9); ax.invert_yaxis()
    if window:
        ax.set_xlim(window)
    ax.set_xlabel('seconds since slide start')


fig, axes = plt.subplots(3, 3, figsize=(20, 13), gridspec_kw=dict(width_ratios=[1.7, 1, 1], height_ratios=[1, 3.2, 2.4]))
XMAX = max(r['total'] for r in runs.values()) * 1.02
for i, m in enumerate(('sasha', 'sasha_par', 'hafed')):
    r = runs[m]; ev = r['events']
    gantt(axes[i, 0], r, (0, XMAX)); axes[i, 0].set_title(f"{title[m]} — test_001, total {r['total']:.1f} s", loc='left', fontsize=10)
    hi = [e for e in ev if e[1] in READ + ('gpu_vit', 'wait_batch')]
    a0, a1 = min(e[2] for e in hi), max(e[3] for e in hi)
    rd_main = sum(e[3] - e[2] for e in ev if e[1] in READ and e[0] == 'main'); rd_all = sum(e[3] - e[2] for e in ev if e[1] in READ)
    gpu = sum(e[3] - e[2] for e in ev if e[1] == 'gpu_vit'); wait = sum(e[3] - e[2] for e in ev if e[1] == 'wait_batch')
    ax = axes[i, 0]; n = len({e[0] for e in ev}); y = -0.8
    ax.annotate('', xy=(a0, y), xytext=(a1, y), arrowprops=dict(arrowstyle='|-|', color='red', lw=1.4, mutation_scale=4))
    if m == 'sasha':
        detail = f"reads on the main thread {rd_main:.1f} s, GPU {gpu:.1f} s"
    elif m == 'sasha_par':  # wall time of the reads = per tile, first thread start to last thread end
        ag = sorted(e[2] for e in ev if e[1] == 'agent') + [1e9]; th = sorted((e[2], e[3]) for e in ev if e[0] != 'main'); span = 0.0; j = 0
        for a, b in zip(ag, ag[1:]):
            seg = []
            while j < len(th) and th[j][0] < b:
                seg.append(th[j]); j += 1
            if seg:
                span += max(t1 for _, t1 in seg) - min(t0 for t0, _ in seg)
        detail = f"reads in 16 threads {span:.1f} s wall ({rd_all:.1f} CPU-s incl. lock waits), GPU {gpu:.1f} s, never overlapping"
    else:
        detail = f"reads in 8 workers {rd_all:.1f} CPU-s, GPU {gpu:.1f} s, GPU waiting {wait:.1f} s"
    ax.text((a0 + a1) / 2, y - 0.1, f"high-res reads + feature extraction: {a1 - a0:.1f} s   ({detail})", color='red', fontsize=9, ha='center', va='bottom')
    ax.set_ylim(n - 0.5, -1.7)
    hr = [e for e in ev if e[1] == 'gpu_vit' and e[0] == 'main']; mid = hr[len(hr) // 2][2]
    gantt(axes[i, 1], r, (mid, mid + 1.0)); axes[i, 1].set_title('zoom: 1 s of the high-res phase', loc='left', fontsize=10)
    if m != 'hafed':  # one tile: agent -> reads -> ViT -> TSU
        ag = [e for e in ev if e[1] == 'agent']; k = len(ag) // 2; t0 = ag[k][2]; t1 = ag[k + 1][2]
        gantt(axes[i, 2], r, (t0 - 0.002, t1 + 0.002)); axes[i, 2].set_title(f'zoom: one tile ({1e3 * (t1 - t0):.0f} ms): agent, reads, ViT, TSU', loc='left', fontsize=10)
    else:
        gantt(axes[i, 2], r, (mid, mid + 0.12)); axes[i, 2].set_title('zoom: 120 ms', loc='left', fontsize=10)
handles = [Patch(color=COL[k], label=LABEL[k]) for k in LABEL if any(e[1] == k or (k == 'read_crop' and e[1] == 'read_16_threads') for r in runs.values() for e in r['events'])]
fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9, frameon=False)
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(os.path.join(OUT, 'timeline_sasha_vs_hafed.png'), dpi=140)
for m, r in runs.items():
    ev = r['events']
    print(f"{m}: total {r['total']:.1f} s | reads main {sum(e[3]-e[2] for e in ev if e[1] in READ and e[0]=='main'):.1f} s | reads all {sum(e[3]-e[2] for e in ev if e[1] in READ):.1f} CPU-s | GPU {sum(e[3]-e[2] for e in ev if e[1]=='gpu_vit'):.1f} s")
