# Can more core units be painted instead of segmented? Train, simulator. Base policy + probe-and-stop as picked;
# then every zoom candidate whose L3 evidence passes a paint rule is painted instead (painting is free and pulls in
# its four neighbours through tracing). Reports units segmented / painted, how many painted units are really core.
import sys
import numpy as np
from experiments.rlpolicy import rl4, sim as S, where as W, probe_eval as PE

sl = S.load('train', 't1')
for s in sl:
    s.add_tok3('s2', 's')
rule = [S.rule(s) for s in sl]
tum = [i for i, s in enumerate(sl) if s.tumour]
fmt = lambda m: f"{m['sec']:.3f} s  Dice {m['dice']:.4f}  FROC {m['froc']:.4f}  L2 {m['l2']:.1f}  paint {m['paint']:.1f}  FP {m['fp']:.2f}"
lost = lambda recs: sum(max(0, rule[i]['found'] - recs[i]['found']) for i in tum)
RULES = (('as trained', None), ('p_core >= 0.9', lambda s: s.p_core >= 0.9), ('p_core >= 0.5', lambda s: s.p_core >= 0.5), ('t_mean >= 0.9', lambda s: s.t_mean >= 0.9),
         ('t_mean >= 0.8', lambda s: s.t_mean >= 0.8), ('t_mean >= 0.7', lambda s: s.t_mean >= 0.7), ('t_mean >= 0.6', lambda s: s.t_mean >= 0.6),
         ('t_mean >= 0.8 and p_core >= 0.5', lambda s: (s.t_mean >= 0.8) & (s.p_core >= 0.5)), ('s_mean >= 0.8', lambda s: s.s_mean >= 0.8))
for pick in sys.argv[1:]:
    base, B, rescue = pick.rsplit(':', 2)
    acts = rl4.base_actions(sl, base)
    core_all = sum(int(((s.gt_tis >= 0.9 * s.n_tis) & (s.gt_tis > 0) & s.tissue).sum()) for s in sl) / len(sl)
    print(f'--- {pick}   (units that are >= 90 % tumour: {core_all:.1f}/slide)', flush=True)
    for name, fn in RULES:
        recs, n_p, n_core, n_half = [], 0, 0, 0
        for s, (screen, cand, paint) in zip(sl, acts):
            if fn is not None:
                more = cand & fn(s)
                cand, paint = cand & ~more, paint | more
            r = PE.play(s, screen, cand, paint, int(B), float(rescue))
            recs.append(r)
            p = r['paint']
            n_p += int(p.sum()); n_core += int((p & (s.gt_tis >= 0.9 * s.n_tis) & (s.gt_tis > 0)).sum()); n_half += int((p & (s.gt_tis >= 0.5 * s.n_tis) & (s.gt_tis > 0)).sum())
        print(f'  paint also if {name:32s} {fmt(S.summarise(recs))}  lost {lost(recs)}  | painted units really >= 90 % tumour: {100 * n_core / max(n_p, 1):.0f} %, >= 50 %: {100 * n_half / max(n_p, 1):.0f} %'
              f'  | L2 on tumour slides {np.mean([recs[i]["n_l2"] for i in tum]):.1f}', flush=True)
