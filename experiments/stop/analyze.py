# Reads the rollout curves and answers the ceiling question: for each cost rate lambda, how
# much does a per-slide oracle stop gain over the best single budget and over a simple
# uncertainty-threshold stop? Budgets and thresholds are chosen on the train rollouts and
# scored on test, the oracle uses the test labels by construction.

import glob
import json
import os

import numpy as np

from experiments.stop import paths as P

LAMBDAS = (0.0, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3)
BUDGETS = (0, .01, .02, .03, .05, .07, .1, .15, .2, .3, .5, .75, 1.0)
TAUS = np.concatenate([[0], np.geomspace(1e-4, 0.69, 40)])
MAIN_ORDERS = ('uncertain', 'adaptive')


def load(split):
    return {os.path.basename(f)[:-4]: dict(np.load(f)) for f in sorted(glob.glob(os.path.join(P.ROLL_DIR, split, '*.npz')))}


def slides_with(rolls, variant):
    return [r for r in rolls.values() if f'{variant}__n' in r]


def J(dice, lam):
    return dice - lam * np.arange(len(dice))


def oracle(rs, variant, order, lam):
    js, ds, ks = [], [], []
    for r in rs:
        j = J(r[f'{variant}__{order}__dice'], lam)
        k = int(np.argmax(j))
        js.append(j[k]); ds.append(r[f'{variant}__{order}__dice'][k]); ks.append(k / max(len(j) - 1, 1))
    return float(np.mean(js)), float(np.mean(ds)), float(np.mean(ks)), float(np.std(ks))


def fixed(rs, variant, order, lam, b):
    return float(np.mean([J(r[f'{variant}__{order}__dice'], lam)[int(round(b * (len(r[f'{variant}__{order}__dice']) - 1)))] for r in rs]))


def thresholded(rs, variant, order, lam, tau):
    out = []
    for r in rs:
        ent = r[f'{variant}__{order}__ent']
        below = np.flatnonzero(ent < tau)
        k = int(below[0]) if len(below) else len(ent)
        out.append(J(r[f'{variant}__{order}__dice'], lam)[k])
    return float(np.mean(out))


def analyse(train, test, variant, order):
    tr, te = slides_with(train, variant), slides_with(test, variant)
    rows = []
    for lam in LAMBDAS:
        o_j, o_d, k_mean, k_std = oracle(te, variant, order, lam)
        b = max(BUDGETS, key=lambda b: fixed(tr, variant, order, lam, b))
        tau = max(TAUS, key=lambda t: thresholded(tr, variant, order, lam, t))
        f_j, t_j = fixed(te, variant, order, lam, b), thresholded(te, variant, order, lam, tau)
        f_best = max(fixed(te, variant, order, lam, bb) for bb in BUDGETS)
        rows.append({'lambda': lam, 'oracle_J': o_j, 'oracle_dice': o_d, 'oracle_k_frac': k_mean, 'oracle_k_frac_std': k_std,
                     'fixed_J': f_j, 'fixed_b': b, 'fixed_J_test_opt': f_best, 'thr_J': t_j, 'thr_tau': float(tau),
                     'full_J': fixed(te, variant, order, lam, 1.0), 'none_J': fixed(te, variant, order, lam, 0.0)})
    return {'n_train': len(tr), 'n_test': len(te), 'rows': rows}


def md_table(res):
    h = '| λ | oracle J | Dice at oracle stop | oracle k/N (mean ± std) | fixed J (b) | fixed J, test-opt b | threshold J (τ) | gap vs fixed | gap vs threshold | J all low-res | J all high-res |'
    out = [h, '|' + '---|' * 11]
    for r in res['rows']:
        out.append(f"| {r['lambda']:g} | {r['oracle_J']:.3f} | {r['oracle_dice']:.3f} | {r['oracle_k_frac']:.2f} ± {r['oracle_k_frac_std']:.2f} | "
                   f"{r['fixed_J']:.3f} ({r['fixed_b']:g}) | {r['fixed_J_test_opt']:.3f} | {r['thr_J']:.3f} ({r['thr_tau']:.3g}) | "
                   f"{r['oracle_J'] - r['fixed_J']:+.3f} | {r['oracle_J'] - r['thr_J']:+.3f} | {r['none_J']:.3f} | {r['full_J']:.3f} |")
    return '\n'.join(out)


def main():
    train, test = load('train'), load('test')
    tumour_test = {k: v for k, v in test.items() if v['real__n_tumour'] > 0}
    tumour_train = {k: v for k, v in train.items() if v['real__n_tumour'] > 0}
    sets = {'test_all_real': (train, test, 'real'), 'test_tumour_real': (tumour_train, tumour_test, 'real')}
    for f in P.TARGET_FRACS:
        sets[f'test_tumour_f{f}'] = (tumour_train, tumour_test, f'f{f}')
    results, lines = {}, ['# Stopping ceiling', '',
                          'J = Dice − λ·(tiles visited). Oracle = per-slide argmax of J with test labels. Fixed budget b and threshold τ',
                          'are chosen on train rollouts, scored on test. Dice is over 512 px sub-patches (16 per tile), unvisited tiles',
                          'scored by the low-res head, visited tiles by the high-res head.', '']
    for sname, (tr, te, variant) in sets.items():
        results[sname] = {}
        for order in MAIN_ORDERS:
            res = analyse(tr, te, variant, order)
            results[sname][order] = res
            lines += [f'## {sname} — visit order: {order} (train {res["n_train"]}, test {res["n_test"]} slides)', '', md_table(res), '']
    # order comparison at one lambda
    lam = 1e-4
    lines += [f'## visit orders compared at λ = {lam:g}, real test tumour slides: oracle J / Dice / k/N', '',
              '| order | oracle J | Dice | k/N |', '|---|---|---|---|']
    for order in P.ORDERS:
        o_j, o_d, k, _ = oracle(list(tumour_test.values()), 'real', order, lam)
        lines.append(f'| {order} | {o_j:.3f} | {o_d:.3f} | {k:.2f} |')
    # oracle stop vs tumour fraction
    lines += ['', f'## oracle stop fraction by real tumour-tile fraction, adaptive order, λ = {lam:g} (test tumour slides)', '',
              '| tumour-tile fraction | slides | oracle k/N mean | std |', '|---|---|---|---|']
    fr = np.array([r['real__n_tumour'] / r['real__n'] for r in tumour_test.values()])
    ks = np.array([np.argmax(J(r['real__adaptive__dice'], lam)) / (r['real__n'] - 1) for r in tumour_test.values()])
    for lo, hi in ((0, .02), (.02, .05), (.05, .1), (.1, .2), (.2, 1)):
        m = (fr >= lo) & (fr < hi)
        if m.any():
            lines.append(f'| {lo:.2f}–{hi:.2f} | {m.sum()} | {ks[m].mean():.2f} | {ks[m].std():.2f} |')
    text = '\n'.join(lines)
    open(os.path.join(P.ROOT, 'report.md'), 'w').write(text)
    json.dump(results, open(os.path.join(P.ROOT, 'results.json'), 'w'), indent=1)
    print(text)


if __name__ == '__main__':
    main()
