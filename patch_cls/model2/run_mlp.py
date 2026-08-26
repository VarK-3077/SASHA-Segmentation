# GPU MLP sweeps for Model-2.
#
# All three training row-sources (deterministic rollout states, stochastic-rollout states,
# true aggregates) are concatenated once into a single GPU-resident pool tagged by source,
# so a config picks its training distribution just by naming sources -- no data is copied
# per run. Feature views are assembled per batch from the cached blocks.
#
# Every config is early-stopped on the clean validation CE measured on DETERMINISTIC
# ROLLOUT-STATE inputs, and the same model is additionally scored on true-aggregate inputs
# so the two-way transfer is visible for free.

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

import common as C
import views as V

DEV = os.environ.get("M2_DEV", "cuda:0")
SRC = ("tr_det", "tr_aug", "tr_agg")
gpu_view = V.make_view_t


# ---------------------------------------------------------------- data

def to_dev(a, dev, dtype=None):
    t = torch.from_numpy(np.ascontiguousarray(a))
    return t.to(dev, dtype=dtype) if dtype else t.to(dev)


def pack(d, dev, src_id=0):
    """One cache -> the GPU tensors the view builder needs."""
    return dict(
        X=to_dev(d["X"], dev), N=to_dev(d["N"], dev), M=to_dev(d["M"], dev),
        K=to_dev(d["K"], dev), V=to_dev(d["V"], dev),
        Sm=to_dev(d["Sm"], dev), slide=to_dev(d["slide"].astype(np.int64), dev),
        frac=d["frac"], ne=d["ne"], names=d["slide_names"],
        src=torch.full((len(d["frac"]),), src_id, dtype=torch.int8, device=dev),
    )


def load_train_pool(dev):
    """Concatenate the three train sources into one pool, remembering which row is which."""
    parts = [C.load_cache(t) for t in SRC]
    for p in parts[1:]:
        assert (p["slide_names"] == parts[0]["slide_names"]).all()
    pool = {}
    for k in ("X", "N", "M", "K", "V"):
        pool[k] = to_dev(np.concatenate([p[k] for p in parts]), dev)
    # every source brings its own slide-mean rows (one per slide per rollout), so the
    # concatenated slide index has to be offset by however many rows each source added,
    # not by the slide count
    pool["Sm"] = to_dev(np.concatenate([p["Sm"] for p in parts]), dev)
    off, sl = 0, []
    for p in parts:
        sl.append(p["slide"].astype(np.int64) + off)
        off += p["Sm"].shape[0]
    pool["slide"] = to_dev(np.concatenate(sl), dev)
    pool["src"] = to_dev(np.concatenate(
        [np.full(len(p["frac"]), i, dtype=np.int8) for i, p in enumerate(parts)]), dev)
    pool["frac"] = np.concatenate([p["frac"] for p in parts])
    pool["y"] = to_dev(C.labels_of(pool["frac"]).astype(np.float32), dev)
    pool["n_per_src"] = [len(p["frac"]) for p in parts]
    return pool


def gather(P, j, dev):
    x = P["X"][j].float()
    n = P["N"][j].float()
    m = P["M"][j].float()
    k = P["K"][j].float()
    s = P["Sm"][P["slide"][j]].float()
    v = P["V"][j].float()
    return x, n, m, k, s, v


# ---------------------------------------------------------------- stats / predict

def view_stats(P, mode, dev, idx, bs=50_000):
    n = idx.numel()
    d = C.view_dim(mode)
    acc = torch.zeros(d, dtype=torch.float64, device=dev)
    acc2 = torch.zeros(d, dtype=torch.float64, device=dev)
    for i in range(0, n, bs):
        v = gpu_view(*gather(P, idx[i:i + bs], dev), mode).double()
        acc += v.sum(0)
        acc2 += (v * v).sum(0)
    mu = acc / n
    sd = (acc2 / n - mu * mu).clamp_min(1e-12).sqrt()
    sd[sd < 1e-6] = 1.0
    return mu.float(), sd.float()


@torch.no_grad()
def predict_logits(model, P, mode, mu, sd, dev, bs=25_000):
    model.eval()
    n = P["X"].shape[0]
    out = []
    for i in range(0, n, bs):
        j = torch.arange(i, min(i + bs, n), device=dev)
        v = gpu_view(*gather(P, j, dev), mode)
        out.append(model((v - mu) / sd).float())
    model.train()
    return torch.cat(out)


# ---------------------------------------------------------------- training

def train_one(cfg, pool, evals, log=print):
    t0 = time.time()
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    mode = cfg.get("view", "L2:x+n+m+k")
    dev = torch.device(DEV)

    want = [SRC.index(s if s.startswith("tr_") else "tr_" + s) for s in cfg["rows"]]
    sel = torch.zeros_like(pool["src"], dtype=torch.bool)
    for w in want:
        sel |= pool["src"] == w
    # a different positive threshold is a different label policy, not a tuning knob: the
    # strip between 0 and the threshold has to leave training too, not just flip label
    thr = cfg.get("pos_thr", C.POS_THR)
    y_pool = pool["y"]
    if thr != C.POS_THR:
        f = pool["frac"]
        sel &= to_dev((f >= thr) | (f == 0.0), pool["src"].device)
        y_pool = to_dev(C.labels_of(f, thr).astype(np.float32), pool["src"].device)
    idx_pool = torch.nonzero(sel).squeeze(1)
    n = idx_pool.numel()

    init = cfg.get("init_from")
    if init and cfg.get("init_stats", True):
        ck = torch.load(f"{C.CKPT}/{init}.pt", map_location=dev, weights_only=False)
        mu, sd = ck["mu"].to(dev), ck["sd"].to(dev)
        assert ck["view"] == mode, f"init view {ck['view']} != {mode}"
    else:
        mu, sd = view_stats(pool, mode, dev, idx_pool)

    model = V.MLP(C.view_dim(mode), cfg.get("hidden", [1024, 512]),
                  cfg.get("dropout", 0.3), cfg.get("bn", True)).to(dev)
    if cfg.get("prior_init"):
        # AdamW moves a parameter by at most ~lr per step, so at lr 3e-4 a shallow model
        # cannot travel the ~3.6 logits from p=0.5 to the 2.7% base rate inside a short
        # run -- it converges in ranking and stays wildly miscalibrated. Start the output
        # bias at the training prior instead of fighting the optimiser for it.
        p0 = float(y_pool[idx_pool].mean().clamp(1e-6, 1 - 1e-6))
        with torch.no_grad():
            model.net[-1].bias.fill_(float(np.log(p0 / (1 - p0))))
    if init:
        ck = torch.load(f"{C.CKPT}/{init}.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["state"])

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 3e-4),
                            weight_decay=cfg.get("wd", 1e-4))
    epochs, bs = cfg.get("epochs", 30), cfg.get("bs", 4096)
    spe = max(1, n // bs)
    total = epochs * spe
    ev = cfg.get("eval_every", max(5, spe // 2))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    patience = cfg.get("patience", 20)

    sel_key = cfg.get("select_on", "va_det")
    E = evals[sel_key]
    y_sel, m_sel = (E["y"], E["clean"]) if thr == C.POS_THR else \
        (C.labels_of(E["raw"]["frac"], thr), C.clean_mask(E["raw"]["frac"], E["raw"]["ne"], thr))
    best = dict(ce=1e9, step=-1, state=None, p={})
    perm = idx_pool[torch.randperm(n, device=dev)]
    cur, run, nrun, stop = 0, 0.0, 0, False
    for step in range(1, total + 1):
        if cur + bs > n:
            perm = idx_pool[torch.randperm(n, device=dev)]
            cur = 0
        b = perm[cur:cur + bs]
        cur += bs
        yb = y_pool[b]
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            v = (gpu_view(*gather(pool, b, dev), mode) - mu) / sd
            logit = model(v)
        loss = nn.functional.binary_cross_entropy_with_logits(logit.float(), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run += loss.item()
        nrun += 1

        if step % ev == 0 or step == total:
            ps = {}
            for tag, E in evals.items():
                ps[tag] = torch.sigmoid(predict_logits(model, E["P"], mode, mu, sd, dev)) \
                    .cpu().numpy().astype(np.float64)
            ce = C.bce(ps[sel_key][m_sel], y_sel[m_sel])
            good = ce < best["ce"] - 1e-6
            if good:
                best = dict(ce=ce, step=step,
                            state={k: v.detach().clone() for k, v in model.state_dict().items()},
                            p=ps)
            log(f"    step{step:6d} ({step/spe:6.2f}ep) loss={run/nrun:.5f} "
                f"val_ce={ce:.5f}{'  *' if good else ''}")
            run, nrun = 0.0, 0
            if (step - best["step"]) // ev >= patience:
                stop = True
        if stop:
            break

    model.load_state_dict(best["state"])
    return dict(model=model, mu=mu, sd=sd, mode=mode, p=best["p"], n_rows=n, thr=thr,
                train_s=time.time() - t0, best_ep=round(best["step"] / spe, 3))


# ---------------------------------------------------------------- driver

def load_evals(dev, tags=("va_det", "va_agg")):
    out = {}
    for t in tags:
        d = C.load_cache(t)
        out[t] = dict(P=pack(d, dev), raw=d, y=C.labels_of(d["frac"]),
                      clean=C.clean_mask(d["frac"], d["ne"]))
    return out


def metrics_of(r, evals):
    thr = r.get("thr", C.POS_THR)
    met = C.eval_set(r["p"]["va_det"], evals["va_det"]["raw"], thr=thr)
    met.update(C.eval_set(r["p"]["va_agg"], evals["va_agg"]["raw"], thr=thr, prefix="agg_"))
    met["best_epoch"] = r["best_ep"]
    met["n_rows"] = r["n_rows"]
    met["pos_thr"] = thr
    return met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    a = ap.parse_args()
    os.makedirs(C.PRED, exist_ok=True)
    os.makedirs(C.CKPT, exist_ok=True)
    dev = torch.device(DEV)
    cfgs = json.load(open(a.sweep))
    pool = load_train_pool(dev)
    evals = load_evals(dev)
    print(f"pool rows per source {dict(zip(SRC, pool['n_per_src']))} | "
          f"val_det n={len(evals['va_det']['y'])} clean={int(evals['va_det']['clean'].sum())} "
          f"| device {DEV}", flush=True)
    for cfg in cfgs:
        name = cfg["name"]
        print(f"\n=== {name} :: {json.dumps(cfg)}", flush=True)
        r = train_one(cfg, pool, evals, log=lambda s: print(s, flush=True))
        met = metrics_of(r, evals)
        C.log_result(name, "mlp", cfg, met, r["train_s"])
        for tag, p in r["p"].items():
            np.save(f"{C.PRED}/{name}_{tag}.npy", p.astype(np.float32))
        torch.save(dict(state=r["model"].state_dict(), mu=r["mu"].cpu(), sd=r["sd"].cpu(),
                        view=r["mode"], cfg=cfg), f"{C.CKPT}/{name}.pt")


if __name__ == "__main__":
    main()
