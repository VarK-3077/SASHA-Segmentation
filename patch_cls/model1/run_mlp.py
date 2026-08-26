# GPU MLP sweeps for Model-1.
#
# Everything lives on the GPU as fp16 and feature views are assembled per batch, so a
# whole sweep of configs shares one data load. Each config is early-stopped on the clean
# validation cross-entropy AFTER the prior-shift correction that undoes whatever class
# re-balancing it trained with -- that correction is a deterministic logit offset, so
# applying it during selection is honest and it is what makes weighted/subsampled runs
# comparable to plain unweighted ones on an unweighted CE.

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

import common as C
import views as V

DEV = os.environ.get("M1_DEV", "cuda:1")
# The feature blocks are ~2 GB each and the GPUs are shared, so they live in host RAM and
# batches are copied over per step. Set M1_DATA_DEV=cuda:N to keep them resident instead.
DATA_DEV = os.environ.get("M1_DATA_DEV", "cpu")
PRED = f"{C.OUT}/preds"
CKPT = f"{C.OUT}/ckpt"


# ---------------------------------------------------------------- feature views

gpu_view = V.make_view_t


def gather(blocks, j, dev):
    """Pull rows j out of the six CPU-resident feature blocks and land them on dev."""
    return [b[j].to(dev, non_blocking=True).float() for b in blocks]


def view_stats(blocks, mode, dev, idx=None, bs=50_000):
    n = blocks[0].shape[0] if idx is None else idx.numel()
    d = C.view_dim(mode)
    acc = torch.zeros(d, dtype=torch.float64, device=dev)
    acc2 = torch.zeros(d, dtype=torch.float64, device=dev)
    for i in range(0, n, bs):
        j = slice(i, i + bs) if idx is None else idx[i:i + bs]
        v = gpu_view(*gather(blocks, j, dev), mode).double()
        acc += v.sum(0)
        acc2 += (v * v).sum(0)
    mu = acc / n
    sd = (acc2 / n - mu * mu).clamp_min(1e-12).sqrt()
    sd[sd < 1e-6] = 1.0
    return mu.float(), sd.float()


# ---------------------------------------------------------------- model

MLP = V.MLP


def focal_bce(logit, y, gamma, alpha):
    p = torch.sigmoid(logit)
    pt = torch.where(y > 0.5, p, 1 - p)
    w = (1 - pt).pow(gamma)
    if alpha is not None:
        w = w * torch.where(y > 0.5, torch.full_like(y, alpha), torch.full_like(y, 1 - alpha))
    return (w * nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none")).mean()


# ---------------------------------------------------------------- training

def predict_logits(model, blocks, mode, mu, sd, dev, bs=50_000):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, blocks[0].shape[0], bs):
            v = gpu_view(*gather(blocks, slice(i, i + bs), dev), mode)
            out.append(model((v - mu) / sd).float())
    model.train()
    return torch.cat(out)


def train_one(cfg, D, val, log=print):
    t0 = time.time()
    torch.manual_seed(cfg.get("seed", 0))
    np.random.seed(cfg.get("seed", 0))
    mode = cfg.get("view", "raw")
    dev = torch.device(DEV)
    blocks, yg = D["blocks"], D["yg"]

    # --- optional negative subsampling (fresh draw per run, fixed across epochs)
    q = cfg.get("neg_keep", 1.0)
    if q < 1.0:
        g = torch.Generator(device="cpu").manual_seed(cfg.get("seed", 0) + 991)
        keep = (yg > 0.5) | (torch.rand(len(yg), generator=g) < q)
        idx_pool = torch.nonzero(keep).squeeze(1)
    else:
        idx_pool = torch.arange(blocks[0].shape[0])
    n = idx_pool.numel()

    mu, sd = view_stats(blocks, mode, dev, idx=idx_pool if q < 1.0 else None)

    # --- prior shift that maps training-time odds back to the true prior
    shift = 0.0
    pw = None
    if cfg.get("loss") == "weighted":
        pw = cfg.get("pos_weight")
        if pw in (None, "auto"):
            ys = yg[idx_pool]
            pw = float((ys < 0.5).sum() / (ys > 0.5).sum())
        shift -= float(np.log(pw))
    if q < 1.0:
        shift += float(np.log(q))

    model = MLP(C.view_dim(mode), cfg.get("hidden", [512, 256]),
                cfg.get("dropout", 0.1), cfg.get("bn", True)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-3),
                            weight_decay=cfg.get("wd", 1e-4))
    epochs, bs = cfg.get("epochs", 30), cfg.get("bs", 4096)
    spe = max(1, n // bs)                       # steps per epoch
    total = epochs * spe
    ev = cfg.get("eval_every", max(1, spe // 4))  # validate 4x per epoch by default
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    ls = cfg.get("label_smooth", 0.0)
    noise = cfg.get("in_noise", 0.0)
    in_drop = cfg.get("in_drop", 0.0)
    mixup = cfg.get("mixup", 0.0)

    vblocks = val["blocks"]
    yv_all, m = val["y_all"], val["clean"]
    best = dict(ce=1e9, step=-1, state=None, p=None)
    patience = cfg.get("patience", 8)           # in eval intervals
    pw_t = torch.tensor(pw, device=dev) if pw else None

    perm = idx_pool[torch.randperm(n)]
    cur, run, nrun, stop = 0, 0.0, 0, False
    for step in range(1, total + 1):
        if cur + bs > n:
            perm = idx_pool[torch.randperm(n)]
            cur = 0
        b = perm[cur:cur + bs]
        cur += bs
        xb, cb, sb, nb, mb, kb = gather(blocks, b, dev)
        yb = yg[b].to(dev, non_blocking=True).float()
        if ls > 0:
            yb = yb * (1 - ls) + ls * 0.5
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            v = (gpu_view(xb, cb, sb, nb, mb, kb, mode) - mu) / sd
            # sub-patches inside a slide are heavily correlated, so the effective sample
            # size is closer to the slide count than the row count; input noise / mixup
            # are the cheap ways to stop the net memorising slide appearance in one epoch
            if noise > 0:
                v = v + noise * torch.randn_like(v)
            if in_drop > 0:
                v = v * (torch.rand_like(v) > in_drop) / (1 - in_drop)
            if mixup > 0:
                lam = float(np.random.beta(mixup, mixup))
                lam = max(lam, 1 - lam)
                pi = torch.randperm(v.shape[0], device=v.device)
                v = lam * v + (1 - lam) * v[pi]
                yb = lam * yb + (1 - lam) * yb[pi]
            logit = model(v)
        logit = logit.float()
        if cfg.get("loss") == "focal":
            loss = focal_bce(logit, yb, cfg.get("gamma", 2.0), cfg.get("alpha"))
        elif cfg.get("loss") == "weighted":
            loss = nn.functional.binary_cross_entropy_with_logits(logit, yb, pos_weight=pw_t)
        else:
            loss = nn.functional.binary_cross_entropy_with_logits(logit, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run += loss.item()
        nrun += 1

        if step % ev == 0 or step == total:
            zl = predict_logits(model, vblocks, mode, mu, sd, dev) + shift
            pv = torch.sigmoid(zl).cpu().numpy().astype(np.float64)
            ce = C.bce(pv[m], yv_all[m])
            good = ce < best["ce"] - 1e-6
            if good:
                best = dict(ce=ce, step=step,
                            state={k: v.detach().clone() for k, v in model.state_dict().items()},
                            p=pv)
            log(f"    step{step:6d} ({step/spe:5.2f}ep) loss={run/nrun:.5f} "
                f"val_ce={ce:.5f}{'  *' if good else ''}")
            run, nrun = 0.0, 0
            if (step - best["step"]) // ev >= patience:
                stop = True
        if stop:
            break

    model.load_state_dict(best["state"])
    return dict(model=model, mu=mu, sd=sd, shift=shift, mode=mode, p_val=best["p"],
                train_s=time.time() - t0, best_ep=round(best["step"] / spe, 3),
                pos_weight=pw, neg_keep=q)


# ---------------------------------------------------------------- driver

def load_data(need_ctx=True):
    tr = C.load_split_arrays("train")
    va = C.load_split_arrays("val")
    dev = torch.device(DEV)
    ddev = torch.device(DATA_DEV)
    grab = lambda d: [torch.from_numpy(d[k]).to(ddev) for k in ("X", "C", "S", "N", "M", "K")]
    D = dict(blocks=grab(tr), yg=torch.from_numpy(tr["y"]).to(ddev).float())
    val = dict(blocks=grab(va),
               y_all=(va["frac"] >= 0.5).astype(np.float64),
               clean=C.clean_mask(va["frac"]),
               raw=va)
    return D, val, va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, help="path to a json list of configs")
    ap.add_argument("--save-all", action="store_true")
    a = ap.parse_args()
    os.makedirs(PRED, exist_ok=True)
    os.makedirs(CKPT, exist_ok=True)
    cfgs = json.load(open(a.sweep))
    D, val, va = load_data()
    print(f"train n={D['blocks'][0].shape[0]} pos={int(D['yg'].sum())} "
          f"| val n={val['blocks'][0].shape[0]} clean={int(val['clean'].sum())} "
          f"| data on {DATA_DEV}, compute on {DEV}", flush=True)
    for cfg in cfgs:
        name = cfg["name"]
        print(f"\n=== {name} :: {cfg}", flush=True)
        r = train_one(cfg, D, val, log=lambda s: print(s, flush=True))
        met = C.eval_val(r["p_val"], va)
        met["best_epoch"] = r["best_ep"]
        C.log_result(name, "mlp", cfg, met, r["train_s"],
                     extra=dict(shift=r["shift"], pos_weight=r["pos_weight"], neg_keep=r["neg_keep"]))
        np.save(f"{PRED}/{name}_val.npy", r["p_val"].astype(np.float32))
        torch.save(dict(state=r["model"].state_dict(), mu=r["mu"].cpu(), sd=r["sd"].cpu(),
                        shift=r["shift"], view=r["mode"], cfg=cfg), f"{CKPT}/{name}.pt")


if __name__ == "__main__":
    main()
