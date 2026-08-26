# Scores the 129 test slides with the exported bundle and writes {slide: (N,16) fp16}.
#
# Also prints the slide-level mean probability split by the CAMELYON16 test ground-truth
# labels. That split is a sanity check only -- it is read after the model is frozen and
# nothing is selected on it.

import argparse
import csv
import json
import os
import pickle
import time

import numpy as np
import torch

import predict as P

HOME = os.path.expanduser("~")
FEAT_TEST = f"{HOME}/seg_outputs/hr_test_feats"
SPLIT = f"{HOME}/SASHA-Segmentation/dataset_csv/camelyon16/splits/split_4.json"
OUT = f"{HOME}/seg_outputs/patch_cls/model1"


def _np(a):
    return a.numpy() if isinstance(a, torch.Tensor) else np.asarray(a)


def test_labels():
    """slide -> 0/1 from the dataset csv, if it is available."""
    for p in (f"{HOME}/SASHA-Segmentation/dataset_csv/camelyon16/camelyon16.csv",
              f"{HOME}/SASHA-Segmentation/dataset_csv/camelyon16.csv"):
        if os.path.exists(p):
            out = {}
            for r in csv.DictReader(open(p)):
                k = r.get("slide_id") or r.get("case_id") or r.get("image_id")
                v = r.get("label") or r.get("Label")
                if k and v is not None:
                    out[str(k).replace(".tif", "")] = 0 if str(v).lower() in ("0", "normal") else 1
            return out
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=f"{OUT}/best")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default=f"{OUT}/test_probs.pkl")
    a = ap.parse_args()
    names = sorted(json.load(open(SPLIT))["test_names"])
    m = P.Model1.from_dir(a.bundle, a.device)
    probs, means, t0 = {}, {}, time.time()
    for i, n in enumerate(names):
        d = torch.load(f"{FEAT_TEST}/{n}.pt", weights_only=False)
        p = m.predict_slide(_np(d["feat"]), _np(d["coords"]))
        probs[n] = p.astype(np.float16)
        means[n] = float(p.mean())
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(names)}  {time.time()-t0:.0f}s", flush=True)
    with open(a.out, "wb") as f:
        pickle.dump(probs, f)
    print(f"wrote {a.out}: {len(probs)} slides, {time.time()-t0:.0f}s")

    lab = test_labels()
    got = {n: lab[n] for n in names if n in lab}
    if got:
        pos = [means[n] for n in got if got[n] == 1]
        neg = [means[n] for n in got if got[n] == 0]
        print(f"\nslide-level sanity (NOT tuned on): {len(pos)} tumour / {len(neg)} normal test slides")
        print(f"  mean sub-patch P on tumour slides : {np.mean(pos):.5f} "
              f"(median {np.median(pos):.5f})")
        print(f"  mean sub-patch P on normal slides : {np.mean(neg):.5f} "
              f"(median {np.median(neg):.5f})")
        for tag, q in (("max", np.max), ("p99", lambda v: np.percentile(v, 99))):
            fp = [q(probs[n].astype(np.float32)) for n in got if got[n] == 1]
            fn = [q(probs[n].astype(np.float32)) for n in got if got[n] == 0]
            print(f"  per-slide {tag} P: tumour {np.mean(fp):.4f} vs normal {np.mean(fn):.4f}")
        from sklearn.metrics import roc_auc_score
        y = [got[n] for n in got]
        for tag, agg in (("mean", lambda n: means[n]),
                         ("p999", lambda n: float(np.percentile(probs[n].astype(np.float32), 99.9)))):
            print(f"  slide-level ROC-AUC using {tag} of sub-patch P: "
                  f"{roc_auc_score(y, [agg(n) for n in got]):.4f}")
    else:
        print("no test labels csv found; skipping slide-level sanity")
    json.dump(means, open(f"{OUT}/test_slide_mean_prob.json", "w"), indent=1)


if __name__ == "__main__":
    main()
