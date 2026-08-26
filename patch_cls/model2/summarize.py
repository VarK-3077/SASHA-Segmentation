# Quick console summary of the leaderboard, grouped over seeds.
#
#   python summarize.py --prefix V --top 20

import argparse
import collections
import re

import numpy as np

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--family", default=None)
    a = ap.parse_args()
    rows = {r["name"]: r for r in C.read_lb()}.values()
    rows = [r for r in rows if r["name"].startswith(a.prefix)]
    if a.family:
        rows = [r for r in rows if r["family"] == a.family]
    g = collections.defaultdict(list)
    for r in rows:
        g[re.sub(r"_s\d+$", "", r["name"])].append(r)
    out = []
    for k, v in g.items():
        ce = np.array([x["ce"] for x in v])
        ag = np.array([x["agg_ce"] for x in v])
        cfg = v[0]["config"] if isinstance(v[0]["config"], dict) else {}
        out.append((ce.mean(), k, len(v), ce.std(ddof=1) if len(ce) > 1 else 0.0, ce.min(),
                    ag.mean(), cfg.get("view", "-"), "+".join(cfg.get("rows", [])) or "-",
                    str(cfg.get("hidden", "-")), cfg.get("dropout", "-"), cfg.get("lr", "-")))
    out.sort()
    print(f"{'mean_ce':>9s} {'sd':>8s} {'min':>8s} {'agg':>8s} {'n':>2s}  {'name':<22s} "
          f"{'rows':<15s} {'view':<22s} {'hidden':<16s} drop lr")
    for m, k, n, sd, mn, ag, vw, rr, hid, dp, lr in out[:a.top]:
        print(f"{m:9.5f} {sd:8.5f} {mn:8.5f} {ag:8.5f} {n:2d}  {k:<22s} {rr:<15s} "
              f"{vw:<22s} {hid:<16s} {dp} {lr}")


if __name__ == "__main__":
    main()
