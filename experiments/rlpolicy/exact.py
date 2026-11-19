# Exact scores for simulator routes: builds each route's map (dense level-2 map kept on the segmented units,
# painted units at 1) and scores it with the cascade's own scorer, in the layout rlcascade.report reads
# ({slide: {rows, scores}}), so its tables and paired bootstrap apply unchanged. The rule is always included.
# Usage (from code): res = score_routes('train', {name: {key: route}}); routes are sim.finish outputs.

from functools import partial
from multiprocessing import get_context

from experiments.rlcascade import cascade as C
from experiments.rlcascade import fast_score as FS
from experiments.rlpolicy.simtab import DENSE


def row(r, key):
    return dict(n_l3=r['n_l3'], n_l2=r['n_l2'], n_paint=r['n_paint'], sec=r['sec'], stopped=bool(r.get('stopped', False)), map=key)


def job(item, split):
    name, routes = item
    maps = {k: C.route_map(DENSE[split], name, dict(zoom=r['zoom'], paint=r['paint'])) for k, r in routes.items()}
    return name, dict(rows={k: row(r, k) for k, r in routes.items()}, scores=FS.score(name, maps))


def score_routes(split, routes, workers=8):
    with get_context('fork').Pool(workers) as pool:
        return dict(pool.imap_unordered(partial(job, split=split), routes.items()))
