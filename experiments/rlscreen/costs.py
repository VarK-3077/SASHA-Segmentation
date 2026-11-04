# Cost model in A100 seconds, from the lowsep timing runs (solo, cold page cache, 129 test slides).
# It prices every action so the reward can charge for compute in a hardware unit, and it turns an
# offline episode into an estimated wall time. Values marked "est." have no direct measurement yet
# and must be replaced by run_real.py timings before any claim. No pipelining credit is given, so
# estimates are conservative for runs whose stages overlap.

TISSUE = 0.56                       # level-5 Otsu mask, shared by every pipeline
FULL_PER_UNIT = 8.44 / 1030         # full level-2 token head, per tissue unit (measured mean)
ZOOM_PER_UNIT = 0.010               # level-2 token head on a selected unit, scattered reads (est. from cascade runs)
ROUND = 0.05                        # per-round dispatch / sync latency (est.)

GLOBAL_PASS = {                     # whole-slide pass at a level, excluding the tissue mask
    ('kaiko_vitl14', 3): 2.28, ('kaiko_vitl14', 5): 0.72, ('kaiko_vitl14', 6): 0.37,
    ('kaiko_vits16', 3): 0.95, ('kaiko_vits16', 5): 0.50, ('kaiko_vits16', 6): 0.25,  # est.: read-bound
}
L3_PER_UNIT = {'kaiko_vitl14': 2.28 / 1030, 'kaiko_vits16': 0.95 / 1030}   # screening one unit at level 3


def action_seconds(backbone, n_screen, n_zoom):
    return n_screen * L3_PER_UNIT[backbone] + n_zoom * ZOOM_PER_UNIT + ROUND


def cost_fraction(backbone, n_screen, n_zoom, n_units):
    """An action round's price as a fraction of segmenting the whole slide with the level-2 head."""
    return action_seconds(backbone, n_screen, n_zoom) / (FULL_PER_UNIT * max(n_units, 1))


def episode_seconds(backbone, g_level, n_screen, n_zoom, rounds):
    return TISSUE + GLOBAL_PASS[(backbone, g_level)] + n_screen * L3_PER_UNIT[backbone] + n_zoom * ZOOM_PER_UNIT + rounds * ROUND
