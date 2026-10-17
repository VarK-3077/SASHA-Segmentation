# RL binning

Can a policy trained by RL sort a slide's tiles into zoom / core / normal? One slide is an
episode. At every step the agent picks a tile and zooms it (its 16 sub-patches then carry the
high-res scorer's prediction), marks it core (all 16 positive), or stops; whatever is left
untouched is normal. Reward is the change in Dice of that composed mask, so the return is the
final Dice. The only constraint is a zoom cap of half the tiles, no per-read cost, so the
result is a ceiling for what the policy can learn to bin rather than a speed/accuracy
trade-off. Level-3 tiles (2048 px) with the SASHA ViT-S/16 DINO embeddings and the two heads
from `experiments/stop` as scorers.

Policy: per-tile MLP over [embedding, low-res score summary, status, high-res summary once
zoomed, episode progress], a pooled slide vector fed back into each tile's zoom/core logits
and into the stop logit and value. PPO, γ = 1, clip 0.2, entropy 0.01, 4 epochs per batch of
8 episodes. Train = split-4 train slides minus the 20 non-exhaustively annotated ones; best
checkpoint by val Dice; test scored once against: low-res mask only, zoom everything, a
supervised router (core if mean low-res score ≥ 0.9, zoom the most uncertain tiles at a given
fraction), and the label-using oracle routing (zoom boundary + small, core = core).

## How to run (server)

```
bash experiments/rlbin/rjob.sh cuda:0 l3     # cache -> train -> evaluate
```
Outputs under `/data2/venkatavks/rlbin/` (`cache/`, `ckpt/`, `train_l3.jsonl`, `report_l3.md`,
`results_l3.json`).
