# Stopping ceiling

How much could a per-slide stopping rule gain over a fixed visit budget, before any agent is
trained? For every slide the visits of a fixed order are played to the end and the Dice of
the combined mask is recorded after each step (unvisited tiles scored by a low-res head
from the level-3 tile embedding, visited tiles by a high-res head on the 16 level-1
sub-patches). With J = Dice − λ·steps, the oracle stop is the per-slide argmax of J using
the test labels; it is compared with the best single budget and with a stop-when-uncertainty-
drops rule, both chosen on the train rollouts. Visit orders: random, greedy, most-uncertain,
most-uncertain with neighbour follow-up after a tumour hit ("adaptive"), and ground-truth
order. The tumour-fraction sweep keeps each tumour slide's lesions and only the nearest
normal tiles, so slides at 10/20/40/60 % tumour tiles are real tiles, never pasted pixels.

Heads are trained fresh here (`heads.py`, ViT-S/16 DINO features). Dice is at 512 px
sub-patch resolution and is not comparable with the pixel-level numbers elsewhere.

## How to run (server)

```
bash experiments/stop/rjob.sh cuda:1     # heads -> rollouts -> report
```
Outputs under `/data2/venkatavks/stop/` (`heads.pt`, `rollouts/<split>/<slide>.npz`,
`report.md`, `results.json`). Needs the separability-study labels (`experiments/sep`) and
the SASHA high-res sub-patch features in `~/seg_outputs/hr_{train,test}_feats`.
