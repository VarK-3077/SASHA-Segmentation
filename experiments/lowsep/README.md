# Low-level screen separability (levels 3-6)

Can a cheap pass at a low pyramid level tell which units the token-head segmenter needs to see?
The unit is the segmenter's own input (2048 px level-0 crop on its (0, 0) grid and Otsu tissue
mask), bucketed normal / core / boundary / small as in `experiments/sep`. The screen reads
256 px crops at level L (each covering 2^(L-3) x 2^(L-3) units), embeds them with frozen Kaiko
ViT-L/14 into 16 x 16 patch tokens, reduces them to 256 dims with PCA, stitches a whole-slide
token map, and a trained head predicts each unit's bucket: with context (PFE GlobalConv over
tokens and over units, mean + max token pooling per unit) or without (pooling + MLP only).
Two seeds per head, a regularisation sweep on head width / dropout, test = every tissue unit of
the 129 test slides.

Scored three ways: bucket recall and recall at 99 % specificity; routing with an ideal
segmenter (zoomed units get ground truth) under several rules; and a budget frontier (zoom
the top-b fraction of units by p(tumour), one global threshold, Dice on tumour slides) that
compares levels at equal cost independent of calibration. `time_screen.py` measures the screen
alone on the A100 with a cold page cache per slide, levels 2-6 (level 2 = the full token pass).

## How to run (GPU box)

```
bash experiments/lowsep/chain.sh cuda:0             # labels, tokens L6..L3, heads, test eval
bash experiments/lowsep/after_chain.sh <pids...>     # timing, after every GPU job has ended
bash experiments/lowsep/sweep.sh <timing pid>        # narrower / more regularised heads
bash experiments/lowsep/reeval.sh <sweep pid>        # re-score all runs, report
```
Outputs under `/data2/venkatavks/lowsep/` (`labels/`, `tokens_L*/`, `runs/<tag>/`, `timing.json`,
`report.md`). Token maps take 52 / 17 / 6 / 2 GB at L3 / L4 / L5 / L6.
