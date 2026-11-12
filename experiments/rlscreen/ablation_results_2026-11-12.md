| row | s/slide | L2 units | painted | Dice | lesion Dice <0.5 / 0.5-2 / >=2 mm | FROC |
|---|---|---|---|---|---|---|
| r1_fixed | 2.72 | 216 | 0.0 | 0.777 | 0.542 / 0.748 / 0.893 | 0.859 |
| r2_paint | 2.64 | 203 | 13.0 | 0.778 | 0.542 / 0.748 / 0.894 | 0.853 |
| r3_escalate | 2.31 | 171 | 13.3 | 0.759 | 0.546 / 0.751 / 0.904 | 0.854 |
| r4_trace | 2.69 | 207 | 13.0 | 0.796 | 0.561 / 0.773 / 0.905 | 0.865 |
| r5_interior | 2.64 | 203 | 13.0 | 0.778 | 0.542 / 0.748 / 0.894 | 0.853 |
| r6_all | 2.34 | 173 | 12.6 | 0.767 | 0.553 / 0.761 / 0.905 | 0.858 |

| row | vs | dDice 95% | dFROC 95% | ds 95% |
|---|---|---|---|---|
| r1_fixed | r3_escalate | +0.019 (-0.003 .. +0.051) | +0.006 (-0.010 .. +0.022) | +0.41 (+0.33 .. +0.51) |
| r2_paint | r3_escalate | +0.019 (-0.003 .. +0.052) | -0.001 (-0.016 .. +0.019) | +0.33 (+0.26 .. +0.41) |
| r4_trace | r3_escalate | +0.038 (+0.001 .. +0.085) | +0.011 (-0.007 .. +0.039) | +0.37 (+0.31 .. +0.45) |
| r5_interior | r3_escalate | +0.019 (-0.003 .. +0.052) | -0.001 (-0.016 .. +0.019) | +0.33 (+0.26 .. +0.41) |
| r6_all | r3_escalate | +0.008 (+0.000 .. +0.025) | +0.004 (-0.004 .. +0.011) | +0.03 (+0.02 .. +0.05) |
| r1_fixed | r2_paint | -0.001 (-0.001 .. -0.000) | +0.007 (+0.000 .. +0.013) | +0.08 (+0.02 .. +0.17) |
| r3_escalate | r2_paint | -0.019 (-0.052 .. +0.003) | +0.001 (-0.019 .. +0.016) | -0.33 (-0.41 .. -0.26) |
| r4_trace | r2_paint | +0.018 (+0.004 .. +0.037) | +0.012 (+0.001 .. +0.027) | +0.04 (+0.03 .. +0.06) |
| r5_interior | r2_paint | +0.000 (+0.000 .. +0.000) | +0.000 (+0.000 .. +0.000) | +0.00 (-0.00 .. +0.01) |
| r6_all | r2_paint | -0.011 (-0.045 .. +0.014) | +0.005 (-0.016 .. +0.018) | -0.30 (-0.38 .. -0.22) |

[exited with code 0]

Real fine-to-coarse feedback ablation, 129 CAMELYON16 test slides, vivi RTX 5090 (GPU 1), test slides on NVMe, cold cache per slide, one run at a time, 09:25-10:04 IST on a quiet machine (load ~1). L6 -> L3, screen 0.6, zoom 0.5. Dice = mean per-slide Dice of the cleaned map (t0.4 seed 0.9); FROC on the raw map at 0.4; bootstraps: 1000 resamples of slides, paired. Settings chosen on out-of-fold train with real segmenter outcomes (experiments/rlscreen/ablate.py, cheapest within 0.01 of the best lesion-aware train score): r3 escalation 0.8 -> 0.7, >= 4 units with segmented share >= 0.05 in the last stage; r5 interior critic eps 0.005; r6 escalation 0.8 -> 0.75 -> 0.7 with the same evidence rule and eps 0.005. Looked at test before: the 0.99 core threshold and the escalation family itself. The base 0.6 / 0.5 came from the out-of-fold ideal-segmenter sweep.
