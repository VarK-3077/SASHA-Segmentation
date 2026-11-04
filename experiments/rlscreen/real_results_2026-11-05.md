# RL screening with the real level-2 segmenter (129 test slides, A100, solo, cold cache)
| run | wall s/slide | median | tissue / global / rounds | screened | zoomed | rounds | Dice cleaned | pooled | lesion Dice <0.5/0.5-2/>=2 | FROC raw |
|---|---|---|---|---|---|---|---|---|---|---|
| full level-2 token head | 8.44 | 8.16 | - | - | 100 % | - | 0.782 | 0.900 | 0.580 / 0.773 / 0.903 | 0.862 |
| supervised L3 cascade, pipelined, 17 % | 3.25 | 2.92 | - | - | 17.2 % | - | 0.782 | 0.899 | 0.513 / 0.749 / 0.902 | 0.838 |
| coarse_l6_vitl_z07 | 3.16 | 2.71 | 0.56 / 0.12 / 2.48 | 48.4 % | 14.1 % | 2.7 | 0.795 | 0.900 | 0.550 / 0.756 / 0.903 | 0.858 |
| coarse_l6_vitl_z08 | 2.85 | 2.33 | 0.56 / 0.12 / 2.17 | 48.4 % | 10.7 % | 2.7 | 0.773 | 0.900 | 0.520 / 0.756 / 0.903 | 0.843 |
| rl_vitl_full_l1 | 3.04 | 2.54 | 0.57 / 1.52 / 0.95 | 0.0 % | 10.1 % | 1.0 | 0.782 | 0.898 | 0.495 / 0.742 / 0.897 | 0.833 |
| rl_vits_light_l1 | 4.46 | 3.74 | 0.57 / 1.25 / 2.63 | 0.0 % | 29.0 % | 1.0 | 0.785 | 0.900 | 0.501 / 0.753 / 0.903 | 0.837 |
| rl_vits_light_l3 | 4.19 | 3.59 | 0.58 / 1.28 / 2.34 | 0.0 % | 25.7 % | 1.0 | 0.780 | 0.900 | 0.487 / 0.748 / 0.900 | 0.831 |
| vitl_full_teacher_t05 | 3.04 | 2.44 | 0.56 / 1.52 / 0.96 | 0.0 % | 10.2 % | 1.3 | 0.774 | 0.900 | 0.498 / 0.734 / 0.903 | 0.825 |
| vits_full_teacher_t05 | 4.70 | 3.94 | 0.61 / 1.27 / 2.83 | 0.0 % | 31.2 % | 1.4 | 0.789 | 0.900 | 0.520 / 0.764 / 0.903 | 0.846 |
| vits_full_teacher_t06 | 3.73 | 3.02 | 0.60 / 1.28 / 1.86 | 0.0 % | 20.4 % | 1.4 | 0.779 | 0.900 | 0.491 / 0.755 / 0.903 | 0.829 |

Caveat: the coarse zoom thresholds 0.7 / 0.8 were picked from an offline sweep on the test split (ideal segmenter); a val-split pick is still owed. RL rows: PPO, per-unit credit, S^J shaping, lambda as listed; teacher rows: fixed-threshold rule with border following.
