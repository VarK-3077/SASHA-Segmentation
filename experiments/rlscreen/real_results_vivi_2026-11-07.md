| run | slides | wall s/slide | median | L2 units / slide | painted / slide | Dice cleaned | lesion Dice <0.5 / 0.5-2 / >=2 | FROC raw |
|---|---|---|---|---|---|---|---|---|
| cascade_L3_b0.2 | 129 | 2.74 | 2.46 | 177 | 0 | 0.782 | 0.513 / 0.749 / 0.902 | 0.839 |
| coarse_z07 | 129 | 2.37 | 1.99 | 146 | 0 | 0.795 | 0.550 / 0.756 / 0.903 | 0.858 |
| coarse_z07_paint | 129 | 2.28 | 1.99 | 133 | 13 | 0.796 | 0.550 / 0.756 / 0.905 | 0.857 |
| esc_08_06_z07 | 129 | 2.28 | 1.86 | 138 | 0 | 0.795 | 0.550 / 0.756 / 0.903 | 0.858 |
| esc_08_06_z07_paint | 129 | 2.21 | 1.84 | 126 | 13 | 0.796 | 0.550 / 0.756 / 0.905 | 0.858 |
| full_l2 | 129 | 6.53 | 6.06 | 1030 | 0 | 0.782 | 0.580 / 0.773 / 0.903 | 0.862 |
| hit_paint | 129 | 2.46 | 2.05 | 81 | 14 | 0.787 | 0.476 / 0.708 / 0.862 | 0.793 |
| les_rl_s0_l08 | 129 | 1.57 | 1.31 | 100 | 12 | 0.755 | 0.483 / 0.713 / 0.871 | 0.819 |
| les_rl_s2_l08 | 129 | 2.13 | 1.89 | 152 | 12 | 0.759 | 0.506 / 0.742 / 0.889 | 0.832 |
| ops_t08_z05_paint | 129 | 2.22 | 1.90 | 161 | 13 | 0.766 | 0.513 / 0.760 / 0.905 | 0.839 |
| ops_tol01 | 129 | 2.82 | 2.41 | 220 | 0 | 0.796 | 0.561 / 0.773 / 0.903 | 0.871 |
| ops_tol01_paint | 129 | 2.72 | 2.40 | 207 | 13 | 0.796 | 0.561 / 0.773 / 0.905 | 0.865 |
| ops_tol05 | 129 | 1.91 | 1.55 | 116 | 0 | 0.764 | 0.505 / 0.743 / 0.903 | 0.832 |
| ops_tol05_paint | 129 | 1.82 | 1.53 | 104 | 13 | 0.765 | 0.505 / 0.743 / 0.905 | 0.828 |
| ppo_l02 | 129 | 3.50 | 2.89 | 210 | 14 | 0.793 | 0.540 / 0.773 / 0.905 | 0.854 |
| ppo_l04 | 129 | 3.02 | 2.49 | 146 | 14 | 0.793 | 0.530 / 0.773 / 0.905 | 0.853 |
| rl_coarse_long_l02 | 129 | 2.35 | 2.07 | 174 | 12 | 0.783 | 0.521 / 0.773 / 0.905 | 0.845 |
| rl_coarse_s0_l08 | 129 | 2.40 | 2.17 | 172 | 12 | 0.781 | 0.537 / 0.761 / 0.905 | 0.852 |

vivi GPU 1, test slides on NVMe, one run at a time, cold cache, other users load ~13-30. ops_* chosen on out-of-fold train by Dice tolerance: tol01 = screen 0.6, zoom 0.5; tol05 = screen 0.8, zoom 0.7; t08_z05 = screen 0.8, zoom 0.5. esc_* = escalation rule, screen 0.8 then 0.6 only if tumour found, zoom 0.7. rl_coarse_*: PPO stopper on the ideal reward; les_rl_*: PPO stopper on segmenter outcomes with the lesion-aware reward. coarse_z07* zoom tau and ppo_l04 lambda were picked on test.
