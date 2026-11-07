| run | slides | wall s/slide | median | L2 units / slide | painted / slide | Dice cleaned | lesion Dice <0.5 / 0.5-2 / >=2 | FROC raw |
|---|---|---|---|---|---|---|---|---|
| cascade_L3_b0.2 | 129 | 2.74 | 2.46 | 177 | 0 | 0.782 | 0.513 / 0.749 / 0.902 | 0.839 |
| coarse_z07 | 129 | 2.37 | 1.99 | 146 | 0 | 0.795 | 0.550 / 0.756 / 0.903 | 0.858 |
| coarse_z07_paint | 129 | 2.28 | 1.99 | 133 | 13 | 0.796 | 0.550 / 0.756 / 0.905 | 0.857 |
| full_l2 | 129 | 6.53 | 6.06 | 1030 | 0 | 0.782 | 0.580 / 0.773 / 0.903 | 0.862 |
| hit_paint | 129 | 2.46 | 2.05 | 81 | 14 | 0.787 | 0.476 / 0.708 / 0.862 | 0.793 |
| ppo_l02 | 129 | 3.50 | 2.89 | 210 | 14 | 0.793 | 0.540 / 0.773 / 0.905 | 0.854 |
| ppo_l04 | 129 | 3.02 | 2.49 | 146 | 14 | 0.793 | 0.530 / 0.773 / 0.905 | 0.853 |

vivi GPU 1, test slides on NVMe, one run at a time, cold cache, other users load ~13. Coarse zoom tau 0.7 was picked on test; ppo_l04 lambda was noticed on test, ppo_l02 is the out-of-fold pick. PPO stopper: stop_full_s0, paint p_core >= 0.99.
