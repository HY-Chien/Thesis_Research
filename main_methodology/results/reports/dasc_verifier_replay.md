# DASC-lite Verifier Replay

This replay uses saved direct/decomposition outputs only; it does not rerun the LLM.

DASC-lite uses no gold labels for making a choice. Gold labels are used only for evaluation.

## Feature Coverage

| Dataset | DASC available | current direct | high-role direct | high-role direct wrong |
|---|---:|---:|---:|---:|
| 2Wiki | 992 | 48 | 6 | 5 |
| HotpotQA | 991 | 132 | 7 | 2 |
| MuSiQue | 997 | 73 | 10 | 10 |
| PopQA | 0 | 30 | 0 | 0 |

## Fixed Policies

| Policy | Dataset | EM | F1 | Direct % | Changed | Rescue/Lost |
|---|---|---:|---:|---:|---:|---:|
| current | 2Wiki | 49.90 | 56.42 | 4.8 | 0 | 0/0 |
| current | HotpotQA | 34.30 | 44.26 | 13.2 | 0 | 0/0 |
| current | MuSiQue | 24.80 | 31.08 | 7.3 | 0 | 0/0 |
| current | PopQA | 52.00 | 46.34 | 3.0 | 0 | 0/0 |
| dasc_guard_e0.3_r0.75 | 2Wiki | 49.90 | 56.42 | 4.2 | 6 | 1/1 |
| dasc_guard_e0.3_r0.75 | HotpotQA | 33.80 | 43.86 | 12.5 | 7 | 0/5 |
| dasc_guard_e0.3_r0.75 | MuSiQue | 25.20 | 31.48 | 6.3 | 10 | 4/0 |
| dasc_guard_e0.3_r0.75 | PopQA | 52.00 | 46.34 | 3.0 | 0 | 0/0 |
| dasc_guard_e0.5_r0.75 | 2Wiki | 49.90 | 56.42 | 4.0 | 8 | 1/1 |
| dasc_guard_e0.5_r0.75 | HotpotQA | 33.80 | 43.86 | 12.4 | 8 | 0/5 |
| dasc_guard_e0.5_r0.75 | MuSiQue | 25.20 | 31.48 | 6.2 | 11 | 4/0 |
| dasc_guard_e0.5_r0.75 | PopQA | 52.00 | 46.34 | 3.0 | 0 | 0/0 |
| dasc_auto_e0.3_low0_high0.75 | 2Wiki | 48.60 | 54.98 | 7.7 | 41 | 6/19 |
| dasc_auto_e0.3_low0_high0.75 | HotpotQA | 34.30 | 44.31 | 14.7 | 29 | 9/9 |
| dasc_auto_e0.3_low0_high0.75 | MuSiQue | 24.90 | 31.41 | 8.2 | 29 | 8/7 |
| dasc_auto_e0.3_low0_high0.75 | PopQA | 52.00 | 46.34 | 3.0 | 0 | 0/0 |
| dasc_auto_e0.5_low0_high0.75 | 2Wiki | 47.90 | 54.35 | 10.1 | 69 | 9/29 |
| dasc_auto_e0.5_low0_high0.75 | HotpotQA | 34.40 | 44.67 | 17.1 | 55 | 13/12 |
| dasc_auto_e0.5_low0_high0.75 | MuSiQue | 24.40 | 31.04 | 10.0 | 49 | 8/12 |
| dasc_auto_e0.5_low0_high0.75 | PopQA | 52.00 | 46.34 | 3.0 | 0 | 0/0 |
| dasc_auto_e0.8_low0.25_high0.75 | 2Wiki | 43.60 | 50.84 | 18.8 | 156 | 17/80 |
| dasc_auto_e0.8_low0.25_high0.75 | HotpotQA | 34.20 | 44.89 | 26.3 | 147 | 31/32 |
| dasc_auto_e0.8_low0.25_high0.75 | MuSiQue | 23.60 | 29.89 | 17.1 | 122 | 15/27 |
| dasc_auto_e0.8_low0.25_high0.75 | PopQA | 52.00 | 46.34 | 3.0 | 0 | 0/0 |

| Policy | Macro EM | Macro F1 | Macro Direct % |
|---|---:|---:|---:|
| current | 40.25 | 44.53 | 7.1 |
| dasc_guard_e0.3_r0.75 | 40.23 | 44.53 | 6.5 |
| dasc_guard_e0.5_r0.75 | 40.23 | 44.53 | 6.4 |
| dasc_auto_e0.3_low0_high0.75 | 39.95 | 44.26 | 8.4 |
| dasc_auto_e0.5_low0_high0.75 | 39.67 | 44.10 | 10.1 |
| dasc_auto_e0.8_low0.25_high0.75 | 38.35 | 42.99 | 16.3 |

## Exploratory Full-Dev Grid

The grid is tuned on the same dev traces, so it should be read as an exploratory upper bound rather than a final test result.

| Rank | ent_tau | low_role | high_role | EM | F1 | Direct % | Changed | Rescue/Lost |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.30 | -0.25 | 0.75 | 40.23 | 44.60 | 7.0 | 45 | 14/15 |
| 2 | 0.30 | -0.25 | 0.90 | 40.23 | 44.60 | 7.0 | 45 | 14/15 |
| 3 | 0.30 | -0.25 | 1.00 | 40.23 | 44.60 | 7.0 | 45 | 14/15 |
| 4 | 0.20 | -0.25 | 0.75 | 40.20 | 44.55 | 7.1 | 37 | 12/14 |
| 5 | 0.20 | -0.25 | 0.90 | 40.20 | 44.55 | 7.1 | 37 | 12/14 |
| 6 | 0.20 | -0.25 | 1.00 | 40.20 | 44.55 | 7.1 | 37 | 12/14 |
| 7 | 0.30 | -0.25 | 0.50 | 40.17 | 44.53 | 6.7 | 59 | 15/18 |
| 8 | 0.10 | -0.25 | 0.75 | 40.17 | 44.51 | 7.2 | 26 | 8/11 |
| 9 | 0.10 | -0.25 | 0.90 | 40.17 | 44.51 | 7.2 | 26 | 8/11 |
| 10 | 0.10 | -0.25 | 1.00 | 40.17 | 44.51 | 7.2 | 26 | 8/11 |
| 11 | 0.20 | -0.25 | 0.50 | 40.17 | 44.50 | 6.8 | 49 | 13/16 |
| 12 | 0.10 | -0.25 | 0.50 | 40.17 | 44.46 | 6.9 | 36 | 9/12 |
| 13 | 0.10 | 0.00 | 0.75 | 40.15 | 44.46 | 7.9 | 55 | 15/19 |
| 14 | 0.10 | 0.00 | 0.90 | 40.15 | 44.46 | 7.9 | 55 | 15/19 |
| 15 | 0.10 | 0.00 | 1.00 | 40.15 | 44.46 | 7.9 | 55 | 15/19 |
