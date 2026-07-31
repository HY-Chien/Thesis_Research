# DASC-lite Verifier Replay

This replay uses saved direct/decomposition outputs only; it does not rerun the LLM.

DASC-lite uses no gold labels for making a choice. Gold labels are used only for evaluation.

## Feature Coverage

| Dataset | DASC available | current direct | high-role direct | high-role direct wrong |
|---|---:|---:|---:|---:|
| 2Wiki | 992 | 65 | 6 | 5 |
| HotpotQA | 991 | 274 | 9 | 4 |
| MuSiQue | 997 | 182 | 11 | 11 |
| PopQA | 0 | 37 | 0 | 0 |

## Fixed Policies

| Policy | Dataset | EM | F1 | Direct % | Changed | Rescue/Lost |
|---|---|---:|---:|---:|---:|---:|
| current | 2Wiki | 50.00 | 56.58 | 6.5 | 0 | 0/0 |
| current | HotpotQA | 35.50 | 46.11 | 27.4 | 0 | 0/0 |
| current | MuSiQue | 24.70 | 31.39 | 18.2 | 0 | 0/0 |
| current | PopQA | 51.90 | 46.28 | 3.7 | 0 | 0/0 |
| dasc_guard_e0.3_r0.75 | 2Wiki | 50.00 | 56.58 | 5.9 | 6 | 1/1 |
| dasc_guard_e0.3_r0.75 | HotpotQA | 35.00 | 45.70 | 26.5 | 9 | 0/5 |
| dasc_guard_e0.3_r0.75 | MuSiQue | 25.10 | 31.79 | 17.1 | 11 | 4/0 |
| dasc_guard_e0.3_r0.75 | PopQA | 51.90 | 46.28 | 3.7 | 0 | 0/0 |
| dasc_guard_e0.5_r0.75 | 2Wiki | 50.00 | 56.58 | 5.7 | 8 | 1/1 |
| dasc_guard_e0.5_r0.75 | HotpotQA | 35.00 | 45.70 | 26.4 | 10 | 0/5 |
| dasc_guard_e0.5_r0.75 | MuSiQue | 25.10 | 31.79 | 16.9 | 13 | 4/0 |
| dasc_guard_e0.5_r0.75 | PopQA | 51.90 | 46.28 | 3.7 | 0 | 0/0 |
| dasc_auto_e0.3_low0_high0.75 | 2Wiki | 48.70 | 55.15 | 9.1 | 38 | 6/19 |
| dasc_auto_e0.3_low0_high0.75 | HotpotQA | 35.00 | 45.54 | 27.7 | 21 | 3/8 |
| dasc_auto_e0.3_low0_high0.75 | MuSiQue | 24.80 | 31.72 | 19.0 | 30 | 8/7 |
| dasc_auto_e0.3_low0_high0.75 | PopQA | 51.90 | 46.28 | 3.7 | 0 | 0/0 |
| dasc_auto_e0.5_low0_high0.75 | 2Wiki | 48.00 | 54.52 | 11.5 | 66 | 9/29 |
| dasc_auto_e0.5_low0_high0.75 | HotpotQA | 35.00 | 45.65 | 29.1 | 37 | 5/10 |
| dasc_auto_e0.5_low0_high0.75 | MuSiQue | 24.30 | 31.35 | 20.6 | 50 | 8/12 |
| dasc_auto_e0.5_low0_high0.75 | PopQA | 51.90 | 46.28 | 3.7 | 0 | 0/0 |
| dasc_auto_e0.8_low0.25_high0.75 | 2Wiki | 43.60 | 50.91 | 19.8 | 151 | 16/80 |
| dasc_auto_e0.8_low0.25_high0.75 | HotpotQA | 34.50 | 45.36 | 35.5 | 103 | 16/26 |
| dasc_auto_e0.8_low0.25_high0.75 | MuSiQue | 23.40 | 30.18 | 26.6 | 112 | 14/27 |
| dasc_auto_e0.8_low0.25_high0.75 | PopQA | 51.90 | 46.28 | 3.7 | 0 | 0/0 |

| Policy | Macro EM | Macro F1 | Macro Direct % |
|---|---:|---:|---:|
| current | 40.52 | 45.09 | 13.9 |
| dasc_guard_e0.3_r0.75 | 40.50 | 45.09 | 13.3 |
| dasc_guard_e0.5_r0.75 | 40.50 | 45.09 | 13.2 |
| dasc_auto_e0.3_low0_high0.75 | 40.10 | 44.67 | 14.9 |
| dasc_auto_e0.5_low0_high0.75 | 39.80 | 44.45 | 16.2 |
| dasc_auto_e0.8_low0.25_high0.75 | 38.35 | 43.18 | 21.4 |

## Exploratory Full-Dev Grid

The grid is tuned on the same dev traces, so it should be read as an exploratory upper bound rather than a final test result.

| Rank | ent_tau | low_role | high_role | EM | F1 | Direct % | Changed | Rescue/Lost |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.30 | -0.25 | 0.75 | 40.45 | 45.09 | 13.8 | 45 | 12/15 |
| 2 | 0.30 | -0.25 | 0.90 | 40.45 | 45.09 | 13.8 | 45 | 12/15 |
| 3 | 0.30 | -0.25 | 1.00 | 40.45 | 45.09 | 13.8 | 45 | 12/15 |
| 4 | 0.20 | -0.25 | 0.75 | 40.42 | 45.04 | 13.8 | 36 | 10/14 |
| 5 | 0.20 | -0.25 | 0.90 | 40.42 | 45.04 | 13.8 | 36 | 10/14 |
| 6 | 0.20 | -0.25 | 1.00 | 40.42 | 45.04 | 13.8 | 36 | 10/14 |
| 7 | 0.30 | -0.25 | 0.50 | 40.40 | 45.02 | 13.4 | 59 | 13/18 |
| 8 | 0.10 | -0.25 | 0.75 | 40.40 | 44.99 | 13.9 | 25 | 6/11 |
| 9 | 0.10 | -0.25 | 0.90 | 40.40 | 44.99 | 13.9 | 25 | 6/11 |
| 10 | 0.10 | -0.25 | 1.00 | 40.40 | 44.99 | 13.9 | 25 | 6/11 |
| 11 | 0.20 | -0.25 | 0.50 | 40.40 | 44.98 | 13.6 | 48 | 11/16 |
| 12 | 0.10 | -0.25 | 0.50 | 40.40 | 44.94 | 13.7 | 35 | 7/12 |
| 13 | 0.50 | -0.25 | 0.75 | 40.33 | 45.04 | 14.0 | 65 | 13/21 |
| 14 | 0.50 | -0.25 | 0.90 | 40.33 | 45.04 | 14.0 | 65 | 13/21 |
| 15 | 0.50 | -0.25 | 1.00 | 40.33 | 45.04 | 14.0 | 65 | 13/21 |
