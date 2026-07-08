# Path-Consistency Final Aggregation Replay

This replay uses saved direct/decompose outputs only; it does not rerun the LLM.

## Bucket Summary

| Dataset | Bucket | N | Current EM | Direct EM | Decomp EM |
|---|---|---:|---:|---:|---:|
| HotpotQA | exact | 339 | 56.64 | 56.64 | 56.64 |
| HotpotQA | partial>=0.5 | 70 | 27.14 | 18.57 | 22.86 |
| HotpotQA | diff<0.5 | 591 | 24.37 | 16.58 | 16.41 |
| 2Wiki | exact | 232 | 60.78 | 60.78 | 60.78 |
| 2Wiki | partial>=0.5 | 120 | 52.50 | 8.33 | 53.33 |
| 2Wiki | diff<0.5 | 648 | 45.68 | 4.48 | 45.83 |
| MuSiQue | exact | 166 | 37.95 | 37.95 | 37.95 |
| MuSiQue | partial>=0.5 | 90 | 30.00 | 11.11 | 30.00 |
| MuSiQue | diff<0.5 | 744 | 21.10 | 3.90 | 22.18 |
| PopQA | exact | 468 | 65.81 | 65.81 | 65.81 |
| PopQA | partial>=0.5 | 193 | 60.62 | 22.28 | 63.73 |
| PopQA | diff<0.5 | 339 | 27.73 | 7.96 | 28.91 |

## Policy Replay

| Policy | Dataset | EM | F1 | Direct % | Changed | Rescue/Lost |
|---|---|---:|---:|---:|---:|---:|
| current | HotpotQA | 35.50 | 46.11 | 27.4 | 0 | 0/0 |
| current | 2Wiki | 50.00 | 56.58 | 6.5 | 0 | 0/0 |
| current | MuSiQue | 24.70 | 31.39 | 18.2 | 0 | 0/0 |
| current | PopQA | 51.90 | 46.28 | 3.7 | 0 | 0/0 |
| pc_low<0.5_to_decomp | HotpotQA | 30.80 | 39.76 | 3.0 | 244 | 20/67 |
| pc_low<0.5_to_decomp | 2Wiki | 50.10 | 56.54 | 1.1 | 54 | 10/9 |
| pc_low<0.5_to_decomp | MuSiQue | 25.50 | 31.48 | 1.7 | 165 | 19/11 |
| pc_low<0.5_to_decomp | PopQA | 52.30 | 46.31 | 0.9 | 28 | 6/2 |
| pc_low<0.75_to_decomp | HotpotQA | 30.60 | 39.50 | 0.9 | 265 | 23/72 |
| pc_low<0.75_to_decomp | 2Wiki | 50.30 | 56.68 | 0.2 | 63 | 13/10 |
| pc_low<0.75_to_decomp | MuSiQue | 25.50 | 31.50 | 0.3 | 179 | 20/12 |
| pc_low<0.75_to_decomp | PopQA | 52.90 | 46.50 | 0.1 | 36 | 12/2 |
| pc_partial>=0.25_to_decomp | HotpotQA | 35.10 | 45.70 | 23.8 | 36 | 6/10 |
| pc_partial>=0.25_to_decomp | 2Wiki | 50.30 | 56.76 | 4.7 | 18 | 5/2 |
| pc_partial>=0.25_to_decomp | MuSiQue | 25.10 | 31.39 | 15.5 | 27 | 5/1 |
| pc_partial>=0.25_to_decomp | PopQA | 52.70 | 46.46 | 2.4 | 13 | 8/0 |
| pc_partial>=0.5_to_decomp | HotpotQA | 35.20 | 45.84 | 24.4 | 30 | 5/8 |
| pc_partial>=0.5_to_decomp | 2Wiki | 50.10 | 56.69 | 5.4 | 11 | 3/2 |
| pc_partial>=0.5_to_decomp | MuSiQue | 24.70 | 31.41 | 16.5 | 17 | 1/1 |
| pc_partial>=0.5_to_decomp | PopQA | 52.50 | 46.47 | 2.8 | 9 | 6/0 |
| pc_any_disagree_to_decomp | HotpotQA | 30.50 | 39.49 | 0.0 | 274 | 25/75 |
| pc_any_disagree_to_decomp | 2Wiki | 50.20 | 56.65 | 0.0 | 65 | 13/11 |
| pc_any_disagree_to_decomp | MuSiQue | 25.50 | 31.49 | 0.0 | 182 | 20/12 |
| pc_any_disagree_to_decomp | PopQA | 52.90 | 46.50 | 0.0 | 37 | 12/2 |

| Policy | Macro EM | Macro F1 | Macro Direct % |
|---|---:|---:|---:|
| current | 40.52 | 45.09 | 13.9 |
| pc_low<0.5_to_decomp | 39.67 | 43.52 | 1.7 |
| pc_low<0.75_to_decomp | 39.83 | 43.55 | 0.4 |
| pc_partial>=0.25_to_decomp | 40.80 | 45.08 | 11.6 |
| pc_partial>=0.5_to_decomp | 40.62 | 45.10 | 12.3 |
| pc_any_disagree_to_decomp | 39.77 | 43.53 | 0.0 |
