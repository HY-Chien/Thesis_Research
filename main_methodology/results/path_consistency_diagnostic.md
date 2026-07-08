## HotpotQA (N=1000)

| Metric | Value |
|---|---:|
| Current/replay EM | 35.50 |
| Direct EM | 30.30 |
| Decompose EM | 30.50 |
| Oracle EM | 41.60 |
| Similarity -> current-correct AUROC | 0.655 |
| Similarity -> oracle-correct AUROC | 0.609 |

| Consistency bucket | N | Share | Current EM | Direct EM | Decomp EM | Oracle EM | Both correct | Both wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exact | 339 | 33.9 | 56.64 | 56.64 | 56.64 | 56.64 | 56.64 | 43.36 |
| partial>=0.5 | 70 | 7.0 | 27.14 | 18.57 | 22.86 | 41.43 | 0.00 | 58.57 |
| diff<0.5 | 591 | 59.1 | 24.37 | 16.58 | 16.41 | 32.99 | 0.00 | 67.01 |

| Escalation set | N | Current EM | Oracle EM | Oracle-current gap |
|---|---:|---:|---:|---:|
| sim < 0.25 | 567 | 24.69 | 33.16 | +8.47 |
| sim < 0.5 | 591 | 24.37 | 32.99 | +8.63 |
| sim < 0.75 | 647 | 24.73 | 33.85 | +9.12 |
| sim < 0.999999 | 661 | 24.66 | 33.89 | +9.23 |

## 2Wiki (N=1000)

| Metric | Value |
|---|---:|
| Current/replay EM | 50.00 |
| Direct EM | 18.00 |
| Decompose EM | 50.20 |
| Oracle EM | 54.10 |
| Similarity -> current-correct AUROC | 0.566 |
| Similarity -> oracle-correct AUROC | 0.555 |

| Consistency bucket | N | Share | Current EM | Direct EM | Decomp EM | Oracle EM | Both correct | Both wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exact | 232 | 23.2 | 60.78 | 60.78 | 60.78 | 60.78 | 60.78 | 39.22 |
| partial>=0.5 | 120 | 12.0 | 52.50 | 8.33 | 53.33 | 61.67 | 0.00 | 38.33 |
| diff<0.5 | 648 | 64.8 | 45.68 | 4.48 | 45.83 | 50.31 | 0.00 | 49.69 |

| Escalation set | N | Current EM | Oracle EM | Oracle-current gap |
|---|---:|---:|---:|---:|
| sim < 0.25 | 601 | 45.59 | 50.08 | +4.49 |
| sim < 0.5 | 648 | 45.68 | 50.31 | +4.63 |
| sim < 0.75 | 743 | 45.76 | 51.28 | +5.52 |
| sim < 0.999999 | 768 | 46.74 | 52.08 | +5.34 |

## MuSiQue (N=1000)

| Metric | Value |
|---|---:|
| Current/replay EM | 24.70 |
| Direct EM | 10.20 |
| Decompose EM | 25.50 |
| Oracle EM | 28.60 |
| Similarity -> current-correct AUROC | 0.577 |
| Similarity -> oracle-correct AUROC | 0.564 |

| Consistency bucket | N | Share | Current EM | Direct EM | Decomp EM | Oracle EM | Both correct | Both wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exact | 166 | 16.6 | 37.95 | 37.95 | 37.95 | 37.95 | 37.95 | 62.05 |
| partial>=0.5 | 90 | 9.0 | 30.00 | 11.11 | 30.00 | 36.67 | 4.44 | 63.33 |
| diff<0.5 | 744 | 74.4 | 21.10 | 3.90 | 22.18 | 25.54 | 0.54 | 74.46 |

| Escalation set | N | Current EM | Oracle EM | Oracle-current gap |
|---|---:|---:|---:|---:|
| sim < 0.25 | 701 | 20.97 | 25.11 | +4.14 |
| sim < 0.5 | 744 | 21.10 | 25.54 | +4.44 |
| sim < 0.75 | 824 | 21.97 | 26.58 | +4.61 |
| sim < 0.999999 | 834 | 22.06 | 26.74 | +4.68 |

## PopQA (N=1000)

| Metric | Value |
|---|---:|
| Current/replay EM | 51.90 |
| Direct EM | 37.80 |
| Decompose EM | 52.90 |
| Oracle EM | 55.70 |
| Similarity -> current-correct AUROC | 0.693 |
| Similarity -> oracle-correct AUROC | 0.649 |

| Consistency bucket | N | Share | Current EM | Direct EM | Decomp EM | Oracle EM | Both correct | Both wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exact | 468 | 46.8 | 65.81 | 65.81 | 65.81 | 65.81 | 65.81 | 34.19 |
| partial>=0.5 | 193 | 19.3 | 60.62 | 22.28 | 63.73 | 65.28 | 20.73 | 34.72 |
| diff<0.5 | 339 | 33.9 | 27.73 | 7.96 | 28.91 | 36.28 | 0.59 | 63.72 |

| Escalation set | N | Current EM | Oracle EM | Oracle-current gap |
|---|---:|---:|---:|---:|
| sim < 0.25 | 255 | 18.43 | 29.02 | +10.59 |
| sim < 0.5 | 339 | 27.73 | 36.28 | +8.55 |
| sim < 0.75 | 512 | 38.87 | 46.29 | +7.42 |
| sim < 0.999999 | 532 | 39.66 | 46.80 | +7.14 |

## ALL (N=4000)

| Metric | Value |
|---|---:|
| Current/replay EM | 40.52 |
| Direct EM | 24.07 |
| Decompose EM | 39.77 |
| Oracle EM | 45.00 |
| Similarity -> current-correct AUROC | 0.647 |
| Similarity -> oracle-correct AUROC | 0.618 |

| Consistency bucket | N | Share | Current EM | Direct EM | Decomp EM | Oracle EM | Both correct | Both wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exact | 1205 | 30.1 | 58.42 | 58.42 | 58.42 | 58.42 | 58.42 | 41.58 |
| partial>=0.5 | 473 | 11.8 | 47.78 | 16.07 | 48.63 | 55.39 | 9.30 | 44.61 |
| diff<0.5 | 2322 | 58.0 | 29.76 | 7.88 | 28.29 | 35.92 | 0.26 | 64.08 |

| Escalation set | N | Current EM | Oracle EM | Oracle-current gap |
|---|---:|---:|---:|---:|
| sim < 0.25 | 2124 | 28.63 | 34.79 | +6.17 |
| sim < 0.5 | 2322 | 29.76 | 35.92 | +6.16 |
| sim < 0.75 | 2726 | 32.28 | 38.74 | +6.46 |
| sim < 0.999999 | 2795 | 32.81 | 39.21 | +6.40 |
