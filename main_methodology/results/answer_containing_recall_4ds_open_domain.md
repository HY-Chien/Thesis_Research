# Four-dataset Open-domain Answer-containing Retrieval Recall

Date: 2026-06-24

Setting: final open-domain wiki retrieval condition. Original-question retrieval uses paragraph text only; decomposed sub-question retrieval uses title + paragraph text, matching path-aware retrieval.

`answer-containing hit@k` means that the retrieved passages contain the gold answer or one of its aliases. Yes/no questions are excluded because yes/no strings are not meaningful evidence strings.

## Main Table, k=5

| Dataset | N eval | Original-query hit@5 | Decomposed-query hit@5 | Union hit@5 | Decomp - Original |
|---|---:|---:|---:|---:|---:|
| 2Wiki | 917 | 31.73 | **62.38** | **65.32** | +30.64 |
| HotpotQA | 941 | 46.55 | **56.54** | **65.25** | +9.99 |
| MuSiQue | 1000 | 16.90 | **37.40** | **41.60** | +20.50 |
| PopQA | 1000 | 68.10 | **68.70** | **73.30** | +0.60 |

## k Sweep

### 2Wiki

N eval = 917; skipped yes/no = 83; sub-question counts = `{1: 8, 2: 698, 3: 2, 4: 209}`.

| k | Original | Decomp | Union | Decomp - Original | Decomp-only | Original-only |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17.45 | 49.40 | 51.36 | +31.95 | 311 | 18 |
| 3 | 26.94 | 58.23 | 61.29 | +31.30 | 315 | 28 |
| 5 | 31.73 | 62.38 | 65.32 | +30.64 | 308 | 27 |
| 8 | 37.40 | 65.21 | 68.27 | +27.81 | 283 | 28 |
| 10 | 40.02 | 66.52 | 69.25 | +26.50 | 268 | 25 |

### HotpotQA

N eval = 941; skipped yes/no = 59; sub-question counts = `{1: 9, 2: 749, 3: 135, 4: 42, 5: 4, 6: 1, 12: 1}`.

| k | Original | Decomp | Union | Decomp - Original | Decomp-only | Original-only |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 27.52 | 45.27 | 50.48 | +17.75 | 216 | 49 |
| 3 | 40.38 | 52.92 | 60.04 | +12.54 | 185 | 67 |
| 5 | 46.55 | 56.54 | 65.25 | +9.99 | 176 | 82 |
| 8 | 51.01 | 59.51 | 68.76 | +8.50 | 167 | 87 |
| 10 | 53.24 | 60.57 | 70.14 | +7.33 | 159 | 90 |

### MuSiQue

N eval = 1000; skipped yes/no = 0; sub-question counts = `{1: 3, 2: 963, 3: 33, 4: 1}`.

| k | Original | Decomp | Union | Decomp - Original | Decomp-only | Original-only |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.30 | 21.10 | 23.40 | +13.80 | 161 | 23 |
| 3 | 12.30 | 32.70 | 36.00 | +20.40 | 237 | 33 |
| 5 | 16.90 | 37.40 | 41.60 | +20.50 | 247 | 42 |
| 8 | 21.40 | 41.20 | 45.20 | +19.80 | 238 | 40 |
| 10 | 23.20 | 44.70 | 48.60 | +21.50 | 254 | 39 |

### PopQA

N eval = 1000; skipped yes/no = 0; sub-question counts = `{1: 1000}`.

| k | Original | Decomp | Union | Decomp - Original | Decomp-only | Original-only |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 51.50 | 57.50 | 61.70 | +6.00 | 102 | 42 |
| 3 | 63.50 | 65.70 | 70.10 | +2.20 | 66 | 44 |
| 5 | 68.10 | 68.70 | 73.30 | +0.60 | 52 | 46 |
| 8 | 71.70 | 72.30 | 76.40 | +0.60 | 47 | 41 |
| 10 | 73.40 | 74.10 | 77.70 | +0.70 | 43 | 36 |
