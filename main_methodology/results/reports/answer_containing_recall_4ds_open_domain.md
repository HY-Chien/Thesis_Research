# Four-dataset Open-domain Answer-containing Retrieval Recall

Date: 2026-07-06

Setting: final open-domain wiki retrieval condition, recomputed with the same answer-containing recall implementation family as the MuSiQue gold-decomposition diagnostic. Original-question retrieval uses paragraph text only; decomposed sub-question retrieval uses title + paragraph text, matching path-aware retrieval.

`answer-containing hit@k` means that the retrieved passages contain the gold answer or one of its aliases in the passage title or text. Yes/no questions are excluded because yes/no strings are not meaningful evidence strings.

Note: this file is a unified rerun for the four-dataset figure. Compare numbers within this file; do not mix them with older saved runs or with the MuSiQue-only oracle arms.

## Main Table, k=5

| Dataset | N eval | Original-query hit@5 | Decomposed-query hit@5 | Union hit@5 | Decomp - Original |
|---|---:|---:|---:|---:|---:|
| 2Wiki | 917 | 32.61 | **60.63** | **64.89** | +28.03 |
| HotpotQA | 941 | 47.40 | **52.92** | **64.61** | +5.53 |
| MuSiQue | 1000 | 19.00 | **38.80** | **43.90** | +19.80 |
| PopQA | 1000 | 71.20 | **71.10** | **75.90** | -0.10 |

## k Sweep

### 2Wiki

N eval = 917; skipped yes/no = 83; sub-question counts = `{1: 8, 2: 698, 3: 2, 4: 209}`.

| k | Original | Decomp | Union | Decomp - Original | Decomp-only | Original-only |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17.99 | 48.53 | 51.15 | +30.53 | 304 | 24 |
| 3 | 27.92 | 57.03 | 61.07 | +29.12 | 304 | 37 |
| 5 | 32.61 | 60.63 | 64.89 | +28.03 | 296 | 39 |
| 8 | 38.39 | 63.03 | 67.72 | +24.65 | 269 | 43 |
| 10 | 41.22 | 64.78 | 69.03 | +23.56 | 255 | 39 |

### HotpotQA

N eval = 941; skipped yes/no = 59; sub-question counts = `{1: 9, 2: 749, 3: 135, 4: 42, 5: 4, 6: 1, 12: 1}`.

| k | Original | Decomp | Union | Decomp - Original | Decomp-only | Original-only |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 28.16 | 40.91 | 48.57 | +12.75 | 192 | 72 |
| 3 | 41.13 | 49.10 | 58.98 | +7.97 | 168 | 93 |
| 5 | 47.40 | 52.92 | 64.61 | +5.53 | 162 | 110 |
| 8 | 52.50 | 56.32 | 69.50 | +3.83 | 160 | 124 |
| 10 | 54.73 | 57.92 | 71.31 | +3.19 | 156 | 126 |

### MuSiQue

N eval = 1000; skipped yes/no = 0; sub-question counts = `{1: 3, 2: 963, 3: 33, 4: 1}`.

| k | Original | Decomp | Union | Decomp - Original | Decomp-only | Original-only |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8.50 | 22.00 | 24.50 | +13.50 | 160 | 25 |
| 3 | 14.10 | 34.70 | 38.30 | +20.60 | 242 | 36 |
| 5 | 19.00 | 38.80 | 43.90 | +19.80 | 249 | 51 |
| 8 | 23.60 | 42.70 | 47.60 | +19.10 | 240 | 49 |
| 10 | 25.60 | 46.00 | 50.90 | +20.40 | 253 | 49 |

### PopQA

N eval = 1000; skipped yes/no = 0; sub-question counts = `{1: 1000}`.

| k | Original | Decomp | Union | Decomp - Original | Decomp-only | Original-only |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 54.20 | 60.00 | 64.30 | +5.80 | 101 | 43 |
| 3 | 66.30 | 68.00 | 72.70 | +1.70 | 64 | 47 |
| 5 | 71.20 | 71.10 | 75.90 | -0.10 | 47 | 48 |
| 8 | 75.10 | 74.60 | 79.40 | -0.50 | 43 | 48 |
| 10 | 77.00 | 76.40 | 80.70 | -0.60 | 37 | 43 |

## Interpretation

The unified rerun supports the same qualitative conclusion as the earlier four-dataset figure: decomposition substantially improves answer-containing retrieval on the multi-hop datasets, with the largest gains on 2Wiki and MuSiQue. PopQA remains a near-single-hop case, so the retrieval gain is small. This table should be used for the four-dataset figure when presenting it next to the MuSiQue gold-decomposition diagnostic.
