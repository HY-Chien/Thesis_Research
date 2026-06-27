# Four-dataset Answer-containing Retrieval Recall

Date: 2026-06-24

This is a four-dataset retrieval proxy for comparing original-question retrieval with decomposed sub-question retrieval under one shared metric. It replaces the three-dataset full-coverage table when PopQA must be included.

Definition:

```text
answer-containing hit@k = percentage of evaluable questions where the top-k retrieved passages contain the gold answer or one of its aliases
```

Setting:

- Candidate pool: per-question candidate/distractor paragraphs, matching the original M3 recall setting.
- Original-query retrieval: BM25 with the original question.
- Decomposed retrieval: BM25 max-score over decomposed sub-questions, using the same top-k budget.
- Metric: answer + aliases string containment over passage title + text.
- Short aliases under 3 characters are ignored.
- Yes/no questions are excluded because `yes` and `no` are not meaningful evidence strings.

## Main Table, k=5

| Dataset | N eval | Skipped yes/no | Original-query answer hit@5 | Decomposed-query answer hit@5 | Gain | Decomp-only | Original-only |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | 917 | 83 | 72.63 | **93.89** | +21.26 | 212 | 17 |
| HotpotQA | 941 | 59 | 81.08 | **87.89** | +6.80 | 114 | 50 |
| MuSiQue | 1000 | 0 | 41.40 | **73.70** | +32.30 | 362 | 39 |
| PopQA | 1000 | 0 | 57.30 | **68.20** | +10.90 | 139 | 30 |

## Full k Sweep

### 2Wiki

| k | Original | Decomp | Gain | Decomp-only | Original-only |
|---:|---:|---:|---:|---:|---:|
| 1 | 23.88 | **37.30** | +13.41 | 174 | 51 |
| 3 | 59.76 | **87.02** | +27.26 | 286 | 36 |
| 5 | 72.63 | **93.89** | +21.26 | 212 | 17 |
| 10 | 98.58 | 98.58 | +0.00 | 0 | 0 |

### HotpotQA

| k | Original | Decomp | Gain | Decomp-only | Original-only |
|---:|---:|---:|---:|---:|---:|
| 1 | 41.02 | **48.88** | +7.86 | 159 | 85 |
| 3 | 71.73 | **81.08** | +9.35 | 155 | 67 |
| 5 | 81.08 | **87.89** | +6.80 | 114 | 50 |
| 10 | 98.62 | 98.62 | +0.00 | 0 | 0 |

### MuSiQue

| k | Original | Decomp | Gain | Decomp-only | Original-only |
|---:|---:|---:|---:|---:|---:|
| 1 | 12.60 | **33.00** | +20.40 | 240 | 36 |
| 3 | 30.80 | **66.60** | +35.80 | 396 | 38 |
| 5 | 41.40 | **73.70** | +32.30 | 362 | 39 |
| 10 | 59.30 | **82.90** | +23.60 | 274 | 38 |

### PopQA

| k | Original | Decomp | Gain | Decomp-only | Original-only |
|---:|---:|---:|---:|---:|---:|
| 1 | 29.00 | **38.10** | +9.10 | 154 | 63 |
| 3 | 47.30 | **59.90** | +12.60 | 173 | 47 |
| 5 | 57.30 | **68.20** | +10.90 | 139 | 30 |
| 10 | 69.00 | **74.40** | +5.40 | 77 | 23 |

## Interpretation

The four-dataset proxy supports the same conclusion as the stricter multi-hop full-coverage metric: decomposition improves retrieval. The largest gains are on MuSiQue and 2Wiki, where intermediate entities matter most. HotpotQA improves less because many supporting entities are already visible in the original question. PopQA also improves because decomposition rewrites many questions into entity-centric retrieval queries, although it is not a multi-hop supporting-evidence dataset.
