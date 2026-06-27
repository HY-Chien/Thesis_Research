# PopQA Answer-containing Retrieval Recall

Date: 2026-06-24

This is a PopQA-specific retrieval proxy. Unlike HotpotQA, 2Wiki, and MuSiQue, PopQA does not provide gold multi-hop supporting-passage labels in the current SEEK file, so full-coverage@k is not defined. Instead, this metric checks whether retrieved passages contain the gold answer or one of its aliases.

Definition:

```text
answer-containing hit@k = percentage of questions where top-k retrieved passages contain the gold answer or an alias
```

Short aliases under 3 characters are ignored to reduce substring noise. Matching is normalized string containment over passage title + text.

## Open-domain Wiki, Path-aware Retrieval

Setting:

- Dataset: PopQA first 1000 examples
- Corpus: `wiki` / `psgs_w100` Elasticsearch index
- Original-question retrieval: paragraph text only, matching path-aware direct retrieval
- Decomposed retrieval: title + paragraph text, matching path-aware sub-question retrieval
- Decomposition trace: `results/popqa_1000_C_maxent0.3_3ds_terse.jsonl`
- Sub-question count: `{1: 1000}`

| k | Original question hit@k | Decomposed sub-question hit@k | Delta | Decomp-only | Original-only |
|---:|---:|---:|---:|---:|---:|
| 1 | 51.50 | **57.50** | +6.00 | 102 | 42 |
| 3 | 63.50 | **65.70** | +2.20 | 66 | 44 |
| 5 | 68.10 | **68.70** | +0.60 | 52 | 46 |
| 10 | 73.40 | **74.10** | +0.70 | 43 | 36 |
| 20 | 76.60 | **78.90** | +2.30 | 47 | 24 |
| 30 | 79.00 | **81.10** | +2.10 | 44 | 23 |

## Candidate-pool Rerank Sanity Check

This version reranks the 30 candidate passages already stored in `datasets/popqa/raw/dev.jsonl`. It is useful as a controlled sanity check, but the open-domain table above is closer to the final retrieval setting.

### Answer Only

Candidate pool contains the gold answer in 75.20% of questions.

| k | Original question hit@k | Decomposed sub-question hit@k | Delta | Decomp-only | Original-only |
|---:|---:|---:|---:|---:|---:|
| 1 | 26.10 | **34.10** | +8.00 | 140 | 60 |
| 3 | 42.50 | **54.30** | +11.80 | 161 | 43 |
| 5 | 51.40 | **62.70** | +11.30 | 139 | 26 |
| 10 | 62.40 | **68.40** | +6.00 | 80 | 20 |
| 20 | 71.90 | **73.90** | +2.00 | 28 | 8 |
| 30 | 75.20 | 75.20 | +0.00 | 0 | 0 |

### Answer + Aliases

Candidate pool contains the gold answer or alias in 81.40% of questions.

| k | Original question hit@k | Decomposed sub-question hit@k | Delta | Decomp-only | Original-only |
|---:|---:|---:|---:|---:|---:|
| 1 | 29.00 | **38.10** | +9.10 | 154 | 63 |
| 3 | 47.30 | **59.90** | +12.60 | 173 | 47 |
| 5 | 57.30 | **68.20** | +10.90 | 139 | 30 |
| 10 | 69.00 | **74.40** | +5.40 | 77 | 23 |
| 20 | 78.30 | **80.00** | +1.70 | 24 | 7 |
| 30 | 81.40 | 81.40 | +0.00 | 0 | 0 |

## Interpretation

PopQA is still decomposed by SEEK. The reason it is excluded from full-coverage@k is only that full-coverage requires gold supporting-passage labels, which the current PopQA file does not provide.

The answer-containing proxy shows that decomposition still improves retrieval ranking for PopQA, especially at low k. The gain is smaller in the open-domain path-aware setting than in the candidate-pool rerank setting because many PopQA questions already contain the target entity directly, so the original question is already a strong retrieval query.
