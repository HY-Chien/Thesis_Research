# Final Performance Table

Date: 2026-06-24

Current final candidate = **SEEK + path-aware retrieval + original-question evidence union + rule-based verifier with conservative type-aware fallback, tau=0.3**.

The type-aware fallback is a deterministic refinement of the existing rule-based verifier. It only changes the final direct-vs-decompose choice and does not rerun generation.

## Main Table

| Dataset | Path-aware baseline EM/F1 | + Original-question evidence union EM/F1 | Final candidate (tau=0.3) EM/F1 | DeepRAG EM/F1 | Final - DeepRAG |
|---|---:|---:|---:|---:|---:|
| HotpotQA | 31.80 / 41.25 | 34.30 / 44.26 | **35.50 / 46.11** | 40.70 / 51.50 | -5.20 / -5.39 |
| 2Wiki | 46.90 / 53.51 | 49.90 / 56.42 | **50.00 / 56.58** | 48.10 / 53.30 | +1.90 / +3.28 |
| MuSiQue | 24.60 / 33.55 | 24.80 / 32.87 | **24.70 / 33.06** | 22.50 / 30.40 | +2.20 / +2.66 |
| PopQA | 51.80 / 54.19 | 52.00 / 54.83 | **51.90 / 54.71** | 42.50 / 47.80 | +9.40 / +6.91 |
| **Average** | 38.78 / 45.62 | 40.25 / 47.09 | **40.52 / 47.61** | 38.45 / 45.75 | +2.07 / +1.86 |

## Notes

- Original-question evidence union is the main method change: each retrieved decomposed sub-question context is augmented with original-question top-k evidence.
- Type-aware fallback is part of the rule-based verifier implementation detail, not a separately trained verifier model.
- HotpotQA remains the only dataset below DeepRAG, but the gap improves from -8.90 EM to -5.20 EM. The final setting uses tau=0.3 as the accuracy-first canonical threshold.
