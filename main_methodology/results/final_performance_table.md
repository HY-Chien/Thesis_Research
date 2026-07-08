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


## Union Ablation Scope

The main table above is a full-system progression. It is useful for the final headline, but it is not the cleanest isolation of original-question evidence union because the final output also depends on whether the verifier selects the direct or decomposition candidate.

Since union only changes the decomposition path, the path-local ablation should compare decomposed candidates:

| Dataset | Decomp w/o union EM/F1 | Decomp + union EM/F1 | Change EM/F1 |
|---|---:|---:|---:|
| HotpotQA | 25.00 / 33.42 | **30.50 / 39.49** | +5.50 / +6.07 |
| 2Wiki | 47.20 / 53.63 | **50.20 / 56.65** | +3.00 / +3.03 |
| MuSiQue | 24.70 / 33.10 | **25.50 / 33.40** | +0.80 / +0.30 |
| PopQA | 51.50 / 53.68 | **52.90 / 55.20** | +1.40 / +1.52 |
| **Average** | **37.10 / 43.46** | **39.78 / 46.19** | **+2.68 / +2.73** |

## Final Path Selection Rate

| Dataset | Direct selected | Decompose selected |
|---|---:|---:|
| HotpotQA | 274 / 1000 = **27.4%** | 726 / 1000 = **72.6%** |
| 2Wiki | 65 / 1000 = **6.5%** | 935 / 1000 = **93.5%** |
| MuSiQue | 182 / 1000 = **18.2%** | 818 / 1000 = **81.8%** |
| PopQA | 37 / 1000 = **3.7%** | 963 / 1000 = **96.3%** |
| **Pooled** | 558 / 4000 = **14.0%** | 3442 / 4000 = **86.1%** |

## Notes

- Original-question evidence union is the main method change inside the decomposition path: each retrieved decomposed sub-question context is augmented with original-question top-k evidence. Cite the path-local table above when arguing for the union effect.
- Type-aware fallback is part of the rule-based verifier implementation detail, not a separately trained verifier model.
- HotpotQA remains the only dataset below DeepRAG, but the gap improves from -8.90 EM to -5.20 EM. The final setting uses tau=0.3 as the accuracy-first canonical threshold.
