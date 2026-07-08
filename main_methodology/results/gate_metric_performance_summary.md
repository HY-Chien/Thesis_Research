# Gate / Metric Performance Summary

Date: 2026-06-28

Scope: existing 1000-question traces over HotpotQA, 2Wiki, MuSiQue, and PopQA.
The comparable full-system numbers are the final direct-vs-decompose outputs.
Headline F1 follows the alias-aware tables where available; path-consistency
replay F1 is a lightweight replay diagnostic and should be treated as secondary.
The full-system variant table mixes verifier path choices; for the isolated
union effect, compare decomposed candidates only.

## Full-System Variants

| Variant | HotpotQA EM/F1 | 2Wiki EM/F1 | MuSiQue EM/F1 | PopQA EM/F1 | Avg EM/F1 | Note |
|---|---:|---:|---:|---:|---:|---|
| Path-aware baseline | 31.80 / 41.25 | 46.90 / 53.51 | 24.60 / 33.55 | 51.80 / 54.19 | 38.78 / 45.62 | Before original-question evidence union |
| + Original-question evidence union | 34.30 / 44.26 | 49.90 / 56.42 | 24.80 / 32.87 | 52.00 / 54.83 | 40.25 / 47.09 | Main method change |
| Final candidate: union + type-aware verifier, tau=0.3 | 35.50 / 46.11 | 50.00 / 56.58 | 24.70 / 33.06 | 51.90 / 54.71 | **40.52 / 47.61** | Best stable full-system setting |

## Original-Question Evidence Union: Path-Local Effect

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

## Entropy Gate Tau Sweep

| Retrieval tau | Avg retrieval rate | Avg EM/F1 | Interpretation |
|---:|---:|---:|---|
| 0.3 | 94.16 | **40.52 / 47.61** | Accuracy-first canonical setting |
| 0.5 | 91.32 | 40.45 / 47.52 | Nearly same accuracy, slightly cheaper |
| 0.8 | 85.97 | 39.98 / 47.04 | Noticeable cost saving, small accuracy drop |
| 1.0 | 82.35 | 39.50 / 46.63 | More cost saving, larger drop |

## Final Aggregation Replays

| Final aggregation rule | Macro EM | Macro F1 | Takeaway |
|---|---:|---:|---|
| Current final candidate | 40.52 | 45.09 | Replay F1 uses simplified token-F1 |
| Path-consistency: partial similarity -> decompose, tau=0.25 | **40.80** | 45.08 | Highest replay EM, but only +0.28 and F1 does not improve |
| Path-consistency: partial similarity -> decompose, tau=0.5 | 40.62 | **45.10** | Tiny change |
| Path-consistency: low similarity -> decompose | 39.67 | 43.52 | Hurts HotpotQA badly |
| Path-consistency: any disagreement -> decompose | 39.77 | 43.53 | Too aggressive |

## Signal Quality

| Signal | Evidence | Best role |
|---|---|---|
| Max entropy | ALL AUROC 0.686 for separating correct vs wrong; correct mean entropy 0.268 vs wrong 0.602 | Main retrieval uncertainty signal |
| Type-aware / QA structure rules | Full-system Avg improves from 40.25 / 47.09 to 40.52 / 47.61 after union | Final verifier guard, not confidence |
| Path consistency | Exact direct/decompose agreement has much higher EM, but direct use only gives +0.28 EM at best | Diagnostic or escalation trigger |
| Self-certainty | Equivalent to inverse entropy when computed from the same answer-token distribution | Related work / terminology comparison, not a separate gain |

## Current Best

The best stable full-system configuration remains:

```text
SEEK + path-aware retrieval
+ original-question evidence union
+ conservative type-aware final verifier
+ max-entropy retrieval gate tau = 0.3
```

Result:

```text
Average: 40.52 EM / 47.61 F1
HotpotQA: 35.50 / 46.11
2Wiki:    50.00 / 56.58
MuSiQue:  24.70 / 33.06
PopQA:    51.90 / 54.71
```

Path-consistency can produce the highest replay EM, 40.80, but the gain is too
small and not robust enough to replace the current final verifier.
