# Direct-vs-Decompose 2x2 Counts (Current Final Traces)

Setting: current final candidate traces with path-aware retrieval, original-question evidence union, and type-aware rule-based verifier replay.

This is an answer-level analysis. It is different from retrieval coverage: a passage can contain the gold answer string, while the final decomposed answer can still be wrong because of decomposition drift, reasoning error, or synthesis error.

| Dataset | N | Both correct | Direct-only | Decomp-only | Both wrong | Direct EM | Decomp EM | Oracle EM | Decomp-only - Direct-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | 1000 | 141 | 39 | 361 | 459 | 18.00 | 50.20 | 54.10 | +322 |
| HotpotQA | 1000 | 192 | 111 | 113 | 584 | 30.30 | 30.50 | 41.60 | +2 |
| MuSiQue | 1000 | 71 | 31 | 184 | 714 | 10.20 | 25.50 | 28.60 | +153 |
| PopQA | 1000 | 350 | 28 | 179 | 443 | 37.80 | 52.90 | 55.70 | +151 |

Interpretation:

- Decomposition improves retrieval coverage, but answer-level correctness still depends on reasoning and final synthesis.
- In the current final traces, HotpotQA is no longer strongly direct-favored: direct-only = 111 and decomp-only = 113.
- The large both-wrong counts show that many errors cannot be fixed by routing alone; they need better retrieval, decomposition, or synthesis.
