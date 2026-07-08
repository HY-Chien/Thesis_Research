# Direct-vs-Decompose Routing Gap

Setting: current final candidate traces, path-aware retrieval + original-question evidence union + rule-based verifier replay.

| Dataset | Direct EM | Decomp EM | Best pure EM | SEEK verifier EM | Oracle EM | Oracle - verifier | direct-only | decomp-only | both-wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | 30.30 | 30.50 | 30.50 | 35.50 | 41.60 | +6.10 | 111 | 113 | 584 |
| 2Wiki | 18.00 | 50.20 | 50.20 | 50.00 | 54.10 | +4.10 | 39 | 361 | 459 |
| MuSiQue | 10.20 | 25.50 | 25.50 | 24.70 | 28.60 | +3.90 | 31 | 184 | 714 |
| PopQA | 37.80 | 52.90 | 52.90 | 51.90 | 55.70 | +3.80 | 28 | 179 | 443 |

Interpretation: the oracle is not a method. It is a gold-label upper bound that quantifies the remaining headroom of the direct-vs-decompose routing decision.
