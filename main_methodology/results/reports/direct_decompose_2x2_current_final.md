# Direct-vs-Decompose 2x2 Counts (Final all-tl_sum Traces)

Setting: final prompt-fix-A traces with the tl_sum step gate, path-aware retrieval, and original-question evidence union.

| Dataset | N | Both correct | Direct-only | Decomp-only | Both wrong | Direct EM | Decomp EM | Oracle EM | Decomp-only - Direct-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | 1000 | 225 | 78 | 132 | 565 | 30.30 | 35.70 | 43.50 | +54 |
| 2Wiki | 1000 | 140 | 40 | 350 | 470 | 18.00 | 49.00 | 53.00 | +310 |
| MuSiQue | 1000 | 76 | 26 | 191 | 707 | 10.20 | 26.70 | 29.30 | +165 |
| PopQA | 1000 | 345 | 33 | 193 | 429 | 37.80 | 53.80 | 57.10 | +160 |

The two paths remain complementary. Decompose-only wins dominate on all four datasets, but HotpotQA retains 78 direct-only wins. Both-wrong cases remain outside the reach of routing alone.
