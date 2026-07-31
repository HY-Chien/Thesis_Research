# Direct-vs-Decompose Routing Gap (Final all-tl_sum Router)

Setting: final prompt-fix-A candidates and all-tl_sum verifier replay with direct tau=0.0005 and post-decomposition tau=0.0158.

| Dataset | Direct EM | Decomp EM | Best pure EM | SEEK final EM | Oracle EM | Oracle - SEEK | Direct choices | Decomp choices | Direct-only | Decomp-only | Both-wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | 30.30 | 35.70 | 35.70 | 37.50 | 43.50 | +6.00 | 167 | 833 | 78 | 132 | 565 |
| 2Wiki | 18.00 | 49.00 | 49.00 | 48.70 | 53.00 | +4.30 | 26 | 974 | 40 | 350 | 470 |
| MuSiQue | 10.20 | 26.70 | 26.70 | 26.60 | 29.30 | +2.70 | 122 | 878 | 26 | 191 | 707 |
| PopQA | 37.80 | 53.80 | 53.80 | 53.70 | 57.10 | +3.40 | 10 | 990 | 33 | 193 | 429 |

The oracle is a gold-label upper bound, not a deployable method. Its gap over SEEK isolates the remaining direct-vs-decompose routing headroom; both-wrong questions remain outside that bound.
