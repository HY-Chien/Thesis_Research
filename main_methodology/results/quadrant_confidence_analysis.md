# Direct-path tl_sum by Correctness Quadrant

Setting: final prompt-fix-A candidates with the tl_sum step gate, path-aware retrieval, original-question evidence union, and the final all-tl_sum verifier (direct tau=0.0005; post-decomposition tau=0.0158).

The signal is direct-path tail-leakage surprisal (`direct_tl`); higher values indicate greater uncertainty.

## 1. All correctness quadrants

| Dataset | Quadrant | n | Mean tl_sum | Median tl_sum | Final router picks direct |
|---|---|---:|---:|---:|---:|
| HotpotQA | both_correct | 225 | 0.01278 | 0.000907 | 0.0% |
| HotpotQA | direct_only | 78 | 0.01944 | 0.001989 | 50.0% |
| HotpotQA | decomp_only | 132 | 0.08865 | 0.011273 | 15.9% |
| HotpotQA | both_wrong | 565 | 0.13039 | 0.020337 | 18.9% |
| 2Wiki | both_correct | 140 | 0.03485 | 0.006896 | 0.0% |
| 2Wiki | direct_only | 40 | 0.03678 | 0.005550 | 10.0% |
| 2Wiki | decomp_only | 350 | 0.10643 | 0.030046 | 2.0% |
| 2Wiki | both_wrong | 470 | 0.20103 | 0.053470 | 3.2% |
| MuSiQue | both_correct | 76 | 0.04983 | 0.004887 | 1.3% |
| MuSiQue | direct_only | 26 | 0.11353 | 0.021072 | 38.5% |
| MuSiQue | decomp_only | 191 | 0.19852 | 0.051154 | 5.8% |
| MuSiQue | both_wrong | 707 | 0.25054 | 0.078252 | 14.1% |
| PopQA | both_correct | 345 | 0.00977 | 0.000462 | 0.0% |
| PopQA | direct_only | 33 | 0.08372 | 0.013379 | 3.0% |
| PopQA | decomp_only | 193 | 0.03167 | 0.007076 | 1.0% |
| PopQA | both_wrong | 429 | 0.10258 | 0.021248 | 1.6% |

## 2. Both-correct versus both-wrong

AUROC ranks both-wrong above both-correct. Confidently wrong is the share of both-wrong examples at or below the both-correct median tl_sum.

| Dataset | Both-correct n | Both-wrong n | Mean both-correct | Mean both-wrong | Median both-correct | Median both-wrong | AUROC | Confidently wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | 225 | 565 | 0.01278 | 0.13039 | 0.000907 | 0.020337 | 0.777 | 15.6% |
| 2Wiki | 140 | 470 | 0.03485 | 0.20103 | 0.006896 | 0.053470 | 0.747 | 21.1% |
| MuSiQue | 76 | 707 | 0.04983 | 0.25054 | 0.004887 | 0.078252 | 0.763 | 12.4% |
| PopQA | 345 | 429 | 0.00977 | 0.10258 | 0.000462 | 0.021248 | 0.845 | 4.9% |
| Pooled | 786 | 2171 | 0.01897 | 0.17932 | 0.001129 | 0.042399 | 0.816 | 9.3% |

## 3. Final routing inside disagreement quadrants

| Dataset | Quadrant | n | Final router picks correct path |
|---|---|---:|---:|
| HotpotQA | direct_only | 78 | 50.0% |
| HotpotQA | decomp_only | 132 | 84.1% |
| 2Wiki | direct_only | 40 | 10.0% |
| 2Wiki | decomp_only | 350 | 98.0% |
| MuSiQue | direct_only | 26 | 38.5% |
| MuSiQue | decomp_only | 191 | 94.2% |
| PopQA | direct_only | 33 | 3.0% |
| PopQA | decomp_only | 193 | 99.0% |

## Interpretation

Direct-path tl_sum separates both-wrong from both-correct examples on every dataset (pooled AUROC 0.816), but the distributions overlap. Using the pooled both-correct median still leaves 9.3% of both-wrong examples confidently wrong. Confidence is therefore informative at the population level without being a reliable per-question correctness test.
