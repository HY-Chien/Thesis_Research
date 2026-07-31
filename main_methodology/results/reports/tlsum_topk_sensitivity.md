# tl_sum candidate-set sensitivity

Direct-path frozen confidence traces, n=1,000 x 4; score = sum_t -log(m_{k,t}); AUROC predicts wrong answers.

| k | 2Wiki | HotpotQA | MuSiQue | PopQA | Macro |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.684675 | 0.757433 | 0.725610 | 0.797275 | 0.741248 |
| 5 | 0.716518 | 0.756590 | 0.726156 | 0.797849 | 0.749278 |
| 10 | 0.721328 | 0.757868 | 0.723656 | 0.796454 | 0.749827 |

Paired within-dataset bootstrap, B=1000, seed=13:

| Comparison | Macro delta | 95% CI | two-sided p |
|---|---:|---:|---:|
| k=2 - k=5 | -0.008030 | [-0.015709, -0.000715] | 0.022 |
| k=10 - k=5 | +0.000548 | [-0.003416, +0.004757] | 0.824 |

Conclusion: k=2 is too narrow on these traces. k=5 and k=10 are on the same performance plateau; k=5 is retained as the smaller tested cutoff. This is an empirical hyperparameter, not a value implied by the entropy identity.
