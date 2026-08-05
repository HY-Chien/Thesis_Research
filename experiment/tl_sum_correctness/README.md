# tl_sum vs. Direct-Path Correctness

Stacked-bar counterpart of the earlier MAX_ENTROPY figure, recomputed with
the final top-5 tail-leakage score:

\[
\mathrm{tl\_sum}(x)=\sum_t-\log m_{5,t},\qquad
m_{5,t}=\sum_{v\in\operatorname{Top5}(p_t)}p_t(v).
\]

## Data and labels

- Frozen direct greedy traces:
  `Research/main_methodology/results/conf_battery_{2wiki,hotpotqa,musique,popqa}.jsonl`
- 1,000 questions per dataset; 4,000 pooled.
- Correctness is the stored alias-aware exact-match field `em`.
- Eight equal-frequency bins, 500 questions per bin.

This is intentionally not computed from the earlier `*_msample.jsonl` files:
those contain `mean_max_ent` for self-consistency answers but do not contain
the per-token top-5 mass needed for tl_sum.

## Result

Spearman(tl_sum, error) = **+0.418623**, p = **1.629e-169**. Error rises from
32.8% in the lowest tl_sum bin to 97.0% in the highest bin.

This is a pooled descriptive association. The direction is also positive
within every dataset (2Wiki +0.288, HotpotQA +0.408, MuSiQue +0.237,
PopQA +0.500), although dataset composition makes the pooled coefficient
somewhat larger.

| Bin | Median tl_sum | Accuracy | Error |
|---:|---:|---:|---:|
| 1 | 0.000171 | 67.2% | 32.8% |
| 2 | 0.001164 | 38.4% | 61.6% |
| 3 | 0.004412 | 26.0% | 74.0% |
| 4 | 0.011919 | 23.8% | 76.2% |
| 5 | 0.029475 | 14.0% | 86.0% |
| 6 | 0.066040 | 11.6% | 88.4% |
| 7 | 0.172737 | 8.6% | 91.4% |
| 8 | 0.523315 | 3.0% | 97.0% |

## Reproduce

```bash
/home/S113065528/miniconda3/envs/thesis/bin/python \
  Research/experiment/tl_sum_correctness/code/plot_tlsum_stacked.py
```

The script writes PNG, vector PDF, and SVG versions under `figures/`.
