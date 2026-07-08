# MAX_ENTROPY vs Correctness — summary

Signal: `mean_max_ent` from `*_msample.jsonl` (self-consistency).
Correctness: normalized EM / token-F1>=0.5 of SC majority vs gold.

| Dataset | n | Acc% | mean ent (correct) | mean ent (wrong) | Spearman(ent,err) | point-biserial(ent,correct) | AUROC |
|---|---|---|---|---|---|---|---|
| 2WikiMultiHopQA | 1000 | 31.3 | 0.353 | 0.568 | +0.206 | -0.204 | 0.627 |
| HotpotQA | 1000 | 63.2 | 0.213 | 0.434 | +0.280 | -0.264 | 0.655 |
| MuSiQue | 1000 | 32.5 | 0.338 | 0.672 | +0.263 | -0.254 | 0.658 |
| PopQA | 1000 | 37.5 | 0.228 | 0.663 | +0.414 | -0.362 | 0.741 |
| ALL | 4000 | 41.1 | 0.268 | 0.602 | +0.326 | -0.306 | 0.686 |

## Follow-up analyses

- `high_low_confidence_error_analysis.md` (2026-06-28) — tail-decile error
  anatomy: all 169 lowest-entropy errors are deterministic wrong paths
  (`n_uniq=1, sc_agree=1.0`); 5 failure categories.
- `confidence_popularity_bias.md` (2026-07-03,
  `code/analyze_confidence_popularity_bias.py`) — PopQA confidence vs entity
  popularity (`s_pop`, original PopQA TSV). Result: **no raw
  popularity→confidence bias** (Spearman |rho|<0.05 n.s., also within wrong
  answers); AUROC(ent→wrong) stable ~0.72–0.75 across popularity bands. The
  popularity-skewed part is the metric: popular-entity confident "errors" are
  disproportionately alias artifacts (28% alias-recovered) while rare-entity
  confident errors are 100% genuine. Confident hallucination is a
  deterministic-wrong-path problem, not a popularity-bias problem.
