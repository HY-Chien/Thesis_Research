# Per-Quadrant Confidence Analysis (Direct-vs-Decompose 2x2, slide p45)

Setting: current final candidate traces with path-aware retrieval, original-question
evidence union, and type-aware rule-based verifier replay (same files as
`direct_decompose_2x2_current_final.md`). Question-level uncertainty signals:

```text
direct_ent        = direct.skip_probe_ent          (skip-decompose probe entropy)
decomp_max_ent    = max over steps of max_ent      (canonical gate signal)
decomp_mean_ent   = mean over steps of mean_ent
decomp_neg_min_lp = -min over steps of min_lp
higher value = more uncertain, for every signal
```

## 1. Confidence by quadrant

| Dataset | Quadrant | n | direct_ent mean | direct_ent median | decomp_max_ent mean | decomp_max_ent median | verifier picks direct % |
|---|---|---:|---:|---:|---:|---:|---:|
| 2Wiki | both_correct | 141 | 0.796 | 0.695 | 2.914 | 3.049 | 0.0 |
| 2Wiki | direct_only | 39 | 0.834 | 0.578 | 3.398 | 3.508 | 28.2 |
| 2Wiki | decomp_only | 361 | 1.232 | 1.029 | 3.471 | 3.565 | 3.6 |
| 2Wiki | both_wrong | 459 | 1.495 | 1.229 | 3.608 | 3.742 | 8.9 |
| HotpotQA | both_correct | 192 | 0.407 | 0.171 | 3.148 | 3.229 | 0.0 |
| HotpotQA | direct_only | 111 | 0.481 | 0.306 | 3.260 | 3.183 | 67.6 |
| HotpotQA | decomp_only | 113 | 1.064 | 0.835 | 3.278 | 3.299 | 22.1 |
| HotpotQA | both_wrong | 584 | 1.151 | 0.822 | 3.448 | 3.470 | 29.8 |
| MuSiQue | both_correct | 71 | 0.742 | 0.525 | 2.475 | 2.587 | 1.4 |
| MuSiQue | direct_only | 31 | 1.121 | 0.772 | 3.002 | 2.944 | 38.7 |
| MuSiQue | decomp_only | 184 | 1.510 | 1.237 | 2.726 | 2.750 | 10.9 |
| MuSiQue | both_wrong | 714 | 1.689 | 1.352 | 3.276 | 3.440 | 20.9 |
| PopQA | both_correct | 350 | 0.351 | 0.145 | 2.905 | 3.111 | 0.0 |
| PopQA | direct_only | 28 | 1.186 | 0.972 | 3.110 | 3.307 | 7.1 |
| PopQA | decomp_only | 179 | 0.804 | 0.677 | 2.907 | 3.017 | 6.7 |
| PopQA | both_wrong | 443 | 1.185 | 0.926 | 3.080 | 3.220 | 5.2 |

## 2. Both-correct vs both-wrong separation

AUROC of each uncertainty signal at ranking both_wrong above both_correct
(disagreement quadrants excluded).

| Dataset | Signal | mean both_correct | mean both_wrong | AUROC (wrong=pos) | Cohen's d |
|---|---|---:|---:|---:|---:|
| 2Wiki | direct_ent | 0.796 | 1.495 | 0.688 | +0.67 |
| 2Wiki | decomp_max_ent | 2.914 | 3.608 | 0.661 | +0.61 |
| 2Wiki | decomp_mean_ent | 1.315 | 1.357 | 0.507 | +0.06 |
| 2Wiki | decomp_neg_min_lp | 1.682 | 1.752 | 0.518 | +0.10 |
| HotpotQA | direct_ent | 0.407 | 1.151 | 0.750 | +0.77 |
| HotpotQA | decomp_max_ent | 3.148 | 3.448 | 0.553 | +0.22 |
| HotpotQA | decomp_mean_ent | 1.184 | 1.304 | 0.549 | +0.16 |
| HotpotQA | decomp_neg_min_lp | 1.347 | 1.523 | 0.564 | +0.25 |
| MuSiQue | direct_ent | 0.742 | 1.689 | 0.752 | +0.78 |
| MuSiQue | decomp_max_ent | 2.475 | 3.276 | 0.659 | +0.59 |
| MuSiQue | decomp_mean_ent | 0.826 | 1.229 | 0.673 | +0.56 |
| MuSiQue | decomp_neg_min_lp | 1.007 | 1.455 | 0.676 | +0.62 |
| PopQA | direct_ent | 0.351 | 1.185 | 0.819 | +1.09 |
| PopQA | decomp_max_ent | 2.905 | 3.080 | 0.538 | +0.14 |
| PopQA | decomp_mean_ent | 1.756 | 1.887 | 0.541 | +0.12 |
| PopQA | decomp_neg_min_lp | 1.278 | 1.392 | 0.547 | +0.17 |
| ALL | direct_ent | 0.485 | 1.404 | 0.782 | +0.89 |
| ALL | decomp_max_ent | 2.928 | 3.352 | 0.588 | +0.33 |
| ALL | decomp_mean_ent | 1.440 | 1.408 | 0.503 | -0.04 |
| ALL | decomp_neg_min_lp | 1.345 | 1.522 | 0.568 | +0.24 |

## 3. Confidently-wrong rate inside both_wrong

Fraction of both_wrong questions whose uncertainty is at or below the *median*
uncertainty of both_correct questions - i.e. wrong answers that look exactly like
confident correct ones and cannot be flagged by any entropy threshold.

| Dataset | both_wrong n | direct_ent <= both_correct median % | decomp_max_ent <= both_correct median % |
|---|---:|---:|---:|
| 2Wiki | 459 | 28.8 | 27.9 |
| HotpotQA | 584 | 18.3 | 42.8 |
| MuSiQue | 714 | 16.5 | 31.5 |
| PopQA | 443 | 9.5 | 47.9 |

## 4. Disagreement quadrants: signal-driven or guessing?

Inside direct_only / decomp_only, compare (a) the current verifier's pick and
(b) a naive rule that picks the path with lower entropy (raw, and z-scored per
dataset because the two entropies live on different scales).

| Dataset | Quadrant | n | verifier picks correct path % | raw lower-entropy path correct % | z-scored lower-entropy path correct % |
|---|---|---:|---:|---:|---:|
| 2Wiki | direct_only | 39 | 28.2 | 97.4 | 69.2 |
| 2Wiki | decomp_only | 361 | 96.4 | 6.9 | 44.9 |
| HotpotQA | direct_only | 111 | 67.6 | 93.7 | 63.1 |
| HotpotQA | decomp_only | 113 | 77.9 | 9.7 | 49.6 |
| MuSiQue | direct_only | 31 | 38.7 | 90.3 | 54.8 |
| MuSiQue | decomp_only | 184 | 89.1 | 24.5 | 55.4 |
| PopQA | direct_only | 28 | 7.1 | 89.3 | 46.4 |
| PopQA | decomp_only | 179 | 93.3 | 5.6 | 53.1 |

## Interpretation

- **Both-correct questions are genuinely confident, not lucky.** Direct-path entropy is
  much lower in both_correct than in both_wrong on every dataset (pooled 0.49 vs 1.40,
  AUROC 0.782, Cohen's d +0.89; PopQA up to AUROC 0.819). Uncertainty rises monotonically
  both_correct -> direct_only -> decomp_only -> both_wrong on 2Wiki/HotpotQA/MuSiQue.
- **The decompose path is always high-entropy and separates poorly.** decomp_max_ent sits
  near ~3 nats in every quadrant (pooled AUROC 0.588; HotpotQA 0.553, PopQA 0.538);
  decomp_mean_ent carries no signal at all (pooled AUROC 0.503). Per-step entropy is a
  usable *retrieval gate* signal (see §4.10) but a poor *question-level correctness* signal.
- **Confidently-wrong is a real, sizable failure mode.** 9.5% (PopQA) to 28.8% (2Wiki) of
  both_wrong questions have direct entropy at or below the both_correct *median* — by the
  decomp signal it is 28-48%. These wrong answers are indistinguishable from confident
  correct ones, so no entropy threshold can flag them. This quantifies slide p47's "low
  entropy does not always mean correctness".
- **In the disagreement quadrants, cross-path entropy comparison is essentially guessing.**
  The raw lower-entropy rule is degenerate (direct entropy is on a smaller scale, so it
  picks direct ~always: 90-97% hit in direct_only but 6-25% in decomp_only). After
  per-dataset z-scoring the hit rate collapses to 45-69% — near coin-flip. The current
  verifier does much better in decomp_only (78-96%) but poorly in direct_only (7-68%),
  i.e. its advantage comes from a decomp prior plus rule-based checks, not from the
  confidence signals themselves.
- **Routing implication.** Confidence tells you *whether* the system is likely right
  (accept/abstain), not *which* path is right. The remaining direct-vs-decompose routing
  gap (slide p46, +3.8~+6.1 oracle EM) cannot be closed by comparing these entropies.
