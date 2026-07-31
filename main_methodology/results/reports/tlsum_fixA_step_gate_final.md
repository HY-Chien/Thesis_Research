# Final-trace tl_sum step-gate ablation (fixed-chain replay)

Inputs: `tlsum_fixA_{ds}_shard{0,1}.jsonl` only for alternate step-answer vectors. For each threshold, a step uses `post_pred` iff `tl_sum >= tau`; otherwise it uses `tentative`. Each unique answer chain is resynthesized with the same adapter-disabled Llama-3-8B synthesis prompt as the deployed run.

**Fixed-chain definition.** Later `sub_q` strings are the already-resolved strings in the deployed tau=0.008461 trace. If an earlier answer changes under another threshold, later queries are not regenerated or re-resolved. Scores away from the deployed threshold are therefore controlled fixed-chain replays, not live end-to-end reruns.

Downstream routing is the adopted all-`tl_sum` verifier: direct tau=0.0005 and post-decomposition tau=0.0158. No entropy score participates in gating or routing. EM/F1 are alias-aware. Retrieval rate is the percentage of decomposed steps that retrieve.

## Sanity check

- hotpotqa: deployed synthesis mismatches = 0 / 1000
- 2wiki: deployed synthesis mismatches = 0 / 1000
- musique: deployed synthesis mismatches = 0 / 1000
- popqa: deployed synthesis mismatches = 0 / 1000

## Never / deployed / always

Each cell is final-system EM/F1/step-retrieval-rate.

| Dataset | Never | Deployed tau=0.008461 | Always | Gate EM - Never EM |
|---|---:|---:|---:|---:|
| hotpotqa | 29.80/39.38/0.00% | 37.50/48.72/79.71% | 37.90/49.03/100.00% | +7.70 |
| 2wiki | 33.90/39.86/0.00% | 48.70/55.63/92.11% | 49.30/56.15/100.00% | +14.80 |
| musique | 15.40/24.17/0.00% | 26.60/34.74/87.80% | 26.40/34.63/100.00% | +11.20 |
| popqa | 24.80/25.93/0.00% | 53.70/55.53/95.60% | 53.90/55.80/100.00% | +28.90 |
| Macro | 25.98/32.34/0.00% | 41.62/48.66/88.80% | 41.88/48.90/100.00% | +15.65 |

## Threshold sensitivity

| tau | HotpotQA | 2Wiki | MuSiQue | PopQA | Macro EM/F1 | Macro retr. | Pooled-step retr. |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.002305 | 37.80/49.03/89.23% | 49.30/56.22/95.75% | 26.50/34.63/94.78% | 53.90/55.73/97.60% | 41.88/48.90 | 94.34% | 93.84% |
| 0.008461 | 37.50/48.72/79.71% | 48.70/55.63/92.11% | 26.60/34.74/87.80% | 53.70/55.53/95.60% | 41.62/48.66 | 88.80% | 87.82% |
| 0.02216 | 36.20/47.40/70.40% | 49.00/55.94/87.82% | 26.10/34.30/80.36% | 53.50/55.40/92.40% | 41.20/48.26 | 82.75% | 81.39% |
| 0.04417 | 36.40/47.33/62.38% | 47.50/54.57/84.06% | 24.80/33.45/73.03% | 53.70/55.53/89.00% | 40.60/47.72 | 77.12% | 75.51% |
| 0.07467 | 36.10/47.08/55.07% | 46.10/53.37/79.98% | 23.90/32.87/66.34% | 53.00/54.71/85.20% | 39.77/47.01 | 71.65% | 69.84% |
| 0.1165 | 36.30/47.22/49.14% | 45.80/52.83/75.65% | 23.70/32.69/59.74% | 51.90/53.74/80.50% | 39.42/46.62 | 66.26% | 64.40% |

## Machine-readable details

The companion JSON stores final-system and decomposition-candidate EM/F1, retrieved/total step counts, and direct-selection counts for every dataset/policy.

