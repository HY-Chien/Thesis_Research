# TARG Paper Tau Sweep Pilot 100

Date: 2026-06-30

Scope: first 100 examples per dataset, open-domain `wiki` corpus, mode C, path-aware title default, same LoRA adapter as canonical SEEK gate runs.

Paper setting checked from arXiv:2511.09803, *Retrieval as a Decision: Training-Free Adaptive Gating for Efficient RAG*:

- TARG margin uncertainty is `U_mar = mean(exp(-gap / beta))`, where `gap` is the top-1/top-2 raw logit gap over the draft prefix.
- Retrieval is triggered when `U > tau`; higher `U` means more uncertain.
- The paper calibrates `tau` on a dev set either by target retrieval budget (`tau = F_U^{-1}(1-rho)`) or by dev accuracy.
- Default margin link uses `beta=3`; the paper adopts `k=20` as the default prefix length after prefix-length ablations.
- Reported margin tau examples include Qwen `(0.20, 0.15, 0.35)` and Llama `(0.50, 0.70, 0.45)` for NQ/TriviaQA/PopQA-style short-answer tasks; appendix sweeps include `0.30, 0.50, 0.65, 0.80`.

This sweep therefore uses `signal=targ_unc`, `targ_prefix_k=20`, `targ_beta=3.0`, and tau grid `0.15, 0.20, 0.30, 0.45, 0.50, 0.65, 0.80`.

## Macro

| tau | EM | F1(alias) | retr% |
|---:|---:|---:|---:|
| 0.15 | 34.2 | 42.1 | 95.5 |
| 0.20 | 34.5 | 42.8 | 91.5 |
| 0.30 | 32.0 | 40.9 | 80.2 |
| 0.45 | 30.5 | 37.7 | 51.9 |
| 0.50 | 28.5 | 35.2 | 42.3 |
| 0.65 | 26.2 | 30.9 | 21.3 |
| 0.80 | 25.0 | 29.4 | 10.7 |

Baseline comparison from the previous pilot: entropy@0.3 macro = EM 35.2 / F1 43.0 / retr 94.5; maqa_margin@0.3 macro = EM 31.8 / F1 38.6 / retr 60.4; old targ_unc@0.3 with `k=8,beta=1` macro = EM 30.0 / F1 37.0 / retr 40.5.

## Per Dataset

### hotpotqa

| setting | EM | F1(alias) | retr% |
|---|---:|---:|---:|
| entropy@0.3 baseline | 31.0 | 39.8 | 89.7 |
| maqa_margin@0.3 prior | 31.0 | 37.9 | 49.6 |
| old targ_unc@0.3 (`k=8,beta=1`) | 27.0 | 33.3 | 28.2 |
| paper targ_unc tau=0.15 | 33.0 | 39.3 | 91.5 |
| paper targ_unc tau=0.20 | 32.0 | 39.3 | 87.6 |
| paper targ_unc tau=0.30 | 30.0 | 37.6 | 67.5 |
| paper targ_unc tau=0.45 | 27.0 | 33.2 | 40.2 |
| paper targ_unc tau=0.50 | 24.0 | 30.2 | 31.2 |
| paper targ_unc tau=0.65 | 23.0 | 27.2 | 13.2 |
| paper targ_unc tau=0.80 | 22.0 | 26.2 | 8.1 |

### 2wiki

| setting | EM | F1(alias) | retr% |
|---|---:|---:|---:|
| entropy@0.3 baseline | 50.0 | 55.3 | 93.9 |
| maqa_margin@0.3 prior | 42.0 | 46.1 | 62.6 |
| old targ_unc@0.3 (`k=8,beta=1`) | 43.0 | 47.2 | 42.3 |
| paper targ_unc tau=0.15 | 48.0 | 53.7 | 94.7 |
| paper targ_unc tau=0.20 | 49.0 | 54.7 | 90.2 |
| paper targ_unc tau=0.30 | 46.0 | 52.9 | 81.3 |
| paper targ_unc tau=0.45 | 43.0 | 46.9 | 48.8 |
| paper targ_unc tau=0.50 | 40.0 | 43.4 | 41.9 |
| paper targ_unc tau=0.65 | 39.0 | 41.3 | 19.9 |
| paper targ_unc tau=0.80 | 37.0 | 38.6 | 10.2 |

### musique

| setting | EM | F1(alias) | retr% |
|---|---:|---:|---:|
| entropy@0.3 baseline | 20.0 | 32.1 | 94.5 |
| maqa_margin@0.3 prior | 10.0 | 22.7 | 53.2 |
| old targ_unc@0.3 (`k=8,beta=1`) | 8.0 | 21.8 | 31.3 |
| paper targ_unc tau=0.15 | 19.0 | 31.8 | 96.0 |
| paper targ_unc tau=0.20 | 20.0 | 33.3 | 89.1 |
| paper targ_unc tau=0.30 | 15.0 | 29.4 | 75.1 |
| paper targ_unc tau=0.45 | 11.0 | 24.7 | 45.8 |
| paper targ_unc tau=0.50 | 9.0 | 22.1 | 38.3 |
| paper targ_unc tau=0.65 | 5.0 | 14.7 | 15.9 |
| paper targ_unc tau=0.80 | 2.0 | 11.3 | 4.5 |

### popqa

| setting | EM | F1(alias) | retr% |
|---|---:|---:|---:|
| entropy@0.3 baseline | 40.0 | 45.0 | 100.0 |
| maqa_margin@0.3 prior | 44.0 | 47.7 | 76.0 |
| old targ_unc@0.3 (`k=8,beta=1`) | 42.0 | 45.7 | 60.0 |
| paper targ_unc tau=0.15 | 37.0 | 43.8 | 100.0 |
| paper targ_unc tau=0.20 | 37.0 | 43.8 | 99.0 |
| paper targ_unc tau=0.30 | 37.0 | 43.8 | 97.0 |
| paper targ_unc tau=0.45 | 41.0 | 45.8 | 73.0 |
| paper targ_unc tau=0.50 | 41.0 | 45.2 | 58.0 |
| paper targ_unc tau=0.65 | 38.0 | 40.3 | 36.0 |
| paper targ_unc tau=0.80 | 39.0 | 41.3 | 20.0 |

## Takeaway

- The paper default scale matters: `k=20,beta=3` makes `tau=0.3` much more aggressive than our first `k=8,beta=1` pilot. Macro retrieval rises from 40.5% to 80.2%, and macro F1 rises from 37.0 to 40.9.
- Best macro point in this sweep is `tau=0.20` with EM 34.5 / F1 42.8 / retr 91.5. It nearly matches entropy@0.3 F1 43.0, but does not improve it.
- Matching the entropy retrieval regime requires low tau around 0.15-0.20. Paper-style low-budget points (`tau >= 0.65`, roughly 10-21% macro retrieval) hurt SEEK badly, especially MuSiQue and HotpotQA.
- PopQA behaves differently: moderate/high tau reduces over-retrieval, with the best TARG paper setting here around `tau=0.45` or `0.50`, but MAQA margin@0.3 remains stronger on this 100-example pilot.
- Recommendation: do not replace SEEK max entropy with TARG margin for the main multi-hop gate. If included, frame it as a cost-reducing ablation rather than an accuracy-improving replacement.

## Files

- Metrics: `results/gate_targpaper_pilot100_metrics_gpu0.jsonl`, `results/gate_targpaper_pilot100_metrics_gpu1.jsonl`
- Traces: `results/gate_pilot100_targpaper_{dataset}_tau{tau}.jsonl`
