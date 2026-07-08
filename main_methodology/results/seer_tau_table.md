# SEER (dual + original-question evidence union) — main result across retrieval-gate tau

honest wiki-21M, 1000Q, mode C, max_ent, path-aware, no-evidence, direct_tau=0.4, decomp_tau=0.8, + type-aware verifier replay. EM/F1 = post-replay headline (the SEER (Ours) numbers). retr% = decompose-path retrieval rate (cost). tau=0.3 is the current final/canonical setting; higher tau values are cost-saving alternatives.

## tau = 0.3 (final/canonical)
| dataset | retr% | EM | F1 |
|---------|------:|---:|---:|
| hotpotqa | 92.07 | 35.50 | 46.11 |
| 2wiki | 94.70 | 50.00 | 56.58 |
| musique | 90.55 | 24.70 | 33.06 |
| popqa | 99.30 | 51.90 | 54.71 |
| **Avg** | **94.16** | **40.52** | **47.61** |

## tau = 0.5
| dataset | retr% | EM | F1 |
|---------|------:|---:|---:|
| hotpotqa | 88.44 | 35.40 | 46.10 |
| 2wiki | 92.03 | 50.00 | 56.65 |
| musique | 86.02 | 24.50 | 32.61 |
| popqa | 98.80 | 51.90 | 54.71 |
| **Avg** | **91.32** | **40.45** | **47.52** |

## tau = 0.8
| dataset | retr% | EM | F1 |
|---------|------:|---:|---:|
| hotpotqa | 81.79 | 35.50 | 46.17 |
| 2wiki | 87.82 | 49.00 | 55.78 |
| musique | 79.28 | 23.70 | 31.84 |
| popqa | 95.00 | 51.70 | 54.38 |
| **Avg** | **85.97** | **39.98** | **47.04** |

## tau = 1.0
| dataset | retr% | EM | F1 |
|---------|------:|---:|---:|
| hotpotqa | 77.58 | 35.50 | 46.12 |
| 2wiki | 85.40 | 48.10 | 55.09 |
| musique | 74.31 | 22.60 | 30.91 |
| popqa | 92.10 | 51.80 | 54.41 |
| **Avg** | **82.35** | **39.50** | **46.63** |

## EM / F1 comparison (post-replay, alias-aware)
| dataset | τ=0.3 | τ=0.5 | τ=0.8 | τ=1.0 |
|---------|------|------|------|------|
| hotpotqa | 35.50 / 46.11 | 35.40 / 46.10 | 35.50 / 46.17 | 35.50 / 46.12 |
| 2wiki | 50.00 / 56.58 | 50.00 / 56.65 | 49.00 / 55.78 | 48.10 / 55.09 |
| musique | 24.70 / 33.06 | 24.50 / 32.61 | 23.70 / 31.84 | 22.60 / 30.91 |
| popqa | 51.90 / 54.71 | 51.90 / 54.71 | 51.70 / 54.38 | 51.80 / 54.41 |
| **Avg** | **40.52 / 47.61** | **40.45 / 47.52** | **39.98 / 47.04** | **39.50 / 46.63** |

## Retrieval rate per dataset (decompose-path, cost)
| dataset | τ=0.3 | τ=0.5 | τ=0.8 | τ=1.0 |
|---------|------|------|------|------|
| hotpotqa | 92.07% | 88.44% | 81.79% | 77.58% |
| 2wiki | 94.70% | 92.03% | 87.82% | 85.40% |
| musique | 90.55% | 86.02% | 79.28% | 74.31% |
| popqa | 99.30% | 98.80% | 95.00% | 92.10% |
| **Avg** | **94.16%** | **91.32%** | **85.97%** | **82.35%** |
