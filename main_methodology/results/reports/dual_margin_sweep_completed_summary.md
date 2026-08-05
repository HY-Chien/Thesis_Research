# Dual Margin Sweep Completed Summary

Only configs whose per-run log contains `Verifier EM` are included. Currently running/partial JSONL files are excluded.

## Macro over completed datasets

| signal | tau | datasets | EM | F1 | retrieval rate |
|---|---:|---|---:|---:|---:|
| targ_unc | 0.100 | 2wiki,hotpotqa,musique,popqa | 39.4 | 46.4 | 97.5 |
| targ_unc | 0.150 | 2wiki,hotpotqa | 39.7 | 48.2 | 94.1 |
| targ_unc | 0.200 | 2wiki,hotpotqa,musique,popqa | 38.9 | 45.9 | 90.4 |
| targ_unc | 0.250 | 2wiki,hotpotqa | 38.1 | 46.7 | 82.7 |
| targ_unc | 0.300 | 2wiki,hotpotqa,musique,popqa | 35.9 | 43.0 | 77.8 |
| targ_unc | 0.350 | 2wiki,hotpotqa | 34.5 | 42.9 | 67.9 |
| targ_unc | 0.400 | 2wiki,hotpotqa | 32.8 | 40.8 | 59.5 |
| targ_unc | 0.450 | 2wiki,hotpotqa,musique,popqa | 31.0 | 37.4 | 52.9 |
| targ_unc | 0.500 | 2wiki,hotpotqa | 29.3 | 36.1 | 41.7 |
| targ_unc | 0.600 | 2wiki,hotpotqa | 26.3 | 32.4 | 26.0 |
| targ_unc | 0.700 | 2wiki,hotpotqa,musique,popqa | 23.4 | 28.4 | 17.9 |
| mean_margin | 0.500 | 2wiki,hotpotqa,musique,popqa | 27.3 | 32.4 | 33.6 |
| mean_margin | 0.600 | 2wiki,hotpotqa | 31.2 | 37.9 | 47.4 |
| mean_margin | 0.650 | 2wiki,hotpotqa | 32.9 | 40.0 | 54.8 |
| mean_margin | 0.700 | 2wiki,hotpotqa,musique,popqa | 34.4 | 41.0 | 64.1 |
| mean_margin | 0.750 | 2wiki,hotpotqa | 36.1 | 44.1 | 68.6 |
| mean_margin | 0.800 | 2wiki,hotpotqa | 37.9 | 46.5 | 75.4 |
| mean_margin | 0.850 | 2wiki,hotpotqa,musique,popqa | 37.8 | 45.1 | 83.3 |
| mean_margin | 0.900 | 2wiki,hotpotqa | 40.7 | 48.9 | 87.1 |
| mean_margin | 0.930 | 2wiki,hotpotqa,musique,popqa | 39.0 | 46.0 | 91.4 |
| mean_margin | 0.950 | 2wiki,hotpotqa | 41.6 | 49.5 | 92.1 |
| mean_margin | 0.970 | 2wiki,hotpotqa,musique,popqa | 39.9 | 46.7 | 94.9 |
| mean_margin | 0.990 | 2wiki,hotpotqa,musique,popqa | 40.4 | 46.9 | 97.1 |
| max_margin | 0.500 | 2wiki,hotpotqa,musique,popqa | 23.5 | 28.1 | 16.8 |
| max_margin | 0.600 | 2wiki,hotpotqa | 25.2 | 30.3 | 16.1 |
| max_margin | 0.650 | 2wiki,hotpotqa | 25.4 | 30.5 | 17.3 |
| max_margin | 0.700 | 2wiki,hotpotqa | 25.6 | 30.8 | 18.6 |
| max_margin | 0.750 | 2wiki,hotpotqa,musique,popqa | 24.7 | 29.6 | 22.7 |
| max_margin | 0.800 | 2wiki,hotpotqa | 26.1 | 31.5 | 21.2 |
| max_margin | 0.850 | 2wiki,hotpotqa | 26.4 | 32.0 | 22.9 |
| max_margin | 0.900 | 2wiki,hotpotqa,musique,popqa | 25.6 | 30.7 | 27.9 |
| max_margin | 0.930 | 2wiki,hotpotqa | 26.9 | 32.8 | 26.7 |
| max_margin | 0.950 | 2wiki,hotpotqa | 27.4 | 33.3 | 28.4 |
| max_margin | 0.970 | 2wiki,hotpotqa,musique,popqa | 26.9 | 32.0 | 33.8 |
| max_margin | 0.980 | 2wiki,hotpotqa | 28.0 | 33.9 | 32.7 |
| max_margin | 0.985 | 2wiki,hotpotqa | 28.2 | 34.2 | 34.1 |
| max_margin | 0.990 | 2wiki,hotpotqa,musique,popqa | 28.0 | 33.2 | 39.4 |
| max_margin | 0.995 | 2wiki,hotpotqa | 29.0 | 35.5 | 39.7 |
| max_margin | 0.998 | 2wiki,hotpotqa,musique,popqa | 30.1 | 35.9 | 49.7 |
| maqa_margin | 0.500 | 2wiki,hotpotqa,musique,popqa | 36.6 | 43.5 | 73.4 |
| maqa_margin | 0.600 | 2wiki,hotpotqa | 38.8 | 46.9 | 75.3 |
| maqa_margin | 0.750 | 2wiki,hotpotqa,musique,popqa | 38.6 | 45.3 | 84.4 |
| maqa_margin | 0.900 | 2wiki,hotpotqa,musique,popqa | 39.7 | 46.3 | 91.1 |
| maqa_margin | 0.950 | 2wiki,hotpotqa,musique,popqa | 39.7 | 46.4 | 93.8 |
| maqa_margin | 0.970 | 2wiki,hotpotqa,musique,popqa | 40.2 | 46.9 | 95.2 |
| maqa_margin | 0.985 | 2wiki,hotpotqa,musique,popqa | 40.2 | 46.8 | 96.6 |
| maqa_margin | 0.998 | 2wiki,hotpotqa,musique,popqa | 40.3 | 46.9 | 99.1 |

## Best per dataset and signal

| dataset | signal | tau | EM | F1 | retrieval rate | n |
|---|---|---:|---:|---:|---:|---:|
| hotpotqa | targ_unc | 0.150 | 33.2 | 42.6 | 92.6 | 1000 |
| hotpotqa | mean_margin | 0.970 | 33.6 | 43.0 | 93.0 | 1000 |
| hotpotqa | max_margin | 0.998 | 24.2 | 31.9 | 49.0 | 1000 |
| hotpotqa | maqa_margin | 0.970 | 33.5 | 43.0 | 93.4 | 1000 |
| 2wiki | targ_unc | 0.100 | 46.2 | 53.9 | 97.9 | 1000 |
| 2wiki | mean_margin | 0.930 | 50.1 | 56.6 | 92.4 | 1000 |
| 2wiki | max_margin | 0.998 | 35.9 | 41.3 | 41.1 | 1000 |
| 2wiki | maqa_margin | 0.998 | 50.1 | 56.7 | 99.0 | 1000 |
| musique | targ_unc | 0.100 | 26.3 | 34.1 | 95.8 | 1000 |
| musique | mean_margin | 0.990 | 25.9 | 33.5 | 95.1 | 1000 |
| musique | max_margin | 0.998 | 16.5 | 24.6 | 46.8 | 1000 |
| musique | maqa_margin | 0.998 | 26.4 | 34.1 | 98.7 | 1000 |
| popqa | targ_unc | 0.100 | 52.1 | 55.1 | 99.8 | 1000 |
| popqa | mean_margin | 0.990 | 52.3 | 54.9 | 99.8 | 1000 |
| popqa | max_margin | 0.998 | 43.9 | 45.9 | 61.9 | 1000 |
| popqa | maqa_margin | 0.985 | 52.7 | 55.1 | 99.8 | 1000 |

## Excluded partial files

- `hotpotqa_dual_maqa_margin_tau0p65.jsonl`
- `2wiki_dual_maqa_margin_tau0p65.jsonl`
