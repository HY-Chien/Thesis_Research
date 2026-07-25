# tl_sum-gated prompt-fix-A full rerun (n=1000x4)

gate: tl_sum >= 0.008461 (old-sweep skip=0.10 quantile); steps + tentative probes carry the original question (fix A); rescue -> verifier v2 -> typeaware -> alias-aware EM/F1

| ds | n | canonical EM | fixA(max_ent) EM/F1 | **tl_sum+fixA EM/F1** | step retr% (tl_sum) | retr% (max_ent would) | direct-choice% |
|---|---:|---:|---:|---:|---:|---:|---:|
| hotpotqa | 1000 | 35.5 | 38.0/49.5 | **37.7/49.3** | 79.7 | 86.2 | 20.2 |
| 2wiki | 1000 | 50.0 | 49.7/56.5 | **49.0/55.8** | 92.1 | 94.3 | 5.2 |
| musique | 1000 | 24.7 | 26.0/34.2 | **25.9/34.3** | 87.8 | 90.8 | 15.2 |
| popqa | 1000 | 51.9 | 53.3/55.7 | **53.2/55.5** | 95.6 | 99.1 | 3.2 |
| **macro** | | 40.52 | 41.75/48.95 | **41.45/48.72** | | | |
