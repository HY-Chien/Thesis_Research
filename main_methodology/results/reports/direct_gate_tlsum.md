# Direct-only tl_sum gate recalculation

- Gate: retrieve iff `tl_sum >= tau`.
- Fixed threshold: `tau=0.008461` (the deployed step-gate threshold).
- Best-dev threshold candidates: every unique observed `tl_sum`, plus always/never endpoints.
- Best-dev tie-break: maximum EM, maximum F1, minimum retrieval rate, then smallest threshold.
- Always-retrieve direct and decompose+gate candidates: final `tl_sum` + prompt-fix-A traces (`replay_verifier_tlsum_fixA.build`).

| Dataset | Direct never EM/F1 | Fixed gate EM/F1 (retr.) | Best-dev EM/F1 @ tau (retr.) | Direct always EM/F1 | Decompose+gate EM/F1 | Delta decomp - direct best | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| hotpotqa | 18.20/26.90 | 30.30/41.18 (81.2%) | 31.00/41.68 @ 0.0043897149 (84.3%) | 30.30/41.51 | 35.70/46.16 | +4.70 | 0.830 |
| 2wiki | 12.60/31.90 | 17.80/28.66 (84.7%) | 19.00/28.14 @ 0.00091558754 (92.8%) | 18.00/27.61 | 49.00/55.84 | +30.00 | 0.827 |
| musique | 5.90/12.44 | 10.10/17.41 (96.3%) | 10.40/17.61 @ 0.130827913 (78.3%) | 10.20/17.58 | 26.70/35.08 | +16.30 | 0.741 |
| popqa | 22.40/24.13 | 38.60/49.43 (95.9%) | 40.80/50.23 @ 0.062935277 (87.0%) | 37.80/49.13 | 53.80/55.68 | +13.00 | 0.775 |

## Source audit

- hotpotqa: no-context regenerated-vs-legacy prediction mismatches = 0/1000; final-vs-legacy always-RAG prediction mismatches = 0/1000 (`results/hotpotqa_1000_wiki_single_k8.jsonl`).
- 2wiki: no-context regenerated-vs-legacy prediction mismatches = 0/1000; final-vs-legacy always-RAG prediction mismatches = 0/1000 (`results/2wiki_1000_single_k8.jsonl`).
- musique: no-context regenerated-vs-legacy prediction mismatches = 0/1000; final-vs-legacy always-RAG prediction mismatches = 0/1000 (`results/musique_1000_single_k8.jsonl`).
- popqa: no-context regenerated-vs-legacy prediction mismatches = 0/1000; final-vs-legacy always-RAG prediction mismatches = 312/1000 (`results/popqa_1000_wiki_single_top3.jsonl`).

The legacy PopQA always-RAG file is a top-3 run. The final trace uses the common top-8 configuration, so the final direct candidate is used for the reported PopQA direct-gate values.
