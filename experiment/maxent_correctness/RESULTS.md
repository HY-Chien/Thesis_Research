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
