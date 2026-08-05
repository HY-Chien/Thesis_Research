# Final tl_sum decomposition-path failure analysis

Configuration: prompt-fix A; step retrieval `tl_sum >= 0.008461`; direct verifier `tl_sum` at `tau_d=0.0005`; post-decomposition verifier `tl_sum` at `tau_p=0.0158`.

A step-level synthesis opportunity uses the pre-existing taxonomy rule: at least one final step answer has exact match or token F1 >= 0.5 against any gold answer or alias. Categories and flags below are computed from the final traces.

## hotpotqa (n=1000)

| candidate correctness group | count | share |
|---|---:|---:|
| both_correct | 225 | 22.5% |
| direct_only | 78 | 7.8% |
| decomp_only | 132 | 13.2% |
| both_wrong | 565 | 56.5% |

Final all-tl_sum router: direct-only capture 39 (50.0%); decomp-only capture 111 (84.1%); final wrong 625/1000 (62.5%).

| source among final wrong answers | count | share |
|---|---:|---:|
| both candidates wrong | 565 | 90.4% |
| router misses a direct-only win | 39 | 6.2% |
| router misses a decomp-only win | 21 | 3.4% |

Decomposition candidate errors: n=643.

| exact final-trace category | count | share of decomp errors |
|---|---:|---:|
| synthesis opportunity: a step answer matches gold/alias | 162 | 25.2% |
| no step answer matches gold/alias | 481 | 74.8% |

| orthogonal final-trace flag | count | share of decomp errors |
|---|---:|---:|
| intermediate_echo | 134 | 20.8% |
| subq1_restates_question | 119 | 18.5% |
| zero_retrieved_steps | 26 | 4.0% |
| mixed_retrieval | 165 | 25.7% |
| all_steps_retrieved | 452 | 70.3% |
| no_decomposition_steps | 0 | 0.0% |

Original-query retrieval coverage (the reusable A arm only): full supporting-title coverage@10 in 147/643 (22.9%) decomposition errors.

## 2wiki (n=1000)

| candidate correctness group | count | share |
|---|---:|---:|
| both_correct | 140 | 14.0% |
| direct_only | 40 | 4.0% |
| decomp_only | 350 | 35.0% |
| both_wrong | 470 | 47.0% |

Final all-tl_sum router: direct-only capture 4 (10.0%); decomp-only capture 343 (98.0%); final wrong 513/1000 (51.3%).

| source among final wrong answers | count | share |
|---|---:|---:|
| both candidates wrong | 470 | 91.6% |
| router misses a direct-only win | 36 | 7.0% |
| router misses a decomp-only win | 7 | 1.4% |

Decomposition candidate errors: n=510.

| exact final-trace category | count | share of decomp errors |
|---|---:|---:|
| synthesis opportunity: a step answer matches gold/alias | 92 | 18.0% |
| no step answer matches gold/alias | 418 | 82.0% |

| orthogonal final-trace flag | count | share of decomp errors |
|---|---:|---:|
| intermediate_echo | 24 | 4.7% |
| subq1_restates_question | 157 | 30.8% |
| zero_retrieved_steps | 10 | 2.0% |
| mixed_retrieval | 49 | 9.6% |
| all_steps_retrieved | 451 | 88.4% |
| no_decomposition_steps | 0 | 0.0% |

Original-query retrieval coverage (the reusable A arm only): full supporting-title coverage@10 in 50/510 (9.8%) decomposition errors.

## musique (n=1000)

| candidate correctness group | count | share |
|---|---:|---:|
| both_correct | 76 | 7.6% |
| direct_only | 26 | 2.6% |
| decomp_only | 191 | 19.1% |
| both_wrong | 707 | 70.7% |

Final all-tl_sum router: direct-only capture 10 (38.5%); decomp-only capture 180 (94.2%); final wrong 734/1000 (73.4%).

| source among final wrong answers | count | share |
|---|---:|---:|
| both candidates wrong | 707 | 96.3% |
| router misses a direct-only win | 16 | 2.2% |
| router misses a decomp-only win | 11 | 1.5% |

Decomposition candidate errors: n=733.

| exact final-trace category | count | share of decomp errors |
|---|---:|---:|
| synthesis opportunity: a step answer matches gold/alias | 139 | 19.0% |
| no step answer matches gold/alias | 594 | 81.0% |

| orthogonal final-trace flag | count | share of decomp errors |
|---|---:|---:|
| intermediate_echo | 68 | 9.3% |
| subq1_restates_question | 235 | 32.1% |
| zero_retrieved_steps | 10 | 1.4% |
| mixed_retrieval | 120 | 16.4% |
| all_steps_retrieved | 603 | 82.3% |
| no_decomposition_steps | 0 | 0.0% |

Original-query retrieval coverage (the reusable A arm only): full supporting-title coverage@10 in 37/733 (5.0%) decomposition errors.

## Cross-dataset table

| exact final-trace category | HotpotQA | 2Wiki | MuSiQue |
|---|---:|---:|---:|
| Step answer matches gold/alias, synthesis wrong | 162 (25.2%) | 92 (18.0%) | 139 (19.0%) |
| No step answer matches gold/alias | 481 (74.8%) | 418 (82.0%) | 594 (81.0%) |
| Flag: intermediate-answer echo | 134 (20.8%) | 24 (4.7%) | 68 (9.3%) |
| Flag: sub-q1 restates question | 119 (18.5%) | 157 (30.8%) | 235 (32.1%) |
| Gate state: zero steps retrieved | 26 (4.0%) | 10 (2.0%) | 10 (1.4%) |
| Gate state: mixed | 165 (25.7%) | 49 (9.6%) | 120 (16.4%) |
| Gate state: all steps retrieved | 452 (70.3%) | 451 (88.4%) | 603 (82.3%) |
| Original-query A full coverage@10 | 147 (22.9%) | 50 (9.8%) | 37 (5.0%) |

## What cannot be retained from the legacy taxonomy

The legacy labels `retrieval_gap_both`, `answer_despite_cov`, and `constraint_loss` require full-coverage@10 for the decomposed B-union queries. The saved detail files contain only coverage bits, not the query strings, and were generated before the final prompt-fix-A chain. Because later resolved sub-questions substitute earlier final answers, they can differ in the final trace. Those three labels therefore must not be copied into the final tl_sum table. They require a new B-union retrieval run over the final resolved sub-questions.

Original-query A coverage is retained because the query is the unchanged original question and is independent of the decomposition answers and confidence metric.
