# Confidence vs Entity-Popularity Bias (PopQA)

Date: 2026-07-03

Join: `popqa_1000_msample.jsonl` (no-RAG self-consistency confidence, `mean_max_ent`) x original PopQA TSV (`s_pop` / `o_pop` = monthly Wikipedia page views). Joined n = 1000 / 1000. Correctness = official rule (normalized EM or token-F1 >= 0.5 vs `gold`); alias-aware uses `possible_answers`.

Overall no-RAG accuracy: official 37.5 / alias-aware 43.1.

## TL;DR

1. **No raw popularity->confidence bias.** Confidence (mean_max_ent) is uncorrelated with subject popularity, overall and within wrong answers (Spearman |rho| < 0.05, n.s., section 1); the zero-entropy wrong group has the same popularity distribution as the zero-entropy correct group (section 4). Confident hallucination is NOT concentrated on famous or obscure entities.

2. **Alias-aware accuracy IS popularity-skewed** (section 2): popular entities' 'errors' are disproportionately metric artifacts (longer, richer answer strings), while rare-entity confident errors are 100% genuine (section 4b). Semantic confident hallucination is therefore mildly skewed toward rare entities even though the official rate looks flat.

3. **The entropy signal does not collapse for rare entities** (section 5): AUROC(ent->wrong) is ~0.72-0.75 in every popularity band. On single-hop PopQA the confidence signal's weakness is the deterministic-wrong plateau (136/1000 zero-entropy wrong), not popularity bias.

## 1. Is confidence driven by popularity?

| pair | Spearman rho | p |
|---|---:|---:|
| mean_max_ent vs log10(s_pop) | -0.022 | 4.80e-01 |
| mean_max_ent vs log10(o_pop) | 0.030 | 3.46e-01 |
| correct vs log10(s_pop) | 0.036 | 2.61e-01 |
| within WRONG only: ent vs log10(s_pop) | -0.027 | 5.03e-01 |
| within CORRECT only: ent vs log10(s_pop) | 0.033 | 5.18e-01 |

The within-wrong correlation is the bias test: if the model is only wrong on rare entities, popularity and correctness are confounded; a negative entropy~popularity correlation *among wrong answers* means the model is more confident about popular subjects even when it is wrong.

## 2. Popularity quartiles (s_pop)

| quartile | s_pop range | n | acc | acc alias | mean ent | zero-ent rate | zero-ent acc |
|---|---|---:|---:|---:|---:|---:|---:|
| Q1 (rarest) | 21-97 | 248 | 30.2 | 31.5 | 0.56 | 33.9% | 51.2 |
| Q2 | 98-227 | 252 | 41.3 | 42.9 | 0.50 | 36.1% | 70.3 |
| Q3 | 229-873 | 250 | 40.8 | 46.0 | 0.46 | 39.6% | 66.7 |
| Q4 (most popular) | 876-15101521 | 250 | 37.6 | 52.0 | 0.48 | 34.4% | 59.3 |

## 3. Accuracy grid: confidence band x popularity tercile

Confidence bands respect the mass at zero entropy: `zero-ent` (fully deterministic SC votes), then the nonzero entropies split at their median.

| | rare acc (n) | mid acc (n) | popular acc (n) |
|---|---:|---:|---:|
| zero-ent (max conf) | 58.3 (120) | 67.7 (124) | 60.3 (116) |
| low ent (nonzero) | 22.2 (99) | 39.1 (110) | 35.1 (111) |
| high ent | 15.0 (113) | 15.0 (100) | 14.0 (107) |

## 4. Where do high-confidence-wrong answers live?

| group | n | median s_pop | mean log10 s_pop |
|---|---:|---:|---:|
| zero-ent WRONG (confident hallucination) | 136 | 225 | 2.59 |
| zero-ent CORRECT (really knows) | 224 | 238 | 2.52 |
| nonzero-ent WRONG | 489 | 208 | 2.57 |

Mann-Whitney zero-ent wrong vs zero-ent correct (log s_pop): U=14698, p=5.77e-01.
Mann-Whitney zero-ent wrong vs nonzero-ent wrong (log s_pop): U=33177, p=9.68e-01.

### 4b. Zero-ent wrong: genuine hallucination vs alias/metric artifact, by popularity

`alias-recovered` = officially wrong but correct under `possible_answers`. If popular-entity confident errors are mostly alias artifacts while rare-entity ones are genuine, the *semantic* confident-hallucination rate is popularity-skewed even though the official rate is flat.

| popularity band | zero-ent wrong n | alias-recovered | genuine wrong | genuine share |
|---|---:|---:|---:|---:|
| rare | 50 | 0 | 50 | 100% |
| mid | 40 | 5 | 35 | 88% |
| popular | 46 | 13 | 33 | 72% |
| ALL | 136 | 18 | 118 | 87% |

## 5. Does the entropy signal degrade for rare entities?

AUROC of mean_max_ent predicting WRONG, within popularity bands:

| band | n | acc | AUROC(ent->wrong) |
|---|---:|---:|---:|
| rare (bottom tercile) | 332 | 32.8 | 0.747 |
| mid | 334 | 42.5 | 0.755 |
| popular (top tercile) | 334 | 37.1 | 0.721 |
| ALL | 1000 | 37.5 | 0.741 |

## 6. Supplementary: deployed SEEK direct path (pathaware trace)

n = 1000 (direct probe entropy available).

| pair | Spearman rho | p |
|---|---:|---:|
| direct probe ent vs log10(s_pop) | -0.025 | 4.30e-01 |
| within direct-WRONG: ent vs log10(s_pop) | -0.045 | 2.58e-01 |

## 7. Examples: zero-entropy wrong, most popular subjects

| subj | s_pop | prop | gold | majority |
|---|---:|---|---|---|
| Mike Flanagan | 449587 | occupation | screenwriter | filmmaker |
| Amitabh Bachchan | 306074 | occupation | film producer | actor |
| Edgar Allan Poe | 287843 | occupation | journalist | writer editor and literary critic |
| Chris Brown | 151737 | place of birth | Tappahannock | atlantic city new jersey |
| Robert Kraft | 85312 | place of birth | Brookline | boston |
| Chris Jericho | 75216 | occupation | actor | professional wrestler musician author actor |
| Scooter Braun | 66280 | occupation | talent manager | entrepreneur talent manager investor philanthropist and entertainment executive |
| Salvador Allende | 43544 | occupation | physician | president of chile |
| Beth Hart | 25256 | occupation | singer-songwriter | singer |
| Ayumi Hamasaki | 19177 | occupation | actor | singersongwriter |
| Vanessa Angel | 16012 | occupation | actor | actress and former model |
| Jean-Baptiste de Lamarck | 14972 | occupation | biologist | naturalist |
