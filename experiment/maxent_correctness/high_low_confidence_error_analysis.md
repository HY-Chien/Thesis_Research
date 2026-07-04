# 高信心錯誤與低信心正確錯誤分析

Date: 2026-06-28

Scope: `Research/main_methodology/results/{2wiki,hotpotqa,musique,popqa}_1000_msample.jsonl`
joined with the corresponding raw dev files for question text and metadata.

## 定義

本分析沿用 `code/analyze_maxent_correctness.py` 的 official correctness:

- prediction: self-consistency `majority`
- gold: `gold`
- correct: normalized exact match, or token-F1 >= 0.5
- confidence: `mean_max_ent` 的反向。`mean_max_ent` 越低，模型越高信心。

Tail selection uses dataset-specific rank deciles:

- 高信心尾端: each dataset 的 lowest entropy 100 題
- 低信心尾端: each dataset 的 highest entropy 100 題

這樣每個資料集都有相同權重，不會因為 HotpotQA / PopQA 的 entropy 分布較低而被 pooled threshold 吃掉。

## 核心數字

| Dataset | 高信心尾端錯誤 | 高信心尾端 Acc | 低信心尾端判對 | 低信心尾端 Acc |
|---|---:|---:|---:|---:|
| 2WikiMultiHopQA | 57 / 100 | 43.0 | 12 / 100 | 12.0 |
| HotpotQA | 27 / 100 | 73.0 | 44 / 100 | 44.0 |
| MuSiQue | 48 / 100 | 52.0 | 10 / 100 | 10.0 |
| PopQA | 37 / 100 | 63.0 | 10 / 100 | 10.0 |
| ALL | 169 / 400 | 57.8 | 76 / 400 | 19.0 |

最重要的現象是 lowest-decile 的 169 個錯誤全部都是:

```text
mean_max_ent = 0
n_uniq = 1
sc_agree = 1.0
```

也就是 15 次 self-consistency sampling 完全產生同一個答案，但答案仍然錯。這不是「模型猶豫」，而是「模型很穩定地停在錯的答案路徑」。

另看 `mean_max_ent = 0` 的整個平台:

| Dataset | zero-entropy n | zero-entropy Acc | zero-entropy wrong |
|---|---:|---:|---:|
| 2WikiMultiHopQA | 301 | 43.9 | 169 |
| HotpotQA | 522 | 75.7 | 127 |
| MuSiQue | 355 | 47.0 | 188 |
| PopQA | 360 | 62.2 | 136 |
| ALL | 1538 | 59.7 | 620 |

因此，MAX_ENTROPY 對整體校準有效，但它不是完美的錯誤偵測器。特別是 multi-hop 題裡，錯誤常常是 deterministic 的。

## 高信心但錯誤的主要種類

### 1. 停在中間 bridge entity，沒有回答 final relation

這是 2WikiMultiHopQA 和 MuSiQue 最典型的錯誤。模型能穩定找出第一跳答案，但沒有繼續做第二跳。

Examples:

| Dataset | Question | Gold | Majority |
|---|---|---|---|
| 2Wiki | Where was the director of film Lettre Ouverte born? | Paris | Alex Joffe |
| 2Wiki | Where was the husband of Caterina Visconti born? | Pavia | Gian Galeazzo Visconti |
| MuSiQue | Who founded the company that distributed the film UHF? | Mike Medavoy | Orion Pictures |
| MuSiQue | What is the competition that the 1965 Federation Cup is part of named after? | International Tennis Federation | Fed Cup |

Interpretation: entropy is low because the first-hop entity is easy and salient. The model is confident about an incomplete reasoning path.

### 2. Answer type mismatch in comparison / yes-no questions

模型知道某些 evidence attribute，但輸出的是日期、國家、人物或兩個候選者，而不是題目要求的選項或 yes/no。

Examples:

| Dataset | Question | Gold | Majority |
|---|---|---|---|
| 2Wiki | Which film came out earlier, Emmy Of Stork's Nest or Indira Vizha? | Emmy Of Stork's Nest | 1915 |
| 2Wiki | Which film has the director born earlier, Jayam Manade or Bulldog Drummond's Revenge? | Bulldog Drummond's Revenge | Louis King |
| 2Wiki | Are both Kaufland and Otrag located in the same country? | yes | Germany |
| HotpotQA | Were Scott Derrickson and Ed Wood of the same nationality? | yes | American |

Interpretation: this is not uncertainty failure. It is output schema failure. The model may retrieve the right evidence but fails to map evidence to the expected answer type.

### 3. Bridge drift or distractor entity lock-in

模型選到鄰近但錯誤的 entity / attribute，且 15 次取樣都穩定重複同一個錯誤。

Examples:

| Dataset | Question | Gold | Majority |
|---|---|---|---|
| 2Wiki | Which country the director of film Daadagiri is from? | Pakistan | India |
| 2Wiki | What is the place of birth of Jo Planckaert's father? | Nevele | Deinze |
| MuSiQue | Where is Ulrich Walter's employer headquartered? | Cologne | Schlage |
| MuSiQue | What is the name of the city where Tess Gallagher's spouse was born in? | Clatskanie | Boston |
| PopQA | In what city was Yuriy Vernydub born? | Zhytomyr | Zaporizhya |

Interpretation: retrieval context or entity prior pushes the model to a plausible neighboring answer. Because the answer phrase is familiar, token entropy stays low.

### 4. Evaluation / alias false negatives

有些 official wrong 並不一定是語義錯，而是 gold/alias/normalization 太窄。

Examples:

| Dataset | Question | Gold | Majority |
|---|---|---|---|
| 2Wiki | What nationality is the director of film World And Time Enough? | United States | American |
| 2Wiki | What nationality is Geodeung Of Geumgwan Gaya's mother? | India | Indian |
| PopQA | What is Ayumi Hamasaki's occupation? | actor | singer-songwriter |
| PopQA | What is Marivi Bilbao's occupation? | actor | actress |

In the bottom-decile HCE set, alias-aware F1 would recover 7 / 169 official errors, all from PopQA. This does not dominate the phenomenon, but it matters for reporting.

### 5. Over-specific or under-specific span

模型輸出語義接近的短名、長名、地名變體，或把完整描述壓縮成 head entity，導致 official metric 不接受。

Examples:

| Dataset | Question | Gold | Majority |
|---|---|---|---|
| HotpotQA | A medieval fortress in Dirleton borders on the south side of what coastal area? | Yellowcraig | Yellowcraigs |
| HotpotQA | What drum machine was introduced in 1989 and allowed 32 notes to be played simultaneously? | The R-8 Human Rhythm Composer | Roland R-8 |
| HotpotQA | Which dog is considered a giant dog breed, the Leonberger or the Basset Hound? | The Leonberger is a giant dog breed. | Leonberger |

Interpretation: these examples are useful when writing caveats. They are officially wrong, but some are closer to answer-span mismatch than true reasoning failure.

## 低信心但判對的主要種類

### 1. 真答案在多個候選中勝出，majority barely wins

這類是「真的低信心但最後答對」。samples 分布很散，代表模型在多個 plausible answer 之間搖擺。

Examples:

| Dataset | Question | Gold | Majority | Sample pattern |
|---|---|---|---|---|
| 2Wiki | Where was the place of burial of the director of The Secret Of Dr. Kildare? | Forest Lawn Memorial Park | Forest Lawn Memorial Park | correct 4 / 15, many cemetery distractors |
| 2Wiki | What is the place of birth of the performer of Thru Your Phone? | Bronx | Bronx | Bronx 3 / 15, NYC / artist-name distractors |
| HotpotQA | What is the latest album of Tim Hecker? | Love Streams | Love Streams | Love Streams 4 / 15, many hallucinated titles |
| PopQA | In what city was Antonio Calcagni born? | Recanati | Recanati | Recanati 5 / 15, many Italian city distractors |

Interpretation: entropy is correctly signaling uncertainty, but the majority answer is still right.

### 2. Boolean / comparison questions with noisy evidence strings

The answer is yes/no or a selected entity, but samples include supporting attributes as alternatives.

這類題目的標準答案通常很短，例如 `yes` / `no`，或是在兩個候選實體中選一個。但模型 15 次 sampling 時，不一定每次都直接輸出最終答案；有些 sample 會輸出它在推理過程中看到的證據字串，例如城市、國家、年份、人物屬性等。因此答案分布看起來很散，MAX_ENTROPY 會判成低信心。不過只要最後 `yes` / `no` 或正確候選仍然拿到最多票，majority answer 還是會被 official metric 判對。

Examples:

| Dataset | Question | Gold | Majority | Alternatives |
|---|---|---|---|---|
| 2Wiki | Are the directors of Back To Back and The White Hen from the same country? | yes | yes | United Kingdom, no, USA |
| HotpotQA | Were Goo Goo Dolls and Echosmith formed in the same city? | no | no | Chino California, yes |
| HotpotQA | Are Skeptical Inquirer and The Progressive Populist both tabloid-style magazines? | no | no | yes |

Interpretation: the model is uncertain over intermediate attributes, but the final boolean label can still win.

### 3. Granularity / surface-form variation counted correct by F1

Some high-entropy correct cases are reasonable answer-span variants rather than exact matches.

這類是「答案意思接近，但寫法或粒度不同」。模型 15 次 sampling 可能有時答短名、有時答完整名稱、有時多加一個修飾詞，所以答案表面形式很不穩定，entropy 會變高。不過 majority answer 和 gold 仍然有大量 token overlap，因此 token-F1 >= 0.5，official metric 會把它算對。這表示低信心不一定代表完全不知道，也可能只是模型在不同答案 span 之間搖擺。

Examples:

| Dataset | Gold | Majority |
|---|---|---|
| 2Wiki | New York | New York City |
| 2Wiki | Kingsbury, London | London |
| HotpotQA | racehorse breeder and owner | breeder and owner of racehorses |
| HotpotQA | officer | military officer |
| PopQA | diplomat | diplomat and politician |

Interpretation: high entropy partly reflects wording and span instability, not necessarily factual uncertainty.

### 4. F1-only false positives

This is the biggest caveat for the low-confidence-correct group. Officially, 76 top-decile cases are correct, but only 39 / 76 are exact gold matches. The remaining 37 / 76 are correct only because token-F1 >= 0.5.

這類要特別小心，因為它們在分數上被算成 correct，但語義上不一定真的答對。原因是 official metric 允許 token-F1 >= 0.5；只要模型答案和 gold 共用一些 token，就可能被算對。尤其在人名、學校名、日期這種答案裡，模型可能只重疊到姓氏、`School of Art`、`BC` 這種片段，但實際指到的是錯的人、錯的機構或錯的年份。因此這組應該說是「officially correct」，不要直接說成「semantically correct」。

| Dataset | LCC n | exact gold | F1-only |
|---|---:|---:|---:|
| 2WikiMultiHopQA | 12 | 7 | 5 |
| HotpotQA | 44 | 24 | 20 |
| MuSiQue | 10 | 1 | 9 |
| PopQA | 10 | 7 | 3 |
| ALL | 76 | 39 | 37 |

MuSiQue is the most problematic: 9 / 10 low-confidence official-correct cases are F1-only. Several are semantically wrong but share one token with gold.

Examples:

| Dataset | Gold | Majority | Why problematic |
|---|---|---|---|
| MuSiQue | 323-272 BC | 5043 BC | shares `BC`, but wrong date |
| MuSiQue | Lena Olin | Stig Olin | shares surname, wrong person |
| MuSiQue | Kim Braden | Briony Braden | shares surname, wrong person |
| MuSiQue | Slade School of Art | Camden School of Art | shares school/art tokens, wrong institution |
| HotpotQA | orange juice | Orange Julius | related entity, not the requested ingredient |

Interpretation: low-confidence-correct should be reported as "officially correct under EM/F1", not automatically as semantically correct.

## Distribution of sample dispersion

High-confidence wrong:

```text
n = 169
n_uniq = 1 for all 169
sc_agree = 1.0 for all 169
mean_max_ent = 0.0 for all 169
```

Low-confidence official-correct:

```text
n = 76
exact gold = 39
F1-only = 37
mean n_uniq is high, especially MuSiQue and PopQA
36 / 76 have sc_agree < 0.5
29 / 76 have 0.5 <= sc_agree < 0.8
10 / 76 have 0.8 <= sc_agree < 1.0
1 / 76 has sc_agree = 1.0
```

So the two groups are qualitatively different:

- High-confidence wrong = deterministic wrong path.
- Low-confidence correct = noisy answer distribution, often with span variants or metric artifacts.

## Implications for SEEK

1. MAX_ENTROPY is a useful global routing signal, but it cannot catch deterministic reasoning mistakes.

2. The largest high-confidence failure mode is not "lack of knowledge"; it is answer-path incompleteness: the model stops at bridge entity, intermediate attribute, or evidence value instead of producing the requested final answer type.

3. This supports using type-aware verifier rules and final-answer guards. Entropy gate alone cannot fix questions where `mean_max_ent = 0`.

4. Low-confidence correct cases explain why aggressive routing can waste computation: high entropy does not mean the answer is necessarily wrong. Sometimes the correct answer is present but competes with many surface variants.

5. For thesis wording, separate "official correctness" from "semantic correctness". This is especially important for MuSiQue, where top-decile low-confidence official-correct cases are mostly F1-only.

## Suggested thesis phrasing

> Error analysis shows that MAX_ENTROPY mainly captures answer dispersion, not all forms of reasoning failure. In the lowest-entropy decile, all 169 official errors have `n_uniq = 1` and `sc_agree = 1.0`, indicating deterministic but wrong answer paths. These failures often arise when the model confidently outputs a bridge entity, an intermediate evidence attribute, or a value with the wrong answer type. Conversely, in the highest-entropy decile, some answers remain officially correct because the correct span wins a noisy self-consistency vote, while many others are only accepted by the loose token-F1 criterion. Therefore, entropy is best used as a routing signal together with answer-type and path-aware verification, rather than as a standalone correctness predictor.
