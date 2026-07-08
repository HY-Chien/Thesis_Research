# Advisor Follow-up Slides: Decomposition, Entropy, and Confident Errors

Date: 2026-07-06
Purpose: report back to advisor on why SEEK uses decomposition and entropy, what entropy can/cannot do, and why alternative signals do not fully solve the failure cases.

---

## Slide 1. Today 的主軸

**老師可能會追問的核心問題**

1. 如果 entropy 不能保證整題正確,為什麼還用 entropy?
2. 如果 direct entropy 不可靠,拆解後會不會比較可靠?
3. 如果 entropy 失敗,有沒有試過 margin / TARG / MAQA 類信號?
4. bridge entity / distractor entity / answer type mismatch 到底是什麼錯?

**一句話回答**

SEEK 不是把 entropy 當成 final correctness verifier。SEEK 的主張是:

> decomposition improves the evidence path; entropy is only a local, conservative retrieval gate.

---

## Slide 2. 先釐清:我們不是這樣主張

**不要說**

- "低 entropy 就代表答案正確。"
- "拆解後 entropy 就能準確判斷整題對錯。"
- "entropy gate 主要提升 accuracy。"
- "只要調一個 threshold 就能解 routing。"

**應該說**

- entropy 對 step-level retrieval need 有用,但不是 correctness guarantee。
- decomposition 的價值是讓 retrieval target 更精準,不是讓 uncertainty 完美 calibrated。
- gate 的定位是 efficiency knob:接近 always-RAG accuracy,但省一部分 retrieval。
- final routing 需要 dual-path + verifier + answer-type rules,不能只靠單一 entropy threshold。

---

## Slide 3. Bridge Entity:為什麼「停在中間」會錯?

**直覺**

多跳題通常要先找一個中間實體:

```text
Step 1: source entity >> relation       -> bridge entity
Step 2: bridge entity >> final relation -> final answer
```

如果 Step 1 的 bridge entity 答錯,Step 2 仍然會被執行,但會沿著錯實體查下去。

**MuSiQue gold-decomp 補實驗**

| Setting | hit@5 |
|---|---:|
| 原問題直接查 | 19.0% |
| SEEK 拆解 + 預測 bridge | 38.8% |
| gold decomposition + gold bridge | 55.8% |

條件化分析(gold 用**同子集**算,避免選擇偏誤):

| Condition | n | seek hit@5 | gold hit@5(同子集) |
|---|---:|---:|---:|
| SEEK step-1 bridge 答對 | 424 | 56.4% | 58.5% |
| SEEK step-1 bridge 答錯 | 576 | 25.9% | 53.8% |

**結論**

SEEK 的拆解文字不是主要瓶頸。bridge 答對時 SEEK 的 retrieval 幾乎追上 gold(56.4 ≈ 58.5);bridge 答錯時**題目沒有變難**(gold 仍 53.8)、是 intermediate-answer error 傳染到第二跳 query,seek 崩到 25.9。

---

## Slide 4. 完整例子:拆解如何救回檢索

來源: `slides/examples/ex3_decompose_helps_retrieval.md`

```text
Q: Who is the child of Caroline LeRoy's spouse?
Gold: Fletcher Webster
Gold passages: idx {0, 10}
```

**原問題直接檢索**

```text
top-5 = {4, 5, 10, 15, 16}
結果:漏掉 gold idx 0 = Fletcher Webster
原因:原問題沒有出現 "Daniel Webster",BM25 無從撈到「Daniel Webster 的小孩」那篇。
```

**拆解後檢索**

```text
Step 1: Caroline LeRoy >> spouse
Answer: Daniel Webster       <- bridge entity

Step 2: Daniel Webster >> child
Retrieved gold idx 0

Final answer: Fletcher Webster
EM = 1
```

**講法**

這個例子說明 decomposition 的核心價值:它把原問題中沒有明講的 bridge entity 顯性化,讓第二跳檢索有正確 target。

---

## Slide 5. 完整例子:bridge 答錯時,不是沒走完,而是走歪

來源: `Research/experiment/maxent_correctness/high_low_confidence_error_analysis.md`

```text
Dataset: MuSiQue
Q: Who founded the company that distributed the film UHF?
Gold: Mike Medavoy
Model majority: Orion Pictures
```

合理的多跳流程應該是:

```text
Step 1: UHF >> distributor
Answer: Orion Pictures

Step 2: Orion Pictures >> founder
Answer: Mike Medavoy
```

錯誤發生在:

```text
模型停在 Step 1 的 bridge entity:
Orion Pictures

但題目要的是 founder:
Mike Medavoy
```

**講法**

這不是「完全沒做 reasoning」,而是模型對第一跳答案很有信心,但沒有把它轉成第二跳 final answer。這也是低 entropy 會失敗的典型:它可能只代表第一跳很容易,不代表整題已經完成。

---

## Slide 6. Answer Type Mismatch:prompt 可以救多少?

**錯誤型態**

模型其實拿到 attribute,但輸出型別不符合問題:

- comparison 題應該回兩個候選之一,卻回日期。
- yes/no 題應該回 yes/no,卻回國家/人物。
- 問 entity,卻回 intermediate value。

**完整例子**

來源: `high_low_confidence_error_analysis.md`

| Dataset | Question | Gold | Model majority | 問題 |
|---|---|---|---|---|
| 2Wiki | Which film came out earlier, Emmy Of Stork's Nest or Indira Vizha? | Emmy Of Stork's Nest | 1915 | 回日期,不是片名 |
| 2Wiki | Are both Kaufland and Otrag located in the same country? | yes | Germany | 回國家,不是 yes |
| HotpotQA | Were Scott Derrickson and Ed Wood of the same nationality? | yes | American | 回屬性,不是 yes |

**prompt / rule 是否能救?**

可以救一部分。你們之前的 synthesize prompt fix:

| Dataset | EM gain |
|---|---:|
| 2Wiki | +9.1 |
| MuSiQue | +2.1 |
| HotpotQA | +1.8 |
| PopQA | 0 |

**講法**

Prompt 可以修 synthesis/schema failure,但不能修 bridge entity 已經錯、retrieval 已經 drift 的 case。

---

## Slide 7. Distractor Entity 是什麼?是不是就答錯?

**先回答**

是,final metric 上就是答錯。

**但 taxonomy 為什麼要標 "distractor entity"?**

因為它描述錯誤來源:模型不是亂答,而是被相似、相鄰、或檢索結果附近的錯實體吸走。

**完整例子**

來源: `high_low_confidence_error_analysis.md`

| Dataset | Question | Gold | Model majority | 錯誤直覺 |
|---|---|---|---|---|
| 2Wiki | What is the place of birth of Jo Planckaert's father? | Nevele | Deinze | 鎖到鄰近地名/家族相關實體 |
| MuSiQue | Where is Ulrich Walter's employer headquartered? | Cologne | Schlage | 鎖到錯的 employer/entity |
| MuSiQue | What is the name of the city where Tess Gallagher's spouse was born in? | Clatskanie | Boston | spouse/entity 鏈漂移 |
| PopQA | In what city was Yuriy Vernydub born? | Zhytomyr | Zaporizhya | plausible neighboring city |

**建議用詞**

> locked onto a lexically similar or relation-neighbor distractor entity.

這樣比「鄰近實體」更清楚。

---

## Slide 8. Confidently-Wrong 在分析什麼?

**要回應的老師問題**

> 既然 entropy 可以代表信心,為什麼不直接調 threshold?低 entropy 就相信,高 entropy 就檢索/換路徑?

**分析目的**

量化「低 entropy 但仍答錯」的比例。

**定義**

先用 both_correct 題的 entropy median 當作「正確題裡相對有信心」的基準:

```text
m = median entropy of both_correct questions
```

如果 both_wrong 題也滿足:

```text
entropy <= m
```

就叫 confidently-wrong:

> 錯了,但信心低到像正確題裡有信心的那一半。

**為什麼用 median?**

不是要抓正確題的一半,而是要定義一個穩定的 low-entropy region。若錯題落在這個區域,它就很難被 entropy-only threshold 乾淨分出來。

---

## Slide 9. Confidently-Wrong 結果:direct 與 decompose 都有

| Dataset | direct_ent confidently-wrong | decomp_max_ent confidently-wrong |
|---|---:|---:|
| 2Wiki | 28.8% | 27.9% |
| HotpotQA | 18.3% | 42.8% |
| MuSiQue | 16.5% | 31.5% |
| PopQA | 9.5% | 47.9% |

**解讀**

- direct entropy:可以看整題 direct answer 的信心,但仍有 confident errors。
- decomp entropy:更不適合當整題 correctness signal,因為它是多個 sub-step 的 uncertainty 聚合。

**講法**

> Low entropy is useful but insufficient. It cannot be used as a standalone correctness verifier.

---

## Slide 10. 拆解後能不能更好分辨整題對錯?

**答案:不能這樣宣稱。**

Question-level correctness separation:

| Signal | ALL AUROC, both_wrong vs both_correct |
|---|---:|
| direct_ent | 0.782 |
| decomp_max_ent | 0.588 |
| decomp_mean_ent | 0.503 |

Dataset-level decomp_max_ent:

| Dataset | decomp_max_ent AUROC |
|---|---:|
| 2Wiki | 0.661 |
| HotpotQA | 0.553 |
| MuSiQue | 0.659 |
| PopQA | 0.538 |

**老師若問:那拆解完有比較會判斷對錯嗎?**

答:

> No. Decomposition does not make entropy a reliable final correctness signal. Decomposition improves the evidence path; entropy remains a local retrieval gate.

---

## Slide 11. 那為什麼還要拆解?

因為不拆解時,gate 只能在兩個弱候選間選:

```text
1. parametric direct answer
2. original-question single-query RAG answer
```

多跳題的第二跳 evidence 常常不在原問題字面上,所以 direct-gate 天花板低。

**Direct-gate baseline vs decompose+gate**

| Dataset | Direct-gate best τ | Decompose+gate SEEK | Difference |
|---|---:|---:|---:|
| HotpotQA | 30.7 | 25.0 | -5.7 |
| 2Wiki | 18.6 | 47.2 | +28.6 |
| MuSiQue | 10.8 | 24.7 | +13.9 |
| PopQA | 39.2 | 51.5 | +12.3 |

**講法**

> The gate does not create good candidates. Decomposition creates a better candidate by exposing bridge entities and improving retrieval targets.

HotpotQA 是例外:direct path 本來較強,所以 final SEEK 用 dual-path verifier 把 direct 優勢吃回來。

---

## Slide 12. 2Wiki Entropy 近隨機與 skip 傷害

**AUROC 是怎麼算?**

隨機拿一題 "需要 retrieval" 和一題 "不需要 retrieval",看 uncertainty 是否把需要 retrieval 的題排得更高。

Decompose step-level retrieval gate:

| Dataset | max_ent AUROC(need) |
|---|---:|
| HotpotQA | 0.684 |
| 2Wiki | 0.541 |
| MuSiQue | 0.783 |
| PopQA | 0.749 |

2Wiki 的 0.541 幾乎隨機,表示 low entropy 不是 easy,很多是 confident-but-wrong。

**skip-decompose τ sweep**

| 2Wiki setting | EM |
|---|---:|
| all-decomp | 45.6 |
| τ=0.5, skip 22% | 41.2 |
| τ=0.8, skip 40% | 35.5 |
| τ=1.2, skip 57% | 29.3 |
| all-single | 18.0 |

**講法**

AUROC 說明 entropy 幾乎選不出安全可 skip 的題;τ sweep 實證顯示 2Wiki 只要開始 skip,EM 就下降。

---

## Slide 13. 有沒有試過不是 entropy 的信號?

有。比較過 8 個信號:

| Signal | Meaning |
|---|---|
| max_ent | token max entropy,主方法 |
| mean_ent | 平均 entropy |
| min_lp | 最低 token log-prob |
| maqa_margin | first-token top1-top2 softmax probability margin |
| mean_margin | 平均 margin |
| max_margin | 最大 margin |
| targ_unc | TARG-style exp(-logit_gap / beta), k=20, beta=3 |
| targ_logit_gap | raw top1-top2 logit gap |

**四資料集 macro AUROC(need)**

| Signal | Macro AUROC |
|---|---:|
| max_ent | 0.689 |
| min_lp | 0.652 |
| mean_ent | 0.637 |
| maqa_margin | 0.617 |
| targ_logit_gap | 0.583 |
| mean_margin | 0.580 |
| targ_unc | 0.575 |
| max_margin | 0.559 |

**結論**

在 SEEK 的 sub-question retrieval gate 設定下,max_ent 是整體最穩的 signal。

---

## Slide 14. 2Wiki 不是換 signal 就能解

2Wiki AUROC(need):

| Signal | 2Wiki AUROC |
|---|---:|
| max_ent | 0.541 |
| max_margin | 0.448 |
| targ_logit_gap | 0.437 |
| mean_ent | 0.436 |
| maqa_margin | 0.416 |
| min_lp | 0.401 |
| targ_unc | 0.378 |
| mean_margin | 0.369 |

**講法**

2Wiki 不是 entropy 特別差,而是所有 uncertainty / margin 類信號都接近隨機或反向。這代表問題在 confident-but-wrong multi-hop chains,不是單純換一個 uncertainty score。

---

## Slide 15. TARG 和 MAQA 到底怎麼說?

**TARG**

- 論文: *Retrieval as a Decision: Training-Free Adaptive Gating for Efficient RAG*.
- 任務: whole-question, single-shot adaptive retrieval gate。
- 做法:先對原問題產生 no-context draft,用前 k 個 tokens 的 logits 決定要不要 retrieve。
- 主張:在 instruction-tuned LLM 上,entropy dynamic range 變小;top1-top2 logit margin 是 robust default。

**MAQA**

- 論文: *MAQA: Evaluating Uncertainty Quantification in LLMs Regarding Data Uncertainty*.
- 任務:multi-answer QA uncertainty benchmark,不是 RAG gate 論文。
- 比較:max softmax,entropy,margin,verbalized confidence,consistency。
- 結論:response consistency 最強;在 white-box 方法中,entropy usually matches or outperforms other methods。

**關鍵差異**

| Paper | Setting | 對 entropy 的結論 |
|---|---|---|
| TARG | whole-question draft retrieval gate | margin 比 mean entropy 更穩 |
| MAQA | multi-answer QA uncertainty quantification | entropy 是強 white-box baseline |
| SEEK | decomposed short answer-span per-step gate | max_ent 在自家 8-signal comparison 最穩 |

---

## Slide 16. 文獻與我們結果的接法

**不要說**

> 文獻都說 entropy 差,但我們還是用。

**應該說**

> Prior work is setting-dependent. TARG motivates margin for whole-question prefix gating, while MAQA supports entropy as a strong white-box UQ signal. We therefore compare both TARG-style and MAQA-style margin in our own decomposed sub-question setting.

**我們的實證**

忠實實作後:

```text
targ_unc: k=20, beta=3
maqa_margin: first generated answer token top1-top2 softmax probability margin
```

結果:

```text
max_ent macro AUROC = 0.689
targ_unc macro AUROC = 0.575
maqa_margin macro AUROC = 0.617
```

所以採用 max_ent 是有實驗支撐,不是沒有試過其他 signal。

---

## Slide 17. 反方向問題:高 entropy 不等於錯

**要回應的問題**

如果高 entropy 觸發 retrieval,會不會有些 retrieval 是浪費?

答案:會。

**例子:Tim Hecker**

來源: `high_low_confidence_error_analysis.md`

```text
Q: What is the latest album of Tim Hecker?
Gold: Love Streams
Majority: Love Streams
Sample pattern: Love Streams 4 / 15,其他 sample 是多個幻覺專輯名
```

解讀:

```text
entropy 很高,因為 sample 分布很散;
但 final majority / official prediction 仍然可能答對。
```

**重要 caveat**

這個 15-sample analysis 是診斷用 probing,不是主方法。SEEK 主方法不跑 15 次,仍是 greedy / single generation。

---

## Slide 18. Low-confidence correct 的 official metric caveat

最高 entropy 尾端仍有 official-correct cases,但很多只是 F1-only。

| Dataset | LCC n | exact gold | F1-only |
|---|---:|---:|---:|
| 2Wiki | 12 | 7 | 5 |
| HotpotQA | 44 | 24 | 20 |
| MuSiQue | 10 | 1 | 9 |
| PopQA | 10 | 7 | 3 |
| ALL | 76 | 39 | 37 |

**問題例子**

| Dataset | Gold | Majority | Why problematic |
|---|---|---|---|
| MuSiQue | 323-272 BC | 5043 BC | 只共享 `BC`,語義錯 |
| MuSiQue | Lena Olin | Stig Olin | 同姓但錯人 |
| MuSiQue | Slade School of Art | Camden School of Art | 共享 school/art token,錯機構 |
| HotpotQA | orange juice | Orange Julius | 相關實體,但不是 ingredient |

**論文措辭**

要說 "officially correct under EM/F1",不要直接說 "semantically correct"。

---

## Slide 19. Gate 的真正優勢:cost-accuracy tradeoff

**Never / gate / always 三方**

| Dataset | never-RAG | gate τ=0.3 | always-RAG | gate retrieval rate |
|---|---:|---:|---:|---:|
| HotpotQA | 15.7 | 25.0 | 25.1 | 92.7% |
| 2Wiki | 29.4 | 47.2 | 47.1 | 95.2% |
| MuSiQue | 9.6 | 24.7 | 25.0 | 90.8% |
| PopQA | 24.2 | 51.5 | 51.4 | 99.3% |

**講法**

拆解後檢索是必要的:never -> gate 大幅提升。

gate vs always 幾乎同 accuracy,但省少量 retrieval。這是效率論證,不是 accuracy 贏 always 的論證。

---

## Slide 20. 最後收束:給老師的完整回答

**一段話版本**

> Entropy is useful, but only in a limited role. We do not use it as a whole-question correctness verifier. In fact, our quadrant analysis shows confidently-wrong cases and poor decompose-level correctness separation. The reason SEEK still uses decomposition is not that decomposition calibrates entropy better, but that decomposition improves the evidence path by exposing bridge entities and retrieving more targeted evidence. Entropy is then used conservatively as a per-step retrieval gate. We also evaluated non-entropy signals, including TARG-style margin and MAQA-style margin; max entropy remains the best macro signal in our decomposed sub-question setting, while 2Wiki remains hard for all uncertainty signals. Therefore, the remaining bottleneck is not choosing another threshold, but improving evidence-aware verification, retrieval/reranking, and bridge-entity robustness.

---

## Appendix A. Full Example: Confident Hallucination

來源: `slides/examples/ex1_confident_hallucination.md`

```text
Dataset: 2Wiki
Q: Who is the paternal grandfather of Alexandru Coconul?
Gold: Mihnea Turcitul
Pred: Radu Mihnea
skip_probe_ent = 0.002
EM = 0
```

```text
Dataset: 2Wiki
Q: Who is Blanche Of Portugal (1259-1321)'s paternal grandmother?
Gold: Urraca of Castile
Pred: Beatrice of Castile
skip_probe_ent = 0.006
EM = 0
```

**講法**

模型不是不確定,而是很確定地走到錯家族成員/錯中間實體。這就是 low entropy 不等於 easy case。

---

## Appendix B. Full Example: MuSiQue Decomposition Failure

來源: `results/decomp_failure_taxonomy.md`

```text
Q: Who is the brother of the artist who made Nothin' My Love Can't Fix's album?
Gold: Matthew Lawrence
Direct: Matthew Lawrence
Decomp: Andy Lawrence
```

Decomposition chain:

```text
Nothin' My Love Can't Fix >> performer
=> Joey Lawrence

Joey Lawrence >> sibling
=> Andy Lawrence
```

**解讀**

這是 synthesis / sibling selection failure。模型找到 artist = Joey Lawrence,但 sibling 有多個可能或 evidence/synthesis 選錯,最後輸出 Andy 而不是 Matthew。

---

## Appendix C. Full Example: Constraint Loss / Query Drift

來源: `results/decomp_failure_taxonomy.md`

```text
Q: When did the first mosque open in the place where Tobolar Copra plant is located?
Gold: September 2012
Direct: September 2012
Decomp: 1884
```

Decomposition chain:

```text
Tobolar Copra >> located in the administrative territorial entity
=> Madang Province

When was the first mosque built in Madang Province?
=> 1884
```

**解讀**

原題問的是 Tobolar Copra plant 所在地的 first mosque,但拆解後 geographic constraint 變粗,漂到 Madang Province 的其他 mosque history。這是 constraint loss / retrieval drift。

---

## Appendix D. Full Example: F1-only Caveat

來源: `high_low_confidence_error_analysis.md`

```text
Gold: 323-272 BC
Majority: 5043 BC
Official: F1-only correct
Problem: shares token "BC", but semantically wrong date
```

```text
Gold: Lena Olin
Majority: Stig Olin
Official: F1-only correct
Problem: shares surname, but wrong person
```

**講法**

這些 cases 不能當成「模型其實答對」。論文中要分清楚 officially correct 和 semantically correct。

---

## Appendix E. Source Files

Local analysis files:

- `Research/main_methodology/results/musique_gold_decomp_recall.md`
- `Research/main_methodology/results/quadrant_confidence_analysis.md`
- `Research/main_methodology/results/direct_gate_ablation.md`
- `Research/main_methodology/results/gate_signal_replay_analysis.md`
- `Research/main_methodology/results/gate_signal_paper_fidelity_and_analysis.md`
- `Research/main_methodology/results/decomp_failure_taxonomy.md`
- `Research/experiment/maxent_correctness/high_low_confidence_error_analysis.md`

External papers:

- TARG: arXiv:2511.09803, *Retrieval as a Decision: Training-Free Adaptive Gating for Efficient RAG*
- MAQA: arXiv:2408.06816, *MAQA: Evaluating Uncertainty Quantification in LLMs Regarding Data Uncertainty*
