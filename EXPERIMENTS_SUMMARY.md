# Self-Certainty (SC) Gated RAG — Experiments Summary

> 最後更新：2026-05-09
> 核心研究問題：用 Self-Certainty (SC) 作為 RAG 觸發信號，能否取代 DeepRAG 的 RL-trained gate？

## 當前狀態 (TL;DR)

**當前最強 production result**（§3.1, §13.6）：vanilla Llama-3-8B + no-CoT first-entity span gate，`neg_max_ent@-0.80` post-hoc gating（4ds × 1000Q）：

| Dataset | gated EM/F1 | trig% | 備註 |
|---|---|---|---|
| HotpotQA | **32.7 / 44.3** | 83% | 同時打敗 always_RAG EM 跟 F1 |
| 2WikiMultiHopQA | 36.2 / 45.4 | 80% | 略低於 always_RAG |
| MuSiQue | 19.0 / 29.1 | 95% | 跟 always_RAG 持平 |
| PopQA | **42.4 / 51.6** | 95% | 同時打敗 always_RAG，**@20% budget 拿到 +6.6 EM** |

**當前 paper framing**（§14.7）：「Confident hallucination 是 **B1 (knowledge error) + B2 (random filling) + B3 (role confusion)** 三種子型的混合物，各自需要不同 detector，且既有 SC 公式只能間接抓到 B2。」

**⚠️ 已知 research gaps**（§4，必須解決才寫得進 paper main claim）：
1. **主結果跟分析在不同 mode**：production gate 用 no-CoT，但 §14 B1/B2/B3 分析用 CoT 8-shot prompt（multi_sample_dump）
2. **「分析」跟「主結果」邏輯上沒連起來**：三條 detector 沒整合進 gate；subtype-aware gate vs single-signal 對照沒做
3. **B3 detector v2 是 hand-tuned**：規則調來 fit 20Q 的 1 正陽 + 1 假陽，sensitivity 沒量化（無 ground-truth annotation）
4. **Step 1 pilot 已部分校正**：`min_margin_full_mean` 20Q AUROC=0.96 → 100Q 跌到 0.58；正式代表是 `n_uniq` AUROC=0.81 + `fe_mean_entropy` 0.78

---

## 1. 專案目錄

```
/home/S113065528/Research/
├── experiment/self_certainty/      ← 階段一/二（exp0–11）on NQ/2wiki, Llama-3.1-8B
│   ├── scripts/                     exp{0..11}_*.py
│   ├── results/                     exp{0..11}_*_results.json + .md 質性報告
│   ├── logs/, figures/
│   └── EXPERIMENTS_SUMMARY.md      ← 階段一二的舊版總結
└── deeprag_compare/                 ← 階段三/五/六：4-dataset benchmark vs DeepRAG
    ├── src/
    │   ├── run_experiment.py         baseline + dynamic + SC-gated post-hoc
    │   ├── run_deeprag_sc.py         DeepRAG-style BFS decomposition + 3 種 gate（支援 model_id 切換 + 4-shot prompt）
    │   ├── retriever_bm25.py         BM25 over wiki ES index
    │   └── dataset_eval.py           EM/F1 (DRAGIN-ported)
    ├── scripts/                     run_dynamic_*.sh, run_deeprag_sc*.sh, compare_all.py
    ├── check/                       階段五/六 分析腳本（exp0/0b, exp1, exp_a, step1/2/3, multi_sample_dump）
    ├── results/                     所有 .jsonl + _summary.json + comparison_all.md
    └── logs/, data/, prompts/
```

---

## 2. 詞彙表 (Glossary)

> 所有 EXPERIMENTS_SUMMARY 中用到的名詞、metric、實驗 protocol、dataset、子型分類解釋。第一次讀本檔的人請先看這節。

### A. 統計與評估指標

| 名詞 | 定義 | 直觀理解 |
|---|---|---|
| **AUROC** (Area Under ROC) | 隨機抽一個正例 + 一個負例，正例得分高的機率。0.5 = random，1.0 = perfect 分類器，<0.5 = 極性反轉 | 0.7 = 「中等鑑別力」、0.8 = 「強」、>0.9 = 「極強」 |
| **Cohen's d** | (mean(pos) − mean(neg)) / pooled_std；effect size 標準度量 | \|d\|<0.2 微弱、0.2-0.5 小、0.5-0.8 中、>0.8 大 |
| **separation_pp** | abs(AUROC − 0.5) × 200，把 AUROC 折成「離 random 多少 pp」（雙向）| 92pp = 跟 random 差 92 個百分點 |
| **specificity** | 真陰性率 = TN / (TN+FP)；不是該類別的樣本被正確排除的比例 | B3 detector specificity 100% = 沒有 false flag |
| **sensitivity** (recall) | 真陽性率 = TP / (TP+FN);該類別的樣本被正確抓出的比例 | B3 detector sensitivity 100% on pilot = 已知 B3 都被抓 |
| **regression to the mean** | 小樣本的極端 AUROC 在大樣本下趨向真實值（通常較低）| `min_margin_full_mean` 從 20Q 0.96 跌到 100Q 0.58 就是這個 |
| **EM** (Exact Match) | 預測經 normalize（lower、去標點、去 articles）後是否完全等於 gold 之一 | 0/1 sample-level，平均後 = `em_rate` |
| **F1** | token-level F1 between normalized pred 跟 gold（DRAGIN-ported metric） | 0~1 連續值 |
| **PPL** (Perplexity) | exp(−mean(token log-prob))；語言模型對自己輸出的「困惑度」| 越低 = 模型對自己越有把握 |

### B. Sampling protocol（multi-sample dump 的設定）

| 名詞 | 定義 |
|---|---|
| **T** (temperature) | sampling temperature，T=0.7 是我們的標準 |
| **top_p** | nucleus sampling cutoff；我們設 1.0 = 不過濾 |
| **N samples** | 每題抽幾次。Pilot=20、scale-up=20、step2=10 per condition |
| **few-shot prompt / 8-shot** | prompt 前面附 8 個 (Q, A) 範例（DRAGIN-style CoT）讓模型 in-context learn 答題格式 |
| **CoT** (Chain of Thought) | 模型先輸出推理過程再給答案 |
| **no-CoT direct** | 模型直接出答案，不寫推理；`Question: X\nAnswer:` 格式 |
| **greedy** | T=0 確定性 decoding；對照 sampling 看穩定性 |

### C. 4 個 stratum（multi_sample_dump 的取樣分層）

從 1000Q HotpotQA NR baseline 結果，按 SC 中位數 × EM 切成 2×2：

| Stratum | 條件 | 含義 |
|---|---|---|
| **HI_correct** | `nr_sc ≥ median AND nr_em == 1` | 信心高 + 答對 — 理想 case |
| **HI_wrong** | `nr_sc ≥ median AND nr_em == 0` | 信心高 + 答錯 — **confident hallucination**，paper 主要研究對象 |
| **LO_correct** | `nr_sc < median AND nr_em == 1` | 信心低 + 答對 — gate false-positive 來源 |
| **LO_wrong** | `nr_sc < median AND nr_em == 0` | 信心低 + 答錯 — gate 該觸發 RAG 的理想情況 |

「median SC」由 1000Q HotpotQA NR baseline 算出 = 12.42。

### D. Confident hallucination 子型分類（已演進）

#### D.1 舊 framing（2026-05-02，已 falsified）— A1 / A2

| 子型 | 假設機制 | 證據狀態 |
|---|---|---|
| ~~**A1: parametric override**~~ | 模型內部知識鎖死，**不讀 input** | ❌ Step 2 perturbation 證偽（5a75383a55 這個 n_uniq=1 的「A1 archetype」在 entity 換成 Einstein 後立刻改答 Einstein → 它**有讀 input**）|
| ~~**A2: in-context filling**~~ | 模型不知道，照 in-context pattern 隨機填空 | 部分對應到新框架的 B2 |

#### D.2 新 framing（2026-05-07）— B1 / B2 / B3

| 子型 | 機制 | 多樣本 signature | Token-level signature | Detector |
|---|---|---|---|---|
| **B1: knowledge error** | parse 對 + 內部知識錯（例如不知道誰先出生）| `n_uniq` 小（1-3）、perturb 後立刻換答 | min_margin 高、fe_entropy 低 — **跟 HI_correct 看不出差別** | 需要 knowledge probing（待做）|
| **B2: random filling / chain break** | schema 不穩、first hop entity 沒鎖定 | `n_uniq` 大（≥8）| min_margin 低、fe_entropy 高 | `n_uniq` AUROC=0.81、`fe_mean_entropy` AUROC=0.78（100Q scale）|
| **B3: role confusion (misparse)** | 把 question 的 constraint entity 當 answer entity | `n_uniq` 鎖在 question 中 entity | （未量化）| spaCy syntactic role × schema filter，specificity 100%, prevalence ~3% |

### E. Token-level features（從 multi_sample_dump 的 token 機率算出）

每題 20 samples，每 sample 100 tokens，每 token 都記錄 top-5 alternatives + 機率。

| Feature | 定義 | 為什麼有用 |
|---|---|---|
| `lp` (log-probability) | log P(chosen_token \| context)；越接近 0 = 越確定 | 直接的 token-level 信心 |
| `top1_lp` | log P(最高機率 token) | 衡量分佈尖銳度 |
| `top2_lp` | log P(第二高 token) | 跟 top1 差距 = margin |
| **margin** | `top1_lp − top2_lp` | 大 = 壓倒性確信；小 = 在兩個選項間猶豫 |
| **entropy** | $-\sum_i P_i \log P_i$ over vocab；nats unit | 低 = 分佈尖（確定）、高 = 分佈平（不確定）|
| **SC (Self-Certainty)** | 跟 uniform 分佈的 KL divergence；formula: $\frac{1}{V}\sum_i \log\frac{1}{P_i} - \log V$，per-token 後 mean | paper [arxiv 2502.18581]；針對數學推理設計 |
| **PPL** | exp(−mean(lp)) | 整段 response 平均水平 |
| **hedge phrase** | "probably / maybe / I think / likely / possibly / might" 等 | 字串級的不確定 cue（結論：模型幾乎不主動講）|

### F. Aggregations: `_full` vs `fe_` vs `_mean` / `_std`

| 前綴 / 後綴 | 含義 |
|---|---|
| `_full` | 對整個 100-token response 做 aggregation |
| `fe_` (first-entity) | **只**對「pred 在 text 中第一次出現的位置對應的 token range」做 aggregation。聚焦答案、避開 filler |
| `_mean` (per qid) | 一個 qid 內 20 samples 的某 feature 的**平均值** |
| `_std` (per qid) | 同上的**標準差**，量「跨 sample 的不穩定度」|
| `_min` / `_max` | response 內 token 級的 min / max |

組合範例：
- `min_margin_full_mean` = 每 sample 算「整段 response 最小 token margin」，再對 20 個 sample 平均
- `fe_mean_entropy_mean` = 每 sample 對 first-entity span 算 mean entropy，再對 20 sample 平均

### G. Sample-aggregate / multi-sample features

| Feature | 定義 |
|---|---|
| `n_uniq` | 一題 20 samples 中 unique（normalized）predictions 的數量。1 = 鎖死、20 = 完全隨機 |
| `em_rate` | 20 samples 中答對的比例 |
| `sc_mean` / `sc_std` | sample-level SC 的平均跟標準差 |
| `ppl_mean` | 20 samples PPL 平均 |

### H. Question schema（Step 3 detector 用）

| Schema | 判定規則 | 例子 |
|---|---|---|
| **comparison** | 含 "or", "older", "first", "more", "either", "neither", ... | "Who was born first, X or Y?" |
| **yes_no** | 句首是 are/is/was/were/do/does/did 且結尾 ? | "Are both X and Y in Michigan?" |
| **fact_finding** | 其他 | "What director worked with X?" |

B3 detector 只在 `fact_finding` schema 觸發。

### I. Entity syntactic role（Step 3 v2 with spaCy）

| Role | spaCy parse 條件 | 含義 |
|---|---|---|
| **CONSTRAINT** | `pobj` of {with, by, starring, featuring, alongside}, OR "by" + VERB head | 合作者 / agent，**不應該是答案** |
| **SUBJECT** | `nsubj` / `nsubjpass` of main verb | 句子在描述這個 entity |
| **DESCRIPTOR** | `pobj` of non-constraint preposition attached to NOUN | "Album of X" 的修飾關係 |
| **OTHER** | 其他 | fallback |

B3 = `pred matches a CONSTRAINT entity AND schema=fact_finding`。

### J. Perturbation conditions（Step 2）

| Condition | 操作 | 目的 |
|---|---|---|
| **original** | 不變 | baseline |
| **mask** | key entity → `[X]` | 模型若 input-dependent，應該失去答案 anchor |
| **replace** | key entity → 同型不同的 entity | 模型若 input-dependent，應該換答案 |
| **distractor** | 加無關前綴句 | 控制變因：證明模型對「無資訊內容」不敏感 |

| 衍生 metric | 定義 |
|---|---|
| **pcr** (pred-change rate) | perturbed mode 中跟 original modal pred **不同**的 sample 比例。1.0 = 每個 sample 都換 |
| **A1_score** | (1 − pcr(mask)) + (1 − pcr(replace))；2.0 = perturbation 完全沒影響。**警告**：高 A1_score 不一定代表 A1，也可能代表「答案本來就不依賴被擾動的 entity」 |

### K. 外部方法、模型、datasets

| 名詞 | 含義 |
|---|---|
| **RAG** | Retrieval-Augmented Generation；先 retrieve 相關 docs 再給模型答 |
| **NR** (no-retrieval) | 不給 docs，模型只用內部 parametric knowledge |
| **gate** / **threshold** | 用某個 score 做門檻決定要不要觸發 RAG |
| **DeepRAG** | [arxiv 2502.01142]；用 SFT + DPO 訓練 RL gate；對比 baseline |
| **DRAGIN** | [arxiv 2403.10081]；mid-CoT 觸發 RAG，用 RIND signal |
| **SC paper** | [arxiv 2502.18581]；Self-Certainty 原始 paper（math reasoning）|
| **DPO** | Direct Preference Optimization；DeepRAG 用來訓 gate 的 RLHF 變體 |
| **SFT** | Supervised Fine-Tuning |
| **IRCoT** | Interleaved Retrieval CoT |
| **PopQA / NQ / SQuAD / TriviaQA** | 單跳 factoid QA datasets |
| **HotpotQA / 2WikiMultiHopQA / MuSiQue** | 多跳 QA |
| **Llama-3-8B-Instruct** | 主要實驗模型 |
| **Qwen-2.5-72B** | 階段四 strong-baseline 候選 |
| **Adaptive-RAG** | [arxiv 2403.14403] |

### L. 實驗階段對應 section

| 階段 | 對應 section | 主要產物 |
|---|---|---|
| 階段一 (exp0-7) | §5 | NQ/2wiki SC validation, Llama-3.1-8B, 500Q |
| 階段二 (exp8-11) | §6 | NQ/2wiki SC-gated RAG with F1≥0.5 hybrid, 500Q |
| 階段三 | §7 | deeprag_compare benchmark, 4ds × 1000Q, vs DeepRAG paper |
| 階段四 | §8 | vanilla Llama-3-8B + Qwen-72B + 4-shot DeepRAG prompt |
| 階段五 | §9 | Span-restricted SC, no-CoT first-ent gate, popqa@20% +6.6 EM |
| 階段六 | §10 | 教授 feedback；A1/A2 → B1/B2/B3 重新分類 + 三條 detector |

---

## 3. 主要結果與 paper claim 演進

### 3.1 當前最強 production gate（§9.6）

vanilla Llama-3-8B-Instruct + no-CoT direct-answer + first-entity span 上 `neg_max_ent` post-hoc gating，threshold = -0.80。

| Dataset | always_NR | always_RAG | **gated** | trig% |
|---|---|---|---|---|
| HotpotQA | 25.0 / 35.4 | 32.2 / 43.7 | **32.7 / 44.3** | 83% |
| 2WikiMultiHopQA | 32.0 / 40.0 | 37.7 / 47.0 | 36.2 / 45.4 | 80% |
| MuSiQue | 11.5 / 20.5 | 19.2 / 29.1 | 19.0 / 29.1 | 95% |
| PopQA | 19.9 / 22.6 | 41.6 / 51.4 | **42.4 / 51.6** | 95% |

PopQA @20%-budget 下 +6.6 EM（vs always-NR）— 最有 paper value 的數字。

### 3.2 跨方法完整對照（4 datasets × 1000Q, Llama-3-8B base）

| Method | hotpotqa EM/F1 | 2wiki EM/F1 | musique EM/F1 |
|---|---|---|---|
| Ours CoT (no-RAG) | 25.2 / 35.6 | 31.2 / 39.3 | 11.5 / 20.3 |
| Ours Single-RAG | 32.4 / 44.0 | 37.9 / 47.0 | 19.2 / 29.0 |
| Ours SC-gated@80 (post-hoc) | 32.1 / 43.2 | 37.3 / 46.6 | 18.2 / 28.1 |
| Ours **no-CoT first-ent gate, min_lp@-0.80** | 30.1 / 41.5 (55% trig) | 35.8 / 44.9 (49%) | 17.9 / 28.2 (76%) |
| Ours **no-CoT first-ent gate, neg_max_ent@-0.80** | **32.7 / 44.3** (83%) | 36.2 / 45.4 (80%) | 19.0 / 29.1 (95%) |
| Ours Dynamic-RAG (best mode) | 20.6 / 32.2 | 20.6 / 30.1 | 10.3 / 19.4 |
| Ours **vanilla Llama-3-8B + SC gate** (200Q, no 4-shot) | 14.5 / – | 8.0 / – | 3.5 / – |
| **Ours DeepRAG-8b + SC gate (th=21)** | **39.5** / 50.4 | **50.2** / 54.9 | **23.0** / 30.4 |
| **Ours DeepRAG-8b + SC gate (th=23)** | **39.7** / 50.4 | **51.1** / 55.9 | 22.8 / 30.2 |
| Ours DeepRAG-8b + learned gate (我們復現) | 39.6 / 51.1 | 50.2 / 54.8 | 23.7 / 30.9 |
| Ours **Qwen-2.5-72B + 4-shot + SC** (200Q) | ⏳ pending | ⏳ pending | ⏳ pending |
| CoT (paper) | 27.2 / 37.8 | 28.2 / 34.9 | 13.7 / 23.0 |
| FLARE (paper) | 23.4 / 32.9 | 30.5 / 37.5 | 17.6 / 23.6 |
| DRAGIN (paper) | 27.6 / 38.1 | 29.1 / 35.7 | 11.8 / 20.0 |
| UAR (paper) | 29.7 / 40.7 | 34.8 / 42.4 | 19.1 / 28.4 |
| **DeepRAG (paper)** | 40.7 / 51.5 | 48.1 / 53.3 | 22.5 / 30.4 |

PopQA：

| Method | EM | F1 |
|---|---|---|
| Ours CoT (no-RAG, 1000Q) | 20.4 | 23.1 |
| Ours Single-RAG (1000Q) | 41.6 | 51.3 |
| Ours SC-gated@80 (1000Q) | 35.0 | 42.6 |
| Ours **no-CoT first-ent gate, min_lp@-0.80** (~76% trig, 1000Q) | 40.1 | 48.7 |
| Ours **no-CoT first-ent gate, neg_max_ent@-0.80** (~95% trig, 1000Q) | **42.4** | **51.6** |
| Ours DeepRAG-8b SC@21 (200Q) | 26.5 | – |
| Ours vanilla Llama-3-8B SC@21 (200Q) | 6.5 | – |
| DeepRAG paper | 42.5 | 47.8 |

### 3.3 Paper claim 演進（時間軸）

| 時間 | Claim 形式 | 觸發 |
|---|---|---|
| 2026-04-25（階段三） | 「SC 作為 training-free gate, 替代 DeepRAG 的 RL-trained gate」 | 階段三 DeepRAG-8b SC@23 跟 learned gate 持平/微贏 |
| 2026-04-27（階段四 pivot） | 「直接用 Qwen-72B prompt-based 拆解 + SC threshold gate，無需 SFT/DPO」 | DeepRAG-8b 結果 inherit 訓練成果，contribution 偏弱 |
| 2026-05-02（階段五） | 「Self-Certainty 公式對 RAG QA 需要 span-restriction 才有效；template-saturated CoT 是它的失效場景」 | exp0/exp0b 證明 sc_orig (full-CoT) AUROC=0.45，no-CoT first-ent span-SC AUROC=0.83 |
| 2026-05-07（階段六） | 「Confident hallucination 是 B1 (knowledge error) + B2 (random filling) + B3 (role confusion) 的混合物，三條互補 detector」 | Step 1/2 結果 falsified A1 hypothesis；Step 3 spaCy detector 命中 5ade69e455 |
| 2026-05-08（scale-up） | 同上，但量化校正：B2 用 `n_uniq` AUROC=0.81 + `fe_mean_entropy` 0.78（不再用 pilot 的 min_margin 0.96）| 100Q scale-up 顯示 min_margin 跌到 0.58 |

**當前 paper structure**：主結果（baseline gate, §3.1）+ analysis（B1/B2/B3 subtyping, §10）+ mechanistic test（perturbation evidence chain, §10.2）。

### 3.4 對 reviewer 的反駁

| 質疑 | 反駁 |
|---|---|
| 「為何不測 NQ？」 | DeepRAG 也沒測 NQ；測 PopQA + 三 multi-hop 已 cover single/multi |
| 「Qwen-72B 比 DeepRAG-8b 大 9 倍，不公平」 | DeepRAG 自己也用 Qwen-72B 當拆解 teacher（paper §4.3）。我們把它從「離線 distill 到 8B」改成「直接 inference」，本質沒用更多算力 |
| 「為何 SC 比 learned 還好？」 | SC@23 retrieves more (avg 2.5 vs 1.7)；trade-off 透過 efficiency curve 呈現 |
| 「為何 Plan A 失敗？」 | multi-hop 中 mid-CoT 插 docs 打斷 reasoning chain（與 FLARE/DRAGIN 弱點一致）|
| 「為何門檻寬到 90% 觸發還算贏？乾脆 always-RAG」| 在 budget-constrained（PopQA @20%）下 +6.6 EM 才是訊號真正的價值 |

---

## 4. ⚠️ 已知 research gaps（必須處理才能寫進 paper main claim）

### Gap 1：主結果跟分析不在同一個 mode

- **§3.1 main result**：no-CoT direct-answer prompt（"Question: X\nAnswer:"），1 sample greedy
- **§10 B1/B2/B3 分析**：CoT 8-shot prompt（DRAGIN-style「reasoning + So the answer is X」），20 samples T=0.7
- **後果**：B1/B2/B3 子型分類**沒在 no-CoT mode 驗證過**。reviewer 可問「你 main result 用 no-CoT，那 B1/B2/B3 在 no-CoT 也成立嗎？」 — 我們**不知道**

**修法**：重跑 multi_sample_dump 的 no-CoT 版本（~80 min GPU），用 production sweep 同款 prompt，對 100Q × 20 samples 重做 step1/3 分析，看 B1/B2/B3 distribution 是否一致。

### Gap 2：「分析」沒連到「主結果」— 最大缺口

- 目前 paper structure 是「main result（neg_max_ent gate）」 + 「analysis（B1/B2/B3 子型存在）」**平行但無 causal link**
- Reviewer 一定會問「**你的 B1/B2/B3 framing 對 main result 有什麼幫助？**」 — 沒答案
- **必殺問題**：subtype-aware gate（HI_correct skip, B3 skip, B1+B2+LO retrieve）能不能在 budget-constrained 下勝過 single-signal `neg_max_ent`？

**修法**：用現有 100Q multi_sample_dump 為每題打 subtype label（HI_correct/B1/B2/B3/LO_*），模擬不同 budget（20%, 30%, 50%）下 subtype-aware vs neg_max_ent 的 EM；無 GPU、1-2 hr coding。

### Gap 3：B3 detector v2 是 hand-tuned

- v2 規則調來 fit 20Q pilot 的**1 個正陽 (5ade69e455) + 1 個假陽 (5abdf12255)** — overfitting 風險顯著
- 100Q scale-up 沒看到新假陽是好兆頭，但**沒做 ground-truth annotation**所以 sensitivity 未知
- 「specificity 100%」trivially 成立，因為規則就是調來 fit 已知 cases

**修法**：人工標 100-200 fact_finding HI_wrong（4-8 hr 人力）；對照 LLM-as-judge baseline（Claude/GPT-4 判每題 entity role），算真實 precision/recall。

### Gap 4：Step 1 pilot 已部分校正

| Feature | 20Q pilot AUROC | 100Q scaled AUROC | 狀態 |
|---|---|---|---|
| `min_margin_full_mean` | 0.96 | **0.58** | ⚠️ pilot 雜訊，**已從 paper claim 移除** |
| `n_uniq` | 0.72 | **0.81** | ✓ 100Q 變最強，**新代表** |
| `fe_mean_entropy_mean` | 0.80 | 0.78 | ✓ 穩定 |
| `fe_min_lp_mean` | 0.72 | 0.76 | ✓ 微強 |
| `sc_mean` (對照) | 0.44 | 0.62 | ~ random（兩端都是雜訊）|

### Gap 5（次要）：Qwen-72B 完整 4ds × 200Q 還沒跑完

§8 階段四 smoke test 結果 pending；如果 Qwen-72B + 4-shot 在 4ds 上跑得動，paper claim 「prompt-based decomposition + SC gate match DeepRAG SFT+DPO」才成立。

---

## 5. 階段一：SC 信號驗證（exp0–7, Llama-3.1-8B, NQ/2wiki, 500Q）

| Exp | 主題 | 結論 |
|---|---|---|
| exp0 | SC vs PPL | SC 與正確性相關，比 PPL 穩定 |
| exp1–3 | SC 穩定性 / CoT 軌跡 | SC 在重複抽樣下穩定，CoT 中逐句變化 |
| exp4 | 分位數 vs 準確率 | 低 SC Q1 準確率最差 → 可作 gating 信號 |
| exp5–7 | 質性 + CoT 統計 | 低 SC 多為需要外部知識的事實型問題 |

---

## 6. 階段二：SC-Gated RAG on NQ/2wiki（exp8–11, Llama-3.1-8B, 500Q, F1≥0.5 hybrid）

| Exp | 設計 | NQ Δ | 2wiki Δ |
|---|---|---|---|
| exp8 Direct QA | 整題 SC 觸發 | 36.0 → 38.8 (+2.8) | 29.8 → 29.2 (−0.6) |
| exp9 CoT-RAG | 整題 SC + CoT | 40.6 → 43.4 (+2.8) | 26.8 → 30.0 (**+3.2**) |
| exp10 Dynamic | 句子層級 SC drop≥3 | 40.6 → 44.8 (**+4.2**) | 26.8 → 27.8 (+1.0) |
| exp11 Decomposed | DeepRAG-style 拆解 + SC | 40.6 → 40.4 (−0.2) | 26.8 → 11.6 (−15.2) ❌ |

主要發現：
- exp10 在 NQ 上最強（+4.2pp），但 multi-hop 效果有限
- exp9 在 2wiki 最穩定
- exp11 失敗 — vanilla Llama 的拆解能力不足

---

## 7. 階段三：deeprag_compare benchmark（4 datasets, Llama-3-8B + DeepRAG-8b, 1000Q）

> Strict EM/F1（DRAGIN-ported, 對齊 DeepRAG paper）。

### 7.1 Baseline + post-hoc SC-gated（run_experiment.py）

| Dataset | no_rag | single_rag | sc@20 | sc@50 | sc@80 |
|---|---|---|---|---|---|
| hotpotqa | 25.2 / 35.6 | 32.4 / 44.0 | 27.5 / 38.0 | 30.6 / 41.4 | **32.1** / 43.2 |
| 2wikimultihopqa | 31.2 / 39.3 | 37.9 / 47.0 | 33.1 / 42.1 | 34.8 / 44.2 | **37.3** / 46.6 |
| musique | 11.5 / 20.3 | 19.2 / 29.0 | 13.5 / 22.8 | 16.8 / 26.1 | **18.2** / 28.1 |
| popqa | 20.4 / 23.1 | 41.6 / 51.3 | 22.6 / 27.5 | 29.5 / 35.6 | **35.0** / 42.6 |

### 7.2 Dynamic SC-RAG（mid-generation 觸發；3 個變體都失效）

| Dataset | continue (basic) | restart | regen+masked (Plan A) |
|---|---|---|---|
| hotpotqa | 20.6 / 32.2 | 19.1 / 30.7 | 20.2 / 31.3 |
| 2wikimultihopqa | 20.6 / 30.1 | 17.2 / 28.0 | 19.5 / 29.2 |
| musique | 10.0 / 19.1 | 10.3 / 19.4 | 10.1 / 19.0 |
| popqa | 7.7 / 17.8 | 8.5 / 18.8 | 7.7 / 17.6 |

→ **全部低於 no_rag**。Plan A（FLARE-style masked query + regenerate_sentence）有正確改變 retrieval 結果（Ed Wood case 證實），但 EM 沒提升 → bottleneck **不在 query 設計或 trigger 處理**。
→ **結論：mid-generation dynamic 在 multi-hop 沒有實證好處**（寫進 paper 當 negative result）。

### 7.3 ⭐ DeepRAG-8b + 三種 gate（1000Q）

| Method | hotpotqa EM/F1 | 2wiki EM/F1 | musique EM/F1 | avg ret |
|---|---|---|---|---|
| **learned gate (paper baseline 復現)** | 39.6 / 51.1 | 50.2 / 54.8 | 23.7 / 30.9 | 1.7 / 1.5 / 2.3 |
| SC gate th=17 | 32.0 / 43.5 | 39.0 / 44.1 | 18.8 / 27.2 | low |
| SC gate th=19 | 36.7 / 48.5 | 47.3 / 52.0 | 22.7 / 30.6 | mid |
| **SC gate th=21** | **39.5** / 50.4 | **50.2** / 54.9 | **23.0** / 30.4 | 1.9 / 2.0 / 2.5 |
| **SC gate th=23** | **39.7** / 50.4 | **51.1** / 55.9 | 22.8 / 30.2 | 2.3 / 2.5 / 2.5 |
| DeepRAG paper | 40.7 / 51.5 | 48.1 / 53.3 | 22.5 / 30.4 | — |

**結論**：DeepRAG-8b 上 SC gate 跟 learned gate 持平/微贏。但這個方法**inherits DeepRAG 的 SFT+DPO 訓練成果**，contribution 偏小 — 觸發階段四 pivot。

### 7.4 DeepRAG-8b + SC gate **on PopQA**（200Q, 2026-04-27）

| Dataset | DeepRAG-8b SC@21 EM | avg_ret |
|---|---|---|
| popqa | **0.265** | 2.385 |

vs 之前 sc_gated@80 (post-hoc, 1000Q) = 0.350 → 在 single-hop popqa 上，**simple post-hoc gate 比 DeepRAG decomposition 還好**。原因可能是 popqa 是單跳，decomposition 反而 over-engineering。

---

## 8. 階段四（進行中）：vanilla Llama-3-8B / Qwen-2.5-72B + DeepRAG framework

> 動機：階段三 SC gate claim 太弱（只是換 gate）。階段四目標：
>   - 證明 **vanilla 8B 跑不動 DeepRAG 框架**（負對照，confirm SFT 必要性）— ✅ 完成
>   - 證明 **Qwen-72B prompt-based 可以替代 DeepRAG 的 SFT 拆解**（升級 paper claim）— ⏳ smoke test pending

### 8.1 ❌ vanilla Llama-3-8B-Instruct + SC gate（4ds × 200Q, **無 4-shot**）

| Dataset | EM | avg_ret | avg_history_len |
|---|---|---|---|
| hotpotqa | 0.145 | 1.00 | 1.00 |
| 2wikimultihopqa | 0.080 | 1.01 | 1.00 |
| musique | 0.035 | 1.01 | 1.00 |
| popqa | 0.065 | 1.00 | 1.00 |

**失敗原因**：
- 每題只生 **1 個 follow-up**（DeepRAG-8b 通常 3-5 個）→ 沒做 multi-hop decomposition
- 只用 1 次 retrieval → 拿不到第二跳資訊

→ 確認 **DeepRAG 的 SFT 訓練是必要的**，**vanilla 8B 沒法靠 prompt 學 multi-hop 拆解格式**。

### 8.2 ⏳ Qwen-2.5-72B + 4-shot DeepRAG prompt + SC gate

正在測：用 DeepRAG 官方 [`ICL-prompt/question_decompose.txt`](https://github.com/gxy-gxy/DeepRAG/blob/main/ICL-prompt/question_decompose.txt) **逐字複製**的 4-shot examples（這也是 DeepRAG 用來叫 72B 生 SFT 訓練資料的同一個 prompt）。

期望結果：
- Qwen-72B 能照 4-shot 例子做出 multi-step decomposition + `<answer short>` 格式
- SC gate 在這個無訓練 setup 下達到接近 DeepRAG-8b 的水準
- **如果成立**：paper claim 升級為「**只用 prompt-based 拆解 + SC gate（不需 SFT、也不需 DPO）能 match DeepRAG 的兩階段訓練**」

Launch 指令見 Appendix B。

---

## 9. 階段五：Span-restricted SC 信號 + 失敗模式分析（2026-05-01 ~ 05-06）

> 動機：階段三的 `sc_gated@80` 後處理在 vanilla Llama-3-8B 上比 `single_rag` 還差或持平。診斷出 `sc_orig` 把整個 CoT 平均，answer 訊號被 ~95% filler tokens 稀釋。

### 9.0 Background：dynamic RAG 信心訊號的演進

| Paper | 訊號 | 特徵 |
|---|---|---|
| FLARE (2305.06983, 2023) | min token prob in lookahead sentence | 只看被選中 token 的 raw probability；長度敏感 |
| DRAGIN (2403.10081, 2024) | entropy × max-attention × semantic（RIND）| 加入重要性 + 語意過濾，但 entropy 仍是 token-level distribution shape |
| **Self-Certainty** (Kang/Zhao/Song 2502.18581, 2025) | $-\frac{1}{nV}\sum_i\sum_j \log(V \cdot p(j\|x,y_{<i}))$ | 整個 vocab 分佈 vs uniform 的 KL，對位置平均；號稱 length-invariant |

> Kang et al. 明確 critique 前面 PPL / log-prob / entropy 系列：(1) 對長 response 有偏（平均稀釋）；(2) 對 no-answer 反而給高分；(3) verbosity 灌水 confidence。Self-Certainty 用「vocab 分佈整體 vs uniform」+ V 正規化來解。

**我們的位置**：Self-Certainty 從 best-of-N math reasoning 搬到 RAG gating，發現它在 full CoT 上仍失效（sc_orig AUROC = 0.45 < 0.5）。原因不是公式壞，而是「位置平均」被 template token 的天然 peaked 分佈淹沒。後面實驗驗證：

- 限縮到 answer-entity span 後，**SC 公式版本**（min_sc）跟 **lp 版本**（min_lp）AUROC 都跳到 0.83+（§9.5 exp0b）
- 1000Q × 4ds gate AUROC +0.10~0.19（§9.2 exp1）
- 27-40% NR 錯誤是 Type-A confident hallucination — 任何後處理 gate 救不到（§9.3 exp_a）

→ paper claim 從「Self-Certainty 取代 DPO」轉向「**Self-Certainty 公式對 RAG QA 需要 span-restriction 才有效；template-saturated CoT 是它的失效場景**」。

**重要術語對照（避免後面 sc 系混淆）**：
- `sc_orig` = Kang 公式套在 full CoT（**不是** DeepRAG 的 SC）
- `min_sc` / `mean_sc` = Kang 公式逐 token 算後在某 span 上取 min / mean
- `min_lp` / `mean_lp` / `first_ent_min_lp` = token log-prob（FLARE / DRAGIN 系列）逐 token 後在某 span 上取統計量
- `min_ent` / `neg_max_ent` = token entropy（DRAGIN 的 $\mathcal{H}_i$ 部分）逐 token 後在某 span 上取統計量

### 9.1 Exp-0：答案 span vs full-CoT confidence metric 診斷

**Setup**：20 questions × 20 temperature samples = 400 samples on hotpotqa

| Metric | AUROC | 說明 |
|---|---|---|
| **first_ent_min_lp** | **0.7565** | answer entity 第一次出現的 token span 取 min lp |
| first_ent_lp (mean) | 0.6967 | 同 span 但取 mean |
| answer_min_lp | 0.6297 | "So the answer is X" 句中 X 的 min lp（291/400 detect rate）|
| sc_orig | 0.4522 | full CoT 的 SC（**比 random 還差**）|
| full_lp | 0.4390 | full CoT mean lp |

關鍵發現：
1. `sc_orig` AUROC = 0.4522 < 0.5 — full-CoT 訊號根本反相關
2. `first_ent_min_lp` 抽 +0.30 — 限縮到 answer entity 那幾個 token 後訊號才出現
3. `min_lp` > `mean_lp` — 多 token 答案中最不確定的那一個是真正瓶頸

→ Script: `deeprag_compare/check/exp0_answer_span_sc.py`
→ Output: `check/exp0_output/exp0_summary.md`

### 9.2 Exp-1：sc_orig gate 換成 first_ent_min_lp（1000Q × 4 datasets）

**Setup**：對現有 `results/{ds}_meta-llama-3-8b-instruct.jsonl` 的 `nr_text` 跑 forward pass scoring（不重新生成），算每題的 first_ent_min_lp，模擬不同 retrieve rate 下的 EM。

| Dataset | NR EM | RAG EM | AUROC sc | AUROC ent | Δ AUROC |
|---|---|---|---|---|---|
| hotpotqa | 0.250 | 0.322 | 0.6049 | 0.7089 | +0.104 |
| 2wikimultihopqa | 0.320 | 0.377 | 0.5703 | 0.6867 | +0.116 |
| musique | 0.115 | 0.192 | 0.6635 | 0.7643 | +0.101 |
| popqa | 0.199 | 0.416 | 0.4737 | **0.6607** | **+0.187** |

**EM at matched retrieve rate**（受限 retrieve budget — 實用情境）

| Dataset | sc@20% | ent@20% | sc@40% | ent@40% |
|---|---|---|---|---|
| hotpotqa | 0.272 | 0.274 | 0.285 | 0.303 |
| 2wikimultihopqa | 0.339 | 0.344 | 0.357 | 0.360 |
| musique | 0.124 | 0.139 | 0.152 | 0.160 |
| popqa | 0.222 | **0.288** | 0.271 | **0.332** |

best-gate EM 幾乎沒差是因為 RAG_EM > NR_EM，oracle threshold = 永遠 retrieve；差距只在 retrieve budget 受限時體現。**popqa @20% retrieve rate 下 +6.6 EM 是最有 paper value 的數字。**

→ Script: `deeprag_compare/check/exp1_gate_rerank.py`
→ Output: `check/exp1_output/exp1_summary.md`、`figures/gate_curve_*.png`

### 9.3 Exp-A：失敗模式分類（六格 confusion matrix）

**Setup**：1000Q × 4ds，confidence 用 first_ent_min_lp 中位數切 HIGH/LOW，乘上 (NR✓/✗) × (RAG saves/fails)

| Cell | 解讀 |
|---|---|
| T1 = HIGH + NR✓ | 正確跳過 retrieval |
| T2 = HIGH + NR✗ + RAG✓ | gate MISS（高信心錯，但 RAG 可救）|
| **T3 = HIGH + NR✗ + RAG✗** | **Type A — confident hallucination，RAG 救不了** |
| T4 = LOW + NR✓ | 過度 retrieve |
| T5 = LOW + NR✗ + RAG✓ | gate WIN（低信心錯，RAG 救）|
| T6 = LOW + NR✗ + RAG✗ | 低信心錯但 RAG 也無效 |

**Type-A share** = T3 / (所有 NR 錯)，**Discrimination** = P(wrong\|LOW) − P(wrong\|HIGH)

| Dataset | Type-A share (ent) | Discrim sc_orig | Discrim first_ent | RAG save HIGH | RAG save LOW |
|---|---|---|---|---|---|
| hotpotqa | 0.332 | +0.123 | **+0.250** | 0.202 | 0.170 |
| 2wikimultihopqa | 0.276 | +0.068 | **+0.297** | 0.294 | 0.177 |
| musique | 0.399 | +0.107 | +0.140 | 0.134 | 0.123 |
| popqa | 0.290 | +0.028 | **+0.182** | 0.345 | 0.380 |

關鍵觀察：
1. **27-40% NR 錯誤是 Type A**（高信心 hallucination）— 後處理 gate 的「天花板」
2. **first_ent Discrim 比 sc_orig 強 1.5×~6.5×**：popqa 上 sc_orig 幾乎不分（+0.028），first_ent 才開始分（+0.182）
3. **musique 最難**：40% Type A + 低 discrim → 解釋為何 musique 後處理改善最少
4. popqa 上 first_ent 的 RAG save (LOW=0.380) > RAG save (HIGH=0.345)，是 gate 真正 work 的徵兆；sc_orig 反向（LOW=0.342 < HIGH=0.388），代表 sc 訊號在 popqa 沒用

→ Script: `deeprag_compare/check/exp_a_failure_modes.py`
→ Output: `check/exp_a_output/exp_a_summary.md`、`figures/failure_modes_*.png`

### 9.4 結論與限制

**做出來的東西**：
- 找到比 sc_orig 強 0.10~0.19 AUROC 的 gate signal `first_ent_min_lp`
- popqa @20% retrieve budget 下 +6.6 EM
- 量化 Type A 佔 NR 錯誤的 27-40%

**限制 — 為什麼純後處理達不到 DeepRAG 水準**：
1. **Type A 是天花板**：musique 40% 錯誤無法被任何 gate 拯救
2. **best-gate ≤ always-RAG**：RAG_EM > NR_EM 時，oracle = 全 retrieve
3. **DeepRAG 優勢不在 gate**：而在 (a) 多步拆解、(b) SFT 教用 docs、(c) DPO 訓練 gate
4. `first_ent_min_lp` 只攻擊了 (c)，且僅僅是更好的訊號，不是更好的決策邊界

→ `first_ent_min_lp` 適合定位成 **diagnostic + analysis tool**，不是主貢獻。配合階段四（Qwen-72B 多步拆解 + SC gate）才有 paper-level claim。

### 9.5 Exp-0b：no-CoT 對照 + span-SC 變體（2026-05-02）

**動機**：（1）CoT 可能有 self-reinforcement 把 first_ent_min_lp 撐高；（2）前面 exp0 沒比 span-restricted **SC 公式版本**，只比了 log-prob 版本。

**Setup**：
- 同 20Q × 20 samples
- 兩種 prompt：原 CoT（重新 forward-pass 算 span-SC/ent）+ no-CoT 直答（現生成 + scoring）
- Metric：`min_lp`、`min_sc`、`mean_sc`、`max_sc`、`min_ent`、`max_ent`、`neg_max_ent` × 三種 span

**結果**：

| Metric | CoT first-ent | no-CoT full | **no-CoT first-ent** |
|---|---|---|---|
| min_lp | 0.6250 | 0.6118 | **0.8665** |
| min_sc | 0.5589 | 0.5657 | **0.8267** |
| mean_sc | 0.5391 | 0.6416 | 0.6779 |
| neg_max_ent | 0.6128 | 0.5853 | **0.8542** |

兩個重要結論：
1. **SC 公式本身沒壞** — `min_sc` 在 no-CoT first-ent 上 AUROC = 0.8267，跟 `min_lp` 0.8665 接近。**之前 sc_orig=0.45 的問題只是 full-CoT 平均稀釋訊號**。
2. **CoT 反而傷害答案訊號** — no-CoT first-ent (0.8665) > CoT first-ent (0.6250)。CoT 的推理過程 in-context 製造 self-reinforcement noise。

注意：CoT first-ent (0.6250) 明顯低於 exp0 的 first_ent_min_lp = 0.7565。差別在 exp0 用原始 generation lp，exp0b 用 forward-pass 重新 score 有 boundary tokenization 飄移。trustworthy 的數字是 no-CoT first-ent 的 0.86。

→ Script: `deeprag_compare/check/exp0b_no_cot_sc.py`
→ Output: `check/exp0b_output/exp0b_summary.md`

### 9.6 Production sweep：no-CoT first-ent gate（4 ds × 1000Q, 2026-05-06）

**動機**：§9.5 在 20Q × 20 samples 上看到 no-CoT first-ent `min_lp` AUROC = 0.8665（vs CoT first-ent 0.6250）。本節在 4×1000Q、單次 greedy decode 下驗證該訊號當 production gate 是否 scale。

**Setup**：
- 4 datasets × 1000Q（hotpot / 2wiki / musique / popqa）
- 對每題：(1) 跑 no-CoT direct-answer（greedy, T=0, MAX_NEW=20）；(2) 抓 first-entity span，算三個 aggregation：`min_lp` / `min_sc` / `neg_max_ent`；(3) Gate：`score < threshold` ⇒ 用既有 RAG 答案，否則用既有 NR (CoT) 答案

**AUROC（no-CoT signal vs `nr_em`，跨資料集 generalizability）**：

| Dataset | min_lp | min_sc | neg_max_ent |
|---|---|---|---|
| hotpotqa | 0.750 | 0.657 | 0.761 |
| 2wikimultihopqa | 0.788 | 0.752 | **0.812** |
| musique | 0.734 | 0.719 | 0.741 |
| popqa | 0.781 | 0.749 | 0.784 |
| **mean** | **0.763** | 0.719 | **0.775** |

**單一 threshold 跨資料集摘要**（mean ΔEM vs always-NR；spread = stdev of trig% across 4 ds）：

| Metric | Threshold | mean trig% | spread (pp) | mean ΔEM | improved |
|---|---|---|---|---|---|
| min_lp | -0.50 | 77% | 8.8 | +0.098 | 4/4 |
| min_sc | 8.40 | 75% | **6.4** | +0.098 | 4/4 |
| neg_max_ent | -0.80 | 88% | 6.7 | **+0.105** | 4/4 |

三個 metric 全部 4/4 dataset 改善。`min_sc` spread 最小（6.4 pp），確認「SC 公式分布最 dataset-invariant」在 1000Q scale 仍成立；`neg_max_ent` ΔEM 最大但 trig% 高達 88%，行為已接近 always-RAG。

**vs §9.2 的 CoT-signal sweep**：

| Setup | mean ΔEM @ ~75% trig | 4/4 improved? |
|---|---|---|
| CoT first-ent min_lp（exp1） | +0.063 | 3/4 |
| **no-CoT first-ent min_lp**（本節） | **+0.098** | **4/4** |
| **no-CoT first-ent neg_max_ent**（本節） | **+0.105** | **4/4** |

換 no-CoT signal 後 mean ΔEM +0.035，且改善範圍從 3/4 → 4/4，1000Q scale 上正式驗證 §9.5 結論「CoT 推理稀釋答案 token 訊號」。

**Per-dataset 絕對 EM / F1**（最佳 cell：`neg_max_ent @ -0.80`，trig 80–95%）見 §3.1。

**比較鬆的 cell**（`min_lp @ -0.80`，trig 49–76%）：用更小 retrieval budget 拿到 67–93% 的 NR→RAG gap。

**Caveat**：本節 gate 仍是 post-hoc（NR/RAG 各跑 1000Q，gate 從現成兩條結果二選一）。Production deployment 要再驗 inference-time 行為：no-CoT direct-answer 多一次 greedy forward pass（~5 min/1000Q on L40S），總成本仍遠低於 always-RAG。

→ Scripts: `check/add_nocot_features.py`、`check/production_sweep_v2.py`
→ Outputs: `check/production_sweep_output/{ds}_per_q_v2.jsonl`、`sweep_nocot.md`

---

## 10. 階段六：教授 feedback + B1/B2/B3 子型分類（2026-05-02 → 05-08）

> 教授核心 critique（2026-05-02）：
>   1. SC 當黑盒子套門檻 → 學術 contribution 弱
>   2. 不要只看「整體 EM 提升幾 %」，要看 **failure cases 為什麼錯**
>   3. SC 是為「數學推理」設計，不見得適合 QA — QA 重在 entity 提取與 knowledge boundary
>   4. 必須做「**擾動實驗**」證明訊號真的捕捉 uncertainty
>
> 重新 framing：研究主軸從「SC 套門檻」轉向「**confident hallucination 子分類 + QA 專屬訊號設計**」。

### 10.1 過去實驗對齊四個方向

> **教授 feedback 跟我們做的東西其實高度重疊** — 下表盤點。

| 教授方向 | 已做 | 對應檔案 | 2026-05-08 狀態 |
|---|---|---|---|
| 1. 打開黑盒子（SC 多次抽樣穩定性）| 階段一 exp1–3：500Q on NQ/2wiki；20Q × 20 samples token-level dump | `experiment/self_certainty/exp{1,2,3}*`、`check/multi_sample_dump.jsonl`（18MB）、`check/analysis_table.md` | ✅ Step 1 完成：sc_mean AUROC=0.62 (100Q) vs n_uniq AUROC=0.81 |
| 2. Confident hallucination 深究 | Type-A 量化（NR 錯誤的 27-40%）+ 個案 token-level 剖析 | `check/confident_hallucination_cases.md`、`check/exp_a_output/` | ✅ Step 2 完成：A1 假設 falsified；改為 B1/B2/B3 三子型 |
| 3. QA 專屬指標（不只套 SC）| span-restricted lp / SC / entropy 都試過；no-CoT first-ent span-SC AUROC=0.83 | `check/exp0_output/`, `exp1_output/`, `exp0b_output/` | ✅ Step 3 v2 完成：spaCy syntactic role × schema filter |
| 4. 擾動實驗 | (Step 2 後完成) | `check/step2_output/` | ✅ 10Q × 4 conditions × 10 samples |

### 10.2 Step 1（2026-05-07, 100Q scaled）— Token-level features 拉開 HI_correct vs HI_wrong

**腳本**：[deeprag_compare/check/step1_feature_analysis.py](deeprag_compare/check/step1_feature_analysis.py)
**輸入**：`multi_sample_dump_100q.jsonl`（25 per stratum × 4 = 100Q × 20 samples）
**輸出**：`check/step1_output_100q/{per_qid_features.csv, group_comparison.{md,csv}}`

每題從 token dump 計算 28 個 features，HI_correct (n=25) vs HI_wrong (n=25) 比較。

| 排名 | Feature | 100Q AUROC | 20Q pilot AUROC | 解讀 |
|---|---|---|---|---|
| **對照** | `sc_mean` | **0.62** | 0.44 | 接近 random，**SC 公式 uninformative**|
| 1 | `n_uniq` | **0.81** | 0.72 | **最強 single feature** — 多樣本答案發散度 |
| 2 | `fe_mean_entropy_mean` | 0.78 | 0.80 | answer span 平均 entropy（穩定）|
| 3 | `fe_min_lp_mean` | 0.76 | 0.72 | answer span 最小 lp |
| 4 | `fe_min_margin_mean` | 0.76 | 0.72 | answer span 最小 margin |
| 5 | `fe_max_entropy_mean` | 0.76 | 0.76 | answer span 最大 entropy |
| ⚠️ | `min_margin_full_mean` | **0.58** | 0.96 | **20Q pilot 雜訊，已從 paper claim 移除** |
| — | `hedge_count_mean` | ~ 0.5 | 0.40 | 模型幾乎不說 "probably/maybe" |

**結論（修正後）**：教授要的「打開黑盒子」直接命中 — **SC AUROC=0.62 vs n_uniq AUROC=0.81**。SC 公式（mean over tokens）把訊號平均掉，多樣本一致性才是最強 single signal。

### 10.3 Step 2（2026-05-07）— Perturbation 實驗 falsify A1 假設

**腳本**：[deeprag_compare/check/step2_perturbation.py](deeprag_compare/check/step2_perturbation.py) + [step2_analyze.py](deeprag_compare/check/step2_analyze.py)
**配置**：5 HI_correct + 5 HI_wrong × 4 conditions × 10 samples = 400 generations，~15 min on L40S
**輸出**：`check/step2_output/{perturbation_dump.jsonl, sensitivity_summary.csv, sensitivity_table.md}`

四種 condition：original / mask（key entity → `[X]`）/ replace（同型 entity 替換）/ distractor（無關前綴）。

#### 10.3.1 ⚠️ A1 假設（parametric override → input-ignorant）— **falsified**

我們以為的 A1 case 是 `5a75383a55`（n_uniq=1，"Who was born first, Ulf Merbold or Mirosław Hermaszewski?"）。實際 perturbation 表現：

| Condition | Modal pred | n_uniq | 解讀 |
|---|---|---|---|
| original | `mirosław hermaszewski` ×10 | 1 | 鎖死 |
| **mask `[X]`** | `ulf merbold` ×7, `unknown` ×3 | 2 | **完全變了** — 模型理解「兩選一少一個就 unknown」 |
| **replace Einstein** | `albert einstein` ×10 | 1 | **完全變了** — 模型知道 Einstein(1879) < Merbold(1941) |
| distractor | `mirosław hermaszewski` ×10 | 1 | 不受無關前綴影響 |

**`n_uniq=1` 不代表 input-ignorant，只代表「對特定 input 的回答穩定」**。Schema parse 完全正確（comparison schema 抓得清楚），失敗只在「Hermaszewski 出生年知識錯」。原本「擾動敏感度可區分 A1」的假設不成立。

#### 10.3.2 實驗暴露三種真實 parse pattern（取代 A1/A2）

| Parse pattern | 例子 | 失敗模式 | 對 perturbation 反應 |
|---|---|---|---|
| **B1: Schema 對 + 知識錯** | 5a75383a55 | parse 對，知識不足 | input 一變立刻換答案 |
| **B2: Random filling / chain break** | 5a7c2f1455, 5abd012055 | schema 連 first hop 都不穩 | 每個 condition 都飄 |
| **B3: Role confusion (misparse)** | 5ade69e455 modal=`vikram bhatt` | 把 question constraint entity 當答案 | mask 後 role confusion **直接消失**（見 §10.4）|

**副發現**：A1_score 排名意外把 HI_correct 排前面（`5ab51dae55`=1.90, `5ae1316555`=1.40），原因不是 input-ignorant 而是「**答案不依賴被擾動的 entity**」（Elko 在 Nevada → 不管配哪個 Michigan 機場答案都是 "no"）。這是 robust correctness 而非 hallucination。

### 10.4 Step 3（2026-05-07/08）— Role-overlap detector for B3

**腳本**：v1 [step3_role_overlap.py](deeprag_compare/check/step3_role_overlap.py)（regex）+ v2 [step3_role_overlap_v2.py](deeprag_compare/check/step3_role_overlap_v2.py)（spaCy）
**輸出**：`check/step3_output{,_100q}/role_overlap_v2_*`

#### 10.4.1 B3 detector v1（regex）+ v2（spaCy）規則

**v1**: 對 question 抽 capitalized entities，計算「pred 子字串等於任一 entity」比例。Schema filter：只在 `fact_finding` 觸發。

**v1 已知 false positive**（5abdf12255）：「Album of Rock singer Robert C Seger... what was the singer's stage name?」 pred="Seger"，模型答對 (em=1.00) 但 v1 標 B3。問題：v1 沒區分 Bob Seger 是 SUBJECT (描述 entity) 不是 CONSTRAINT。

**v2**: spaCy dependency parse 判 entity 角色：
- **CONSTRAINT** = `pobj` of {with, by, starring, featuring, alongside}, OR "by" + VERB head, OR appositive of co-host/co-star
- **SUBJECT** = `nsubj` / `nsubjpass`
- **DESCRIPTOR** = `pobj` of non-constraint prep attached to NOUN（"Album of Seger"）
- **OTHER** = 其他

B3 = `pred matches a CONSTRAINT entity AND schema=fact_finding`。

#### 10.4.2 B3 detector 命中 5ade69e455 + perturbation 通過 mechanistic test

`5ade69e455`：「What director worked **with Vikram Bhatt** ...」答案應該是 director，但模型答 "Vikram Bhatt"。

| Condition | overlap_rate | Modal pred |
|---|---|---|
| original | **1.00** | `vikram bhatt` ×10 |
| **mask `[X]`** | **0.10** | 10 個不同的 actual directors |
| replace Karan Johar | 0.10 | mohit suri ×3, puri jagannadh ×3 |
| distractor | 0.50 | 部分恢復原樣 |

**這就是教授要的 mechanistic evidence**：detector 標 → 把 entity 拿掉 → role confusion 消失。整條因果鏈閉合。

#### 10.4.3 v1 vs v2（20Q pilot）

| Case | v1 | v2 | 解讀 |
|---|---|---|---|
| 5ade69e455 (Vikram Bhatt) | 0.70 | **0.70** | ✅ 維持命中（CONSTRAINT×14）|
| 5abdf12255 (Bob Seger) | 1.00 | **0.00** | ✅ **FIXED**（DESCRIPTOR×20, "of" 接 noun）|
| 5a73822555 / 5a75383a55 / 5a7cc50e55 (comparison) | 0.85-1.00 | 0.00 | schema filter 過濾 |
| 其他 16 題 | 0.00 | 0.00 | 乾淨 |

#### 10.4.4 B3 detector 在 100Q scale 規模

只有 **3/100 (3%) 觸發 v2_b3_rate > 0**，且都是 sporadic（1/20 samples）：

| qid | stratum | 觸發 sample | 解讀 |
|---|---|---|---|
| 5adf58f155 | HI_wrong | 1/20 | Q: "TV series featured an actor who also performed in The Young Ones" — 1 sample 把 actor 當答案 |
| 5ab4bea155 | LO_correct | 1/20 | Q: "...soccer team lose in that is now known as Croke Park?" — 1 sample 把 constraint "Croke Park" 當答案 |
| 5adfd2f255 | LO_wrong | 1/20 | Q: "...developed by the Valve Corporation is part of what 2012 game?" — 1 sample 把 "Valve Corporation" 當答案 |

**驗證後的 Step 3 paper claim（含 caveat）**：
- ✅ Specificity ~100% 在 100Q 上維持（3 個 flag 經人工確認都是真 B3）
- ✅ B3 在 random HotpotQA 中 sporadic prevalence ~3%
- ⚠️ Pilot 的 5ade69e455（v2_b3_rate=0.70 = 14/20）是嚴重 case，random sample 100Q 沒命中類似強度 → **systematic B3 罕見**
- ⚠️ **Sensitivity 沒測** — 規則是 hand-tuned to 1 正陽 + 1 假陽，可能漏掉真 B3 也不知道（見 §4 Gap 3）

### 10.5 三子型 + 各自 detector（取代 A1/A2 framing）

| 子型 | 機制 | 多樣本 signature | Token-level signature | 專屬 detector | RAG 救援預期 |
|---|---|---|---|---|---|
| **B1: schema 對 + 知識錯** | parse 正確，但內部知識錯 | n_uniq 小（1-3）、perturb 後立刻換答案 | min_margin 高、fe_entropy 低 — **跟 HI_correct 看不出差別** | 無單一訊號可分（這也是 finding）；只能靠 multi-sample n_uniq + EM | ✅ RAG 直接補知識應該有效 |
| **B2: random filling / chain break** | schema 不穩、no first hop 鎖定 | n_uniq 大（≥8）| min_margin 低、fe_entropy 高 | `n_uniq` AUROC=0.81、`fe_mean_entropy` AUROC=0.78（100Q 驗證）| ⚠️ 取決於 retrieval 能否補對 |
| **B3: role confusion (misparse)** | constraint entity 誤當 answer | n_uniq 鎖在 question 中 entity | （未量化）| spaCy syntactic role × schema filter（pilot specificity=100%）| ❓ retrieval 救不了，需要 prompt restructure |

---

## 11. 下一步（priority-ordered，2026-05-09）

按優先序，每項列出對 paper claim 的影響跟成本。

### 11.1 ⭐ Subtype-aware gate vs single-signal（Gap 2 — 最關鍵）

**動機**：`neg_max_ent@-0.80` gate 在 80-95% 觸發率達到強表現，但跟 always-RAG 差不多。**訊號的 paper value 在 budget-constrained 才看得出來**（PopQA @20% +6.6 EM 是現有最佳 evidence）。

**假設**：subtype-aware gate（HI_correct skip, B3 skip, B1+B2+LO retrieve）在 30%/50% budget 下勝過 single-signal `neg_max_ent`。

**步驟**：
1. 用現有 100Q multi_sample_dump 為每題打 subtype label
2. 模擬不同 budget（20%, 30%, 50%）下 subtype-aware vs neg_max_ent 的 EM
3. 畫 budget × EM 曲線比較
4. **如果 subtype-aware 勝 → paper claim 從「analysis only」升級到「analysis-driven gate」**

**成本**：無 GPU、1-2 hr coding

### 11.2 CoT/no-CoT 跨模式驗證（Gap 1）

**動機**：B1/B2/B3 分析在 CoT mode (multi_sample_dump)，main result 在 no-CoT。**reviewer 必問跨 mode 一致性**。

**步驟**：
1. 用 production_sweep 同款 no-CoT prompt 重跑 multi_sample_dump（100Q × 20 samples）
2. 重做 step1 / step3 v2 分析
3. 比較 B1/B2/B3 distribution 是否在兩 mode 一致

**成本**：~80 min GPU + 30 min 分析

### 11.3 B3 detector ground-truth annotation（Gap 3）

**動機**：v2 規則 hand-tuned to 1+1 case；100Q scale 上 specificity 100% 但 sensitivity 未量化。

**步驟**：
1. 從 1000Q HotpotQA NR baseline 抽 100-200 fact_finding HI_wrong
2. 人工標每題 modal pred 是否是 role confusion（ground truth）
3. 算 v2 detector 的真實 precision / recall
4. 對照 LLM-as-judge baseline（Claude/GPT-4 判每題 entity role）

**成本**：4-8 hr 人力 + 30-60 min API

### 11.4 B1 knowledge probing

**動機**：B1（5a75383a55 等）的機制目前是猜測 — 是真知識錯還是 prompt-level inference bug？

**步驟**：對 B1 候選 case 直接問模型「When was X born?」「Did A direct B?」確認模型內部知識是否真的錯。

**成本**：~30 min GPU

### 11.5 階段四 Qwen-72B 完整 4ds × 200Q（Gap 5）

**動機**：階段四 paper claim「prompt-based decomposition + SC gate match DeepRAG SFT+DPO」需要 Qwen-72B 跑得動 4-shot decomposition。

**步驟**：見 Appendix B 的 launch 指令；smoke test (hotpotqa 30Q) 確認後 launch 完整 4ds×200Q。

**成本**：~5-7 hr GPU（FP16 across 2 GPU）

### 11.6 中長期（往 paper draft）

- Iterative SC（Path A）：用 ΔSC 當 retrieval 品質 closed-loop signal
- Threshold efficiency curve：x=avg_retrievals, y=EM 跨方法
- PopQA SC sentence-level dynamic
- Method section、Experiments table、Analysis、Negative results 寫作

---

## 12. 環境 / 路徑 / 結果檔命名 / Code 變更

### 12.1 環境 / 路徑

| 用途 | 路徑 |
|---|---|
| Python env | `/dataspace/S113065528/envs/research/bin/python`（也可用 `prag` conda env）|
| Llama-3-8B base | `/workspace/models/Meta-Llama-3-8B-Instruct/` |
| **Qwen-2.5-72B-Instruct** (FP16 cache) | `/workspace/models/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31` |
| DeepRAG-8b ckpt | `/home/S113065528/Adaptive-RAG/hf_cache/models--xinyan233333--DeepRAG-8b/` |
| Elasticsearch | `localhost:9200` (wiki, field=`paragraph_text`, 880K docs) |
| 必要 env vars | `HF_HUB_OFFLINE=1`, `CUDA_VISIBLE_DEVICES=...` |

### 12.2 結果檔命名規則

| 檔名模式 | 內容 |
|---|---|
| `<ds>_meta-llama-3-8b-instruct.jsonl` | baseline (no_rag + single_rag + 16 sc_gated_rate*) |
| `<ds>_meta-llama-3-8b-instruct_dyn_{continue,restart,planA}.jsonl` | dynamic 變體 |
| `<ds>_deeprag-8b_learned.jsonl` | DeepRAG-8b 原始 learned gate |
| `<ds>_deeprag-8b_sc_th{17,19,21,23}.jsonl` | DeepRAG-8b SC gate sweep |
| `<ds>_llama3-8b-instruct_sc_th21.jsonl` | vanilla Llama-3-8B + SC gate (no 4-shot) |
| `<ds>_qwen2.5-72b-fs_sc_th21.jsonl` | Qwen-2.5-72B + 4-shot + SC gate（待產出）|
| `comparison_all.md` | compare_all.py 產出的單一 markdown 表 |
| `multi_sample_dump.jsonl` | 階段六 20Q × 20 samples token-level dump (CoT, T=0.7) |
| `multi_sample_dump_100q.jsonl` | 階段六 100Q scale-up dump (CoT, T=0.7) |

### 12.3 Code 變更紀錄

| File | 變更 |
|---|---|
| `deeprag_compare/src/run_experiment.py` | (階段三) 加 dynamic 變體 + skip_baselines |
| `deeprag_compare/src/run_deeprag_sc.py` | (階段三) gate suffix 在檔名；(階段四) `--use_fewshot` flag + DeepRAG 官方 4-shot prompt 整合 |
| `deeprag_compare/scripts/run_deeprag_sc.sh` | 加 `MODEL_ID/MODEL_TAG/HF_CACHE/DEVICE/PROMPT_STYLE/USE_FEWSHOT` env vars |
| `deeprag_compare/check/run_multi_sample.py` | (階段六) 加 `--sampled_file`、`--dump_file`、`--summary_file` CLI args 支援 100Q scale-up |
| `deeprag_compare/check/step1_feature_analysis.py` | (階段六) 加 `--dump_file`、`--out_dir` |
| `deeprag_compare/check/step3_role_overlap_v2.py` | (階段六) spaCy dependency-parse 版 detector + `--skip_step2` |

---

## Appendix A. Decision Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-23 | 用 deeprag_compare benchmark 對齊 DeepRAG | 取得跨方法可比較的 1000Q 規模結果 |
| 2026-04-25 | 加 dynamic restart mode + skip_baselines flag | 不重跑 baseline，省時 |
| 2026-04-26 | DeepRAG-SC threshold=8/12 失效後做 SC distribution 分析 | 發現實際 SC 分布在 17-22 |
| 2026-04-26 | Plan A (regen+masked) 實作並 4ds × 1000Q 跑 | 修 Dynamic-RAG 的 query/CoT 設計 |
| 2026-04-27 (上午) | Plan A 確認失敗（4ds 全部 ±0.01）；threshold=21/23 確認成功 | 寫入 paper：dynamic 是 negative result |
| 2026-04-27 (中午) | 跑 learned gate baseline vs SC@23 | 確認 SC ≥ learned，但 inherits DeepRAG-8b 訓練 |
| 2026-04-27 (晚間) | **Pivot 方向**：vanilla Llama-3-8B + Qwen-72B + 官方 4-shot prompt | 把 paper claim 升級到「SFT teacher (72B) 直接 inference + SC = 跳過 SFT+DPO」 |
| 2026-05-01 | exp0：sc_orig AUROC=0.4522 < 0.5 | 確認 full-CoT SC 訊號被 filler 稀釋；改抓 answer span |
| 2026-05-01 | first_ent_min_lp AUROC=0.7565（Δ=+0.30） | 證實 span-restricted lp 是強訊號 |
| 2026-05-02 | exp1：1000Q × 4ds gate 替換完成 | popqa@20%-budget +6.6 EM |
| 2026-05-02 | exp_a：Type-A 失敗模式量化 | 27-40% NR 錯誤為高信心 hallucination |
| 2026-05-02 | exp0b：no-CoT first-ent span-SC AUROC=0.8267 | SC 公式本身沒壞；CoT 還傷害答案訊號 |
| 2026-05-02 | 教授 feedback：黑盒子 critique → 重新 framing | 主軸從「SC gate」轉向「confident hallucination 子分類」|
| 2026-05-06 | Production sweep 4ds × 1000Q：no-CoT first-ent neg_max_ent@-0.80 | mean ΔEM +0.105，4/4 dataset 改善 |
| 2026-05-07 | Step 1：sc_mean AUROC=0.44 vs min_margin AUROC=0.96（pilot）| 「打開黑盒子」直接 evidence — 但後來 100Q 校正 |
| 2026-05-07 | Step 2：perturbation falsify A1 假設 | n_uniq=1 cases 對 input perturbation 完全敏感 |
| 2026-05-07 | Step 3 v1：role-overlap detector pilot | 5ade69e455 命中 B3，false positive 5abdf12255 待修 |
| 2026-05-07 | 子型重新分類為 B1/B2/B3 | 各對應不同 detector |
| 2026-05-08 | B3 detector v2 (spaCy)：false positive 修正 | 5abdf12255 v2=0.00 (DESCRIPTOR)，5ade69e455 維持 0.70 (CONSTRAINT) |
| 2026-05-08 | Scale-up 100Q × 20 samples 完成 | min_margin_full_mean 從 0.96 大跌至 0.58；最強 single feature 改為 n_uniq AUROC=0.81 |
| 2026-05-09 | 文檔重組：Gap 1-5 明確列出 | 把「主結果跟分析在不同 mode」、「analysis 沒連到 main result」等 reviewer 必問列為下一步優先順序 |

---

## Appendix B. Qwen-72B 完整 4ds × 200Q launch 指令

當 smoke test (hotpotqa 30Q) 確認格式對齊後，launch 完整 run：

```bash
cd /home/S113065528/Research/deeprag_compare
QWEN72B_PATH=/workspace/models/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31

nohup env \
  DATASETS="hotpotqa 2wikimultihopqa musique popqa" \
  MODEL_ID=$QWEN72B_PATH \
  MODEL_TAG=qwen2.5-72b-fs \
  HF_CACHE=/workspace/models \
  DEVICE=auto \
  PROMPT_STYLE=auto \
  USE_FEWSHOT=yes \
  CUDA_VISIBLE_DEVICES=2,3 \
  bash scripts/run_deeprag_sc.sh 200 sc 21 \
  > logs/qwen72b_fs_4ds_200q.log 2>&1 &
echo "PID=$!"
```

預估時間：**~5-7 hr**（FP16 across 2 GPU × 30s/Q × 800Q）。

監看：
```bash
tail -f /home/S113065528/Research/deeprag_compare/logs/qwen72b_fs_4ds_200q.log
ls -lt /home/S113065528/Research/deeprag_compare/results/*qwen2.5-72b-fs*_summary.json
```

跑完後：
```bash
/dataspace/S113065528/envs/research/bin/python scripts/compare_all.py
```
