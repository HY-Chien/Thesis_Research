# 兩週進度報告（4/30 – 5/10）

> 主題：Self-Certainty (SC) Gated RAG —— 用 LLM 內部不確定度訊號取代 DeepRAG 的 RL gate

---

## 一頁式總覽

### 研究脈絡
- **動機**：DeepRAG 用 RL 訓練 gate 決定每題要不要 RAG。我們要驗證「不訓 RL，只用 LLM 內部訊號」能否達到類似效果，並嘗試套到多跳資料集。
- **重大轉折（5/7）**：從「找更好的 SC threshold」轉向「拆解 confident hallucination 的子類型」。
- **最終結論（5/10）**：子類型標籤對 gating 雖無顯著增益，但揭露 RAG 本身的能力上限才是主要瓶頸。

### 實驗總表

| 階段 | 實驗 | 規模 | 涵蓋題數 | 比例 | 重點結論 |
|---|---|---|---|---|---|
| 1 | Exp0 answer-span SC | 20Q × 10s | 200 records | 2% | SC 在 answer span 訊號弱 |
| 1 | Exp0b no-CoT SC | 20Q × 10s | 200 records | 2% | no-CoT 模式趨勢類似 |
| 1 | Exp1 gate + rerank | 20Q | 20 | 2% | rerank 邊際收益 |
| 2 | Production sweep（4 dataset） | **4 × 1000Q** | 4000 | 100% × 4 | first_ent_min_lp gate 在 4/4 dataset 全部正向，**+3.6 ~ +15.1 pp** |
| 3 | Step1 token-feature ranking | 1000Q × 5 samples | 5000 | 100% | em_rate AUROC=0.949、first-entity entropy 0.78（強於 full-seq） |
| 3 | Step2 perturbation sensitivity | 10Q × 4 conditions × 10s | 400 | 1% | A1 vs A2 子類型可被輸入擾動分離 |
| 3 | Step3 B3 role-overlap detector v2 | 1000Q | 1000 | 100% | spaCy CONSTRAINT 偵測 ~1.9% B3，特異度 100% |
| 4 | **Step A — LLM judge subtype gate** | **998Q** | 998 | 99.8% | **E ≈ C（+0.001 EM）**；per-label help_rate 全部 < 19% |

### 關鍵 performance（HotpotQA 1000Q）

| 策略 | EM | trig% | 備註 |
|---|---|---|---|
| always-NR (no-CoT) | 0.218 | 0% | 下界 |
| always-RAG | 0.323 | 100% | 上界（最貴） |
| **C neg_max_ent @ 30%** | **0.265** | 30% | 現有最佳單訊號 |
| E subtype × neg_max_ent @ 30% | 0.266 | 30% | 子類型加入幾乎無效 |
| D subtype-aware binary | 0.319 | 73% | 規則式逼近 always-RAG 但便宜 27% |
| **Oracle @ 30%** | **0.389** | 30% | 上限 — 與 E 差 12.4 pp |

### Cross-dataset gate validation（4 × 1000Q）

| Dataset | always-NR | always-RAG | gate@45% trig | ΔEM vs NR |
|---|---|---|---|---|
| HotpotQA | 0.250 | 0.322 | 0.286 | +3.6 pp |
| 2WikiMultiHopQA | 0.320 | 0.377 | 0.349 | +2.9 pp |
| MuSiQue | 0.115 | 0.192 | 0.148 | +3.3 pp |
| **PopQA** | 0.199 | 0.416 | 0.302 | **+10.3 pp** |

→ Single-signal gate 在 4/4 dataset 全部正向；PopQA（單跳）最有效，多跳 dataset 改進相對小。

---

## 詳細介紹

### 為什麼做這個研究

DeepRAG 用 RL 訓練 gate 決定每題要不要 RAG。我們想驗證：**不需要 RL，只用 LLM 內部的不確定度訊號（Self-Certainty）能不能達到類似效果？**
這是計算便宜、不需訓練資料的替代方案。如果可以，下一步就是把這個 gate 接到多跳資料集，看能否在不訓 RL 的前提下逼近 DeepRAG 的表現。

---

### 實驗 1 — 早期 SC 探索（4/30 – 5/2，20Q pilot）

**目的**：在 20 題小樣本上測試 Self-Certainty 在不同位置（整句 vs answer-span）能否區分對/錯。

**規模**：20Q × 10–20 samples，total ~200–400 records。

**結果**：
- Exp0 answer-span SC：AUROC ~0.55（弱）
- Exp0b no-CoT SC：類似
- Exp1 加入 rerank：邊際收益

**意義**：**SC 作為單訊號區分力不夠** → 開啟尋找更強訊號的需求。

**佔比**：20/1000 = 2%（pilot 性質，沒有正式論文佐證價值）

---

### 實驗 2 — Production Sweep（5/6，4 dataset × 1000Q = 4000 題）

**目的**：要先確認「是否有任一個簡單 single-signal 在 production scale 上有效」，否則整個方向就要放棄。

**做法**：
- HotpotQA / 2Wiki / MuSiQue / PopQA 各 1000 題
- 用 `first_ent_min_lp`（no-CoT 答案第一個實體 span 的 min log-prob）當 gate 訊號
- 掃 4 個閾值（threshold = -1.5 / -1.1 / -0.8 / -0.5）

**結果**（gate@45% trig）：

| Dataset | ΔEM vs NR |
|---|---|
| HotpotQA | +3.6 pp |
| 2WikiMultiHopQA | +2.9 pp |
| MuSiQue | +3.3 pp |
| **PopQA** | **+10.3 pp** |

**意義**：證實 single-signal gate 確實有效，但**多跳 dataset 改進有限**，引導出後續：「是不是不同類型的錯誤需要不同處理？」

**佔比**：4000/4000 = 100%（首次 production-scale 驗證）

---

### 實驗 2b — Exp-A：失敗模式量化（六格矩陣）

**目的**：量化 confident hallucination 對後處理 gate 的天花板影響，驅動 5/7 轉向。

**做法**：以 `first_ent_min_lp` 中位數切 HIGH/LOW，交叉 (NR✓/✗) × (RAG saves/fails)，得六格矩陣。
重點格：**T3 = HIGH + NR✗ + RAG✗**（高信心答錯 + RAG 也救不了）= Type A。

**關鍵指標**：
- **Type-A share** = T3 / 所有 NR 錯（gate 理論天花板）
- **Discrimination** = P(wrong|LOW) − P(wrong|HIGH)（訊號區分力）

| Dataset | Type-A share | Discrim sc_orig | Discrim first_ent |
|---|---|---|---|
| HotpotQA | 33.2% | +0.123 | **+0.250** |
| 2WikiMultiHopQA | 27.6% | +0.068 | **+0.297** |
| MuSiQue | **39.9%** | +0.107 | +0.140 |
| PopQA | 29.0% | +0.028 | **+0.182** |

**關鍵觀察**：
1. **27–40% 的 NR 錯誤是 Type A**（高信心 + RAG 也無法拯救）— 這是後處理 gate 的理論天花板
2. **first_ent Discrim 比 sc_orig 強 1.5×–6.5×**（popqa：sc_orig +0.028 vs first_ent +0.182）
3. **MuSiQue 最難**：Type-A 最高（40%）+ Discrim 最弱 → 直接解釋為何 MuSiQue 改善最少

→ Script: `deeprag_compare/check/exp_a_failure_modes.py`

**佔比**：4000/4000 = 100%

---

### 實驗 3 — Confident Hallucination 子類型發掘（5/7 – 5/9）

**為什麼轉向（5/7）**：Production sweep 在多跳 dataset 改進小，懷疑 confident hallucination 不是單一現象。把它拆成子類型可能能用不同策略處理。

#### 子類型定義

| 標籤 | 描述 | 該不該 RAG |
|---|---|---|
| HI_correct | 5/5 sample 都對（模型本來就會） | ❌ |
| **B1** knowledge error | 都錯但答同一個錯誤（固定錯誤事實） | ✅ |
| **B2** random filling | 都錯且每次不同（schema 不穩） | ✅ |
| **B3** role confusion | 把題目裡的約束實體當成答案 | ❌（要重寫 prompt） |
| Other | 模糊（em_rate 中間） | ✅ |

#### Step 1 — Token-level Feature Ranking（1000Q × 5 samples）

**目的**：找出最能區分 HI_correct vs HI_wrong 的 token-level 特徵。

**結果**（AUROC 排名，1000Q scale）：

| Rank | 特徵名稱 | 位置 | AUROC |
|---|---|---|---|
| 1 | Multi-Sample Exact Match Rate（em_rate） | multi-sample | 0.949 |
| 2 | Maximum Uncertainty on Answer Entity（fe_max_entropy_mean） | first-entity | 0.784 |
| 3 | Unique Answer Count（n_uniq） | multi-sample | 0.784 |
| 4 | Lowest Confidence on Answer Entity（fe_min_lp_mean） | first-entity | 0.781 |
| 5 | Full-Sequence Self-Certainty | full-answer | 0.554 |
| 6 | Full-Sequence Perplexity | full-answer | 0.541 |

**關鍵洞見**：**first-entity span 的 entropy / log-prob 遠強於整句版本**（0.78+ vs 0.54）。不確定度集中在答案實體位置，full-sequence 平均後反而引入雜訊。

**佔比**：1000/1000 = 100%

#### Step 2 — Perturbation Sensitivity（10Q × 4 conditions × 10 samples）

**目的**：區分 A1（內部固定錯誤，輸入無關）vs A2（輸入敏感的錯誤）。

**原始假設（A1）**：n_uniq=1（五次輸出相同）→ 模型忽略 input，使用 parametric knowledge override。

**做法**：對每題做 4 種輸入擾動（original / mask 關鍵實體→[X] / replace 同型實體 / 加 distractor），觀察 modal prediction 與 n_uniq 怎麼變。

**關鍵結果 — A1 假設被 falsify**（以 `5a75383a55` 為例）：

| Condition | Modal pred | n_uniq | 解讀 |
|---|---|---|---|
| original | mirosław hermaszewski ×10 | 1 | 鎖死 |
| mask [X] | ulf merbold ×7, unknown ×3 | 2 | **完全變了** |
| replace Einstein | albert einstein ×10 | 1 | **完全變了** |
| distractor | mirosław hermaszewski ×10 | 1 | 不受無關前綴影響 |

mask 一下去答案立刻改變 → 模型理解 input，**n_uniq=1 只是「對特定 input 穩定」，不是 input-ignorant**。

**實驗暴露三種真實失敗模式（取代 A1/A2）**：

| 類型 | 描述 | 對擾動的反應 |
|---|---|---|
| **B1** knowledge error | schema 正確，知識有誤，輸出同一個錯答案 | input 一變立刻換答案 |
| **B2** random filling | schema 不穩，答案每次不同 | 各 condition 都飄 |
| **B3** role confusion | 把問題的約束實體誤當答案輸出 | mask 後 role confusion 直接消失 |

**佔比**：10/1000 = 1%（pilot 規模，但 falsify 了假設方向）

#### Step 3 — B3 Role Confusion Detector v2（1000Q）

**為什麼**：B3 是最有趣的子類型——代表 schema 解析失敗，靠 RAG 救不了。需要可重複偵測它。

**做法**：
1. 用 spaCy dependency parse 把問題實體分類成 CONSTRAINT / SUBJECT / DESCRIPTOR / OTHER
2. 偵測「答案實體 = 問題的 CONSTRAINT」

**結果**：
- 1000Q 中偵測到約 1.9% v2_b3_rate ≥ 0.2
- 跟 LLM judge 標的 0.6% 差距大（後者更嚴格）
- 100Q scale 維持 100% 特異度

**佔比**：1000/1000 = 100%

---

### 實驗 4 — Step A：LLM-as-Judge Subtype Gate（5/10）

**目的**：驗證「子類型標籤」是否比「single signal neg_max_ent」更會挑該 RAG 的題。

**做法**：
1. 用 GPT-OSS-20B 做 judge（vs 受測模型 Llama-3-8B-Instruct）標 1000Q 子類型
2. 30Q 手工標 gold 校驗：**Cohen's κ = 0.711**（substantial agreement）
3. 比較 5 個 gate 策略：
   - A always-NR
   - B always-RAG
   - C neg_max_ent（top k% 排序）
   - D 規則式（B1/B2/Other → RAG）
   - E 子類型 × neg_max_ent（剃掉 HI_correct/B3 後再排序）
4. × 4 個 budget（10/20/30/50%）

**Label 分布**（n=998）：HI_correct 264、B2 443、B1 178、Other 107、B3 6

**結果**：

| Budget | C | E | Oracle |
|---|---|---|---|
| 10% | 0.235 | 0.234 | 0.319 |
| 20% | 0.246 | 0.246 | 0.389 |
| 30% | 0.265 | **0.266 (+0.001)** | 0.389 |
| 50% | 0.288 | **0.295 (+0.007)** | 0.389 |

**Per-label help_rate**（這個 label 內有多少題 RAG 真的救得了）：

| label | n | help_rate |
|---|---|---|
| HI_correct | 264 | 14.8% |
| B1 | 178 | 18.5% |
| B2 | 443 | 17.6% |
| **B3** | 6 | **0%** |
| Other | 107 | 18.7% |

**核心結論**：
1. **E 比 C 只好 0.001 EM** → 子類型標籤對 gating 訊號**幾乎冗餘**
2. **per-label help_rate 全部 < 19%** → 真正瓶頸不在 gate，**在 RAG 本身對這些題就救不了**
3. **B3 6/6 全部 RAG 救不了** → 驗證「B3 需要重寫 prompt 不是 RAG」假設（n=6 太小但方向一致）
4. Oracle 0.389 vs always-RAG 0.323 → 即便完美 gating 也只能拿到 +6.6 pp，**RAG content 本身才是上限**

**佔比**：998/1000 = 99.8%（2 題 LLM judge PARSE_ERROR 已排除）

---

## 跟老師討論的 3 個重點

1. **Single-signal gate 在 production scale 有效**（4/4 dataset，PopQA +10.3 pp）— 可寫進論文的正向結果
2. **子類型 gate 是 negative result**，但**揭露的 Oracle headroom 才是真正的研究問題** — RAG 本身對 80%+ 題目沒救
3. **下一步建議**：用 B2（443 題）+ B3（6 題）作為 decomposition 實驗測試集，驗證「拆題後再 retrieve 是否能把 help_rate 從 17.6% 拉高」

---

## Pending / 後續計畫

- [ ] Qwen3-32B 跨模型 cross-check（驗證 LLM judge robustness）
- [ ] 寫入 EXPERIMENTS_SUMMARY.md
- [ ] Decomposition experiment design — B2/B3 子類型作為測試集
