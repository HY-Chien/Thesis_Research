# 兩週進度報告 — 投影片大綱

> 預計 12–14 張投影片，~15 分鐘報告

---

## Slide 1 — 封面
- **標題**：Self-Certainty Gated RAG — 兩週進度報告
- **副標**：4/30 – 5/10
- **報告人 / 日期**

---

## Slide 2 — 研究動機（Why）
- DeepRAG 用 **RL** 訓練 gate 決定「這題要不要 RAG」
- 我們的問題：**能不能用 LLM 內部訊號（Self-Certainty）直接當 gate？**
  - 不需訓練資料
  - 計算成本低
  - 可以泛化到多跳資料集
- **目標**：在不訓 RL 的前提下逼近 DeepRAG performance

---

## Slide 3 — 兩週時間軸（一張圖）
```
4/30 ─── 5/2 ─── 5/6 ──────── 5/7 ──────── 5/9 ────── 5/10
 │        │       │            │            │          │
Pilot   Failure Production   *PIVOT*     1000Q       Step A
20Q     modes   sweep        子類型分類   feature     LLM judge
                4×1000Q                  + B3 detect
```
**重大轉折（5/7）**：從「找更好的 SC threshold」轉向「拆解 confident hallucination 子類型」

---

## Slide 4 — 實驗總表（一張表）

| 階段 | 實驗 | 規模 | 比例 |
|---|---|---|---|
| 1 | Early SC pilot | 20Q | 2% |
| 2 | Production sweep | 4 dataset × 1000Q | 100% × 4 |
| 3 | Token-feature ranking | 1000Q × 5 samples | 100% |
| 3 | Perturbation sensitivity | 10Q × 4 conditions | 1% |
| 3 | B3 role-confusion detector | 1000Q | 100% |
| 4 | **Step A: LLM judge gate** | **998Q** | **99.8%** |

→ 主要 production result 都在 1000Q 規模驗證

---

## Slide 5 — Production Sweep：Single-signal Gate 在 4 個 Dataset 都有效

**訊號**：`first_ent_min_lp`（no-CoT 答案第一個實體 span 的 min log-prob）

| Dataset | always-NR | always-RAG | gate@45% | ΔEM vs NR |
|---|---|---|---|---|
| HotpotQA | 0.250 | 0.322 | 0.286 | **+3.6 pp** |
| 2WikiMultiHopQA | 0.320 | 0.377 | 0.349 | **+2.9 pp** |
| MuSiQue | 0.115 | 0.192 | 0.148 | **+3.3 pp** |
| **PopQA** | 0.199 | 0.416 | 0.302 | **+10.3 pp** |

→ 4/4 全部正向；**PopQA（單跳）效果最好**，多跳 dataset 改進有限
→ **這是論文的正向結果**

---

## Slide 6 — 為什麼 5/7 轉向？
**觀察**：多跳 dataset 改進有限（HotpotQA 只 +3.6 pp）

**假設**：confident hallucination 不是單一現象，是**多種子類型混在一起**：
- 有些題模型 **知識真的錯**（B1）→ RAG 能救
- 有些題模型 **schema 不穩**（B2）→ RAG 可能救
- 有些題模型 **解析錯誤**（B3）→ RAG 救不了
- 不同子類型應該用**不同 gate 策略**

→ 開啟 Step A：用子類型標籤當 gate

---

## Slide 7 — Confident Hallucination 子類型定義

| 標籤 | 描述 | 該不該 RAG |
|---|---|---|
| HI_correct | 5/5 sample 都對（模型本來就會） | ❌ |
| **B1** knowledge error | 都錯但答**同一個**錯誤 | ✅ |
| **B2** random filling | 都錯且**每次不同**（schema 不穩） | ✅ |
| **B3** role confusion | 把題目裡的**約束實體**當成答案 | ❌（要重寫 prompt） |
| Other | em_rate 中間（2/5、3/5 對） | ✅ |

**B3 例子**：問「導演 X 主演的電影由誰拍？」，模型回答 X（X 是約束，不是答案）

---

## Slide 8 — Step 1: Token Feature Ranking（1000Q × 5 samples）
**問題**：哪個 token-level 訊號最能區分 HI_correct vs HI_wrong？

| Rank | Feature Name | Location | AUROC |
|---|---|---|---|
| 1 | Multi-Sample Exact Match Rate | multi-sample | 0.949 |
| 2 | **Maximum Uncertainty on Answer Entity** | first-entity | 0.784 |
| 3 | Unique Answer Count | multi-sample | 0.784 |
| 4 | **Lowest Confidence on Answer Entity** | first-entity | 0.781 |
| 5 | Full-Sequence Self-Certainty | full-answer | 0.554 |
| 6 | Full-Sequence Perplexity | full-answer | 0.541 |

**關鍵發現**：**first-entity span 的 entropy/log-prob 遠強於整句版本（0.78+ vs 0.54）** —— 不確定度集中在答案實體位置

---

## Slide 9 — Step A: LLM-as-Judge 設計
- **Judge 模型**：GPT-OSS-20B（vs 受測 Llama-3-8B-Instruct，避免循環）
- **校驗**：30Q 手工標 gold → Cohen's κ = **0.711**（substantial agreement）
- **規模**：1000Q（998 valid，2 PARSE_ERROR）
- **Label 分布**：HI_correct 264 (26%) / B1 178 (18%) / B2 443 (44%) / B3 6 (0.6%) / Other 107 (11%)

→ B2 是最大群（44%）；B3 比預期稀有

---

## Slide 10 — Step A: 5 個 Gate 策略 × 4 Budget

**核心問題**：子類型訊號加進來會比 single signal 好嗎？

| Budget | C neg_max_ent | E subtype × neg_max_ent | Oracle |
|---|---|---|---|
| 10% | 0.235 | 0.234 | 0.319 |
| 20% | 0.246 | 0.246 | 0.389 |
| 30% | **0.265** | **0.266 (+0.001)** | 0.389 |
| 50% | 0.288 | 0.295 (+0.007) | 0.389 |

**Negative result**：E 比 C 只好 0.001 EM —— **子類型訊號幾乎冗餘**

---

## Slide 11 — 為什麼 E ≈ C？Per-label Help Rate

| label | n | RAG help_rate |
|---|---|---|
| HI_correct | 264 | 14.8% |
| B1 | 178 | 18.5% |
| B2 | 443 | 17.6% |
| **B3** | 6 | **0%** |
| Other | 107 | 18.7% |

**關鍵發現**：**全部子類型的 help_rate 都 < 19%**
→ 即使是「該救」的 B1/B2/Other，**RAG 也只能救 17–19% 的題**
→ **問題不在 gate，在 RAG 本身的能力上限**

---

## Slide 12 — Oracle Gap 揭露真正瓶頸
- always-RAG = 0.323
- Oracle @ 30% = **0.389**
- 完美 gate 比 always-RAG 只多 +6.6 pp
- C/E 比 Oracle 少 12.4 pp

**結論**：
- ✅ Single-signal gate 有效（+3 ~ +10 pp）
- ❌ 加子類型沒幫助
- ⚠️ **真正的瓶頸是 RAG content 本身對多數題沒幫助**

---

## Slide 13 — 三個論文可寫的點
1. **Single-signal gate 在 4 dataset 全部正向** — 計算便宜替代 RL gate
2. **B3 systematic detector**：spaCy CONSTRAINT 偵測 + LLM judge 驗證（特異度 100%，prevalence ~0.6%–1.9%）
3. **RAG ceiling characterization**：per-label help_rate 全部 < 19% —— 第一篇明確量化「multi-hop QA 中 RAG 對哪些題沒救」的工作

---

## Slide 14 — 下一步（請老師指示方向）
**Option A**：寫 Step A negative result + 完成 Qwen3-32B cross-check（保守路線，1 週）

**Option B**：直接做 decomposition 實驗
- 用 B2（443 題）+ B3（6 題）作為測試集
- 驗證「拆題後 retrieve 子問題能否把 help_rate 從 17.6% 拉高」
- 這是接 DeepRAG 的核心想法（2 週）

**個人建議**：**Option B** — Step A 已給出足夠證據說明「不拆題的 RAG 有上限」，正好接 decomposition

---

## 附錄（備用）

### A1 — 實驗檔案位置
- `/home/S113065528/Research/deeprag_compare/check/`
- 主要結果：`step4_output/gate_comparison.md`、`strategy_diff_summary.md`

### A2 — Cohen's κ 詳細
- 30Q 手工標 → judge 標
- κ = 0.711 (substantial)
- B3 perfect agreement (1/1)

### A3 — PARSE_ERROR 處理
- 6 個 GPT-OSS-20B Harmony 格式輸出 token 不夠長
- 從 256 → 1024 → 2048 max_new_tokens
- 加上 JSON-in-analysis fallback parser
- 最終 998/1000 valid（99.8%）
