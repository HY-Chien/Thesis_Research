# 研究完整方向 — Self-Certainty Gated RAG with Learned Decomposition

**作者**：S113065528
**日期**：2026-05-12
**對應前期成果**：[progress_report_2weeks.md](progress_report_2weeks.md)
**預計執行期**：6–7 週

---

## 0. 一句話 pitch

> 我們證明 DeepRAG 用 RL 同時訓練的兩個元件（decomposer + gate），可以分別用「**SFT 監督式拆題**」和「**LLM 內部 entropy 訊號**」取代，達到接近 DeepRAG 的 multi-hop QA 表現，**且完全不需要 RL**。

---

## 1. 整體願景

DeepRAG 用 RL 訓練兩個核心元件：
1. **Decomposer**：複雜問題 → 子問題序列
2. **Gate**：每個子問題判斷該不該 retrieve

**本工作的替代方案**：

| 元件 | DeepRAG 做法 | 本工作做法 |
|---|---|---|
| Decomposer | RL-trained | **SFT on synthetic decomposition data** |
| Gate | RL-trained | **Entropy / self-certainty signal（免訓）** |
| 訓練資料來源 | reward signal（黑箱） | (1) MuSiQue gold + (2) GPT-OSS-20B 合成 |
| 訓練成本 | 高（RL pipeline） | 低（標準 SFT） |
| Gate 可移植性 | 綁特定 reward 設計 | 任何 LLM 直接抽 logits 即可 |

---

## 2. 兩階段 Pipeline 總覽

### Stage 1：建構 decomposition 訓練資料

| 資料來源 | Decomposition 從哪來 | 角色 |
|---|---|---|
| **MuSiQue** | Native `question_decomposition`（含 gold sub-Q + paragraph_support_idx） | **Gold anchor** |
| **HotpotQA** | GPT-OSS-20B prompted decomposition | Synthetic（需驗證） |
| **2WikiMultiHopQA** | GPT-OSS-20B prompted decomposition | Synthetic（需驗證） |

**為什麼選 GPT-OSS-20B 當 decomposer**：
- 已在前期 Step A 用過（pipeline 熟悉）
- 開源、結果可重現
- 20B 比受測模型 Llama-3-8B 大 → decomposition 能力顯著更強，可當 teacher

**品質控管（核心步驟）**：
1. **Calibration on MuSiQue**：拿 OSS-20B 對 MuSiQue 同題拆題，跟 gold decomposition 比對
   - 算 sub-Q 數量是否一致
   - sub-Q 之間變數綁定是否正確
   - 答對最終答案的拆題視為 valid
2. **Self-verify on HotpotQA / 2WikiMHQA**：用 oracle answer chain 驗證
   - 對每組合成的 (sub-Q1, sub-Q2)，跑 Llama 在 oracle paragraph 上回答 → 最終答案 == gold？
   - **答對才保留**（自篩 valid 訓練樣本）

### Stage 2：訓練 Llama-3-8B-Instruct（兩個能力）

**(2a) Decomposer**（SFT）
- 輸入：原始多跳問題 Q
- 輸出：sub-question list with `#N` variable binding
- 訓練資料：Stage 1 全部 valid decomposition（MuSiQue gold + HotpotQA/2WikiMHQA filtered）
- Loss：standard cross-entropy on decomposition tokens

**(2b) Entropy-based Gate**（免訓，calibrate threshold）
- 從前期實驗已知：**no-CoT + first-entity span 的 max_entropy / min_log_prob 是最強 gate signal**（AUROC 0.85+）
- Calibrate threshold：在 MuSiQue dev set 上找使 EM 最大的 threshold
- Threshold 一次決定，所有 dataset 共用（或 per-dataset 微調）

---

## 3. 完整 Inference Pipeline

```
Input: complex multi-hop question Q
       │
       ▼
┌─────────────────────────┐
│ Llama Decomposer (SFT)  │  ← Stage 2a 訓出
│   Q → [SubQ1, SubQ2, ...]│
└─────────────────────────┘
       │
       ▼
For each SubQ_i (sequential):
  ┌──────────────────────────────────────┐
  │ 1. Variable substitution:            │
  │    SubQ_i' = SubQ_i.replace("#j", A_j)│
  └──────────────────────────────────────┘
       │
       ▼
  ┌──────────────────────────────────────┐
  │ 2. Llama generates initial answer +  │
  │    first-entity span signals         │
  └──────────────────────────────────────┘
       │
       ▼
  ┌──────────────────────────────────────┐
  │ 3. Entropy Gate:                     │
  │    if max_ent < τ: A_i = initial     │ ← no retrieve
  │    else:                             │
  │      retrieve(SubQ_i') → P_i         │
  │      A_i = Llama(SubQ_i' | P_i)      │
  └──────────────────────────────────────┘
       │
       ▼
Final answer = A_last
```

**關鍵特性**：
- Gate 在每一跳獨立判斷，可能跑出「hop1 直接答、hop2 retrieve」的混合策略
- 每次 retrieve 都是針對一個原子 sub-question，不會被多跳 query 混淆

---

## 4. 實驗分四階段

### Phase 1：驗證假設（MuSiQue oracle decomposition）
**對應前一份 plan 的 C0–C4，免訓**

| Condition | 內容 | 角色 |
|---|---|---|
| C0 | NR baseline | lower bound |
| C1 | Single-pass RAG | 現有 baseline |
| C2 | Gold sub-Q retrieval | **驗證拆題本身有效** |
| C3 | Oracle paragraphs single-pass | retrieval 上限 |
| C4 | Oracle paragraphs + sub-Q answer | 拆題上限 |

- 規模：MuSiQue 1000Q
- 模型：Llama-3-8B-Instruct（no-CoT）
- **里程碑**：若 C2 > C1 → 證明拆題能突破 17% help_rate ceiling
- **若失敗（C2 ≈ C1）**：立刻停 — 拆題假設不成立，整個方向需要重新 reframe

### Phase 2：建構 synthetic decomposition 資料
- 用 GPT-OSS-20B 對 HotpotQA / 2WikiMHQA 各 1000Q 生成 decomposition
- 用 MuSiQue 對齊測試 OSS-20B 拆題品質
- 用 oracle answer chain 自篩 valid 樣本
- **里程碑**：
  - OSS-20B 在 MuSiQue calibration 上拆題正確率 ≥ X%
  - HotpotQA / 2WikiMHQA 各 500+ valid decomposition

### Phase 3：訓練 Llama-3-8B-Instruct Decomposer
- SFT on 合併資料集（MuSiQue gold + HotpotQA/2WikiMHQA filtered）
- 評估：
  - In-domain：held-out subset of training datasets
  - Out-of-domain：可選跨 dataset transfer
- **里程碑**：trained Llama 拆題正確率接近 OSS-20B（甚至超越 oracle 級拆題）

### Phase 4：整合 + 端到端評估
- 完整 pipeline：Llama Decomposer (Phase 3) + Llama answerer + entropy gate
- Baselines：
  - always-NR
  - always-RAG
  - DeepRAG（公開 checkpoint 或 reproduce）
  - 我們的 pipeline
- 評估 dataset：MuSiQue / HotpotQA / 2WikiMHQA
- **里程碑**：在多跳 QA 上 EM 接近或超過 DeepRAG

---

## 5. Timeline（7 週）

| 週次 | 階段 | 工作 | 里程碑 |
|---|---|---|---|
| W1 | Phase 1 | 寫 5 條 condition pipeline + smoke test | C0-C4 在 50Q 跑通 |
| W2 | Phase 1 | MuSiQue 1000Q 主實驗 | C2 vs C1 結果定向 |
| W3 | Phase 2 | OSS-20B 對 HotpotQA/2WikiMHQA 拆題 + MuSiQue calibration | 合成資料集 ready |
| W4 | Phase 3 | SFT Llama decomposer | trained checkpoint |
| W5 | Phase 3 | 評估 decomposer quality | 拆題正確率報表 |
| W6 | Phase 4 | 端到端 pipeline + baseline 比對 | 主表 EM |
| W7 | Phase 4 | DeepRAG cross-check + draft paper | progress report v2 |

---

## 6. 風險與 Fallback

| 風險 | 機率 | Fallback |
|---|---|---|
| Phase 1 失敗（C2 ≈ C1） | 中 | **整個方向轉向「retrieval limitation」** — 不訓 decomposer，改攻 retriever |
| OSS-20B 拆題品質太差（calibration 不過） | 低 | 改用 GPT-4o 或 Claude 當 teacher decomposer（但失去開源優勢） |
| Self-verify 過篩後資料量太少 | 中 | (a) 放寬篩選；(b) 用 retrieval-augmented decomposition prompt 提高品質 |
| SFT Llama decomposer 品質 < OSS-20B | 中 | (a) 多輪 iterative refinement；(b) 用 OSS-20B 蒸餾 |
| Entropy gate threshold 在不同 dataset 不一致 | 中 | per-dataset calibration（仍可接受） |
| DeepRAG checkpoint 拿不到 | 低 | 用其論文報的數字當對比 |

---

## 7. 與 DeepRAG 的對比（論文 positioning）

|  | DeepRAG | **本工作** |
|---|---|---|
| Decomposer | RL-trained | **SFT on synthetic data** |
| Gate | RL-trained | **Entropy threshold (免訓)** |
| 訓練 pipeline 複雜度 | 高（reward + policy） | 中（單純 SFT） |
| 訓練資料來源 | reward signal（implicit） | (1) MuSiQue gold + (2) OSS-20B synthetic（explicit） |
| Inference 成本 | 中 | 低（gate 無額外計算） |
| 可移植性 | 綁訓練時 reward | 任何 LLM 即插即用 entropy gate |
| 論文貢獻 | RL 設計 | **證明 RL 不必要** |

---

## 8. 為什麼這個方向值得做

1. **承接前期所有正向結果**：
   - 兩週進度證明 entropy signal 對 gate 強（AUROC 0.85+，4/4 dataset 都正向）
   - 也證明 single-signal gate 的 EM 增益被 RAG ceiling 卡住
   - 本工作的拆題實驗剛好突破這個 ceiling

2. **承接前期所有 negative 結果**：
   - Step A subtype gate 失敗 → 證明 gate signal 已逼近上限，要進的是 RAG content 品質
   - 拆題 retrieval 正是攻 RAG content 品質的乾淨方法

3. **三個劇本都有故事**：
   - 拆題贏 → no-RL DeepRAG（主線）
   - 拆題部分贏 → 拆題 + retriever 改進
   - 拆題輸 → reframe 成 retrieval limitation paper

4. **資料路徑乾淨**：
   - MuSiQue gold decomposition 是 hard anchor
   - OSS-20B 拆 HotpotQA/2WikiMHQA 是已驗證流程
   - SFT pipeline 標準，無 RL 風險

---

## 9. 與兩週進度的銜接

兩週進度交付的三個 publishable 點，本工作分別擴展：

| 兩週進度結論 | 本工作擴展 |
|---|---|
| 1. Single-signal gate 在 4 dataset 全部正向 | **沿用為 Phase 4 gate 元件**，不重做 |
| 2. B3 systematic detector | 獨立論文點，本工作不動 |
| 3. RAG ceiling characterization（help_rate < 19%） | **本工作的核心 motivation** — 用拆題打破 ceiling |

---

## 附錄 A：相關檔案
- 既有實驗位置：[/home/S113065528/Research/deeprag_compare/check/](.)
- 兩週進度：[progress_report_2weeks.md](progress_report_2weeks.md)
- 投影片大綱：[progress_slides_outline.md](progress_slides_outline.md)
- 前一版 plan（單 Phase 1）：[plan_musique_decomp.md](plan_musique_decomp.md)

## 附錄 B：MuSiQue Gold Decomposition 範例
```
2hop__482757_12019
Question: When was the institute that owned The Collegian founded?
question_decomposition: [
  { id: 482757, question: "The Collegian >> owned by",
    answer: "Houston Baptist University", paragraph_support_idx: 5 },
  { id: 12019, question: "When was #1 founded?",
    answer: "1960", paragraph_support_idx: 9 }
]
Gold answer: 1960
```
→ 每題附 sub-Q list、各自 gold answer、各自 gold paragraph。
這是 Phase 1 的 oracle 來源，也是 Phase 2 calibration 的 anchor。
