# Self-Certainty 實驗總整理

> 模型：meta-llama/Llama-3.1-8B-Instruct
> 資料：NQ / 2WikiMultiHopQA，各 500 題（test_subsampled.jsonl）
> 評估：normalize_answer + EM / F1≥0.5 hybrid accuracy

---

## 第一階段：SC 信號驗證（exp0–exp7）

| 實驗 | 做了什麼 | 關鍵結論 |
|------|----------|----------|
| exp0 | SC vs PPL 相關性 | SC 與正確性相關，且比 PPL 穩定 |
| exp1–3 | SC 穩定性、CoT 軌跡 | SC 在重複抽樣下穩定；CoT 中 SC 會逐句變化 |
| exp4 | 分位數 vs 準確率 | 低 SC 分位 Q1 準確率最差 → SC 可作為 gating 信號 |
| exp5–7 | 定性案例、CoT 統計 | 低 SC 題目多為需要外部知識的事實型問題 |

---

## 第二階段：SC-Gated RAG（核心貢獻）

### 整體準確率對比（500 題）

| 實驗 | 設計 | NQ baseline → SC-RAG | 2wiki baseline → SC-RAG |
|------|------|----------------------|-------------------------|
| **exp8** Direct QA | 整題 SC 觸發（Q1, 20%） | 36.0 → 38.8 (**+2.8**) | 29.8 → 29.2 (−0.6) |
| **exp9** CoT-RAG | 整題 SC 觸發 + CoT | 40.6 → 43.4 (**+2.8**) | 26.8 → 30.0 (**+3.2**) |
| **exp10** Dynamic | 句子層級動態觸發（SC<12 or drop≥3） | 40.6 → 44.8 (**+4.2**) | 26.8 → 27.8 (+1.0) |
| **exp11** Decomposed | DeepRAG-inspired 拆解子問題 + SC 替換 classifier | 40.6 → 40.4 (−0.2) | 26.8 → 11.6 (−15.2) ❌ |

### Q1（低 SC 分位，底 20%）修復率

| 資料集 | exp8 Q1 | exp9 Q1 |
|--------|---------|---------|
| NQ | 17% → 31% (+14pp)，fixed=26 / regressed=12 | 19% → 33% (+14pp)，fixed=25 / regressed=11 |
| 2wiki | 13% → 10% (−3pp) ❌ | 1% → 17% (+16pp)，fixed=16 / regressed=0 ✅ |

---

## 核心發現

- ✅ **exp10 (Dynamic 句子層級)** 是目前 NQ 最強方法（+4.2pp）
- ✅ **exp9 (CoT-RAG)** 在 2wiki 最穩定（+3.2pp，0 regressed）
- ❌ **exp11 (Decomposed)** 失敗
  - NQ：96% 題目被強制拆成 3 個子問題，拆解品質差
  - 2wiki：錯誤累積嚴重（Regressed=104 >> Fixed=28），子問題答案互相污染
  - 問題不在 SC，而在**拆解品質**——DeepRAG 用 MDP 訓練 classifier 控制拆解，SC 無法替代這個角色

---

## 論文定位：SC 作為 RAG 觸發信號

**核心貢獻**：用 Self-Certainty（無需訓練的不確定性信號）取代 DeepRAG 的 learned classifier，決定何時做 retrieval。

### 方法對比

| | DeepRAG | 本研究 |
|--|---------|--------|
| 觸發信號 | MDP-trained classifier | Self-Certainty（無訓練） |
| 觸發粒度 | 子問題層級（decomposed） | 整題（exp8/9）或句子（exp10） |
| 需要額外訓練 | 是 | 否 |
| 適用場景 | Multi-hop（需要 decompose） | 所有 QA（通用） |

### DeepRAG 流程（exp11 嘗試替換）

1. **Decompose**：把問題拆成 atomic sub-questions
2. **Atomic Decision**：每個子問題用 MDP classifier 決定「parametric vs retrieval」← exp11 把這步換成 SC threshold
3. **Synthesize**：合併子答案

exp11 失敗原因：SC 能判斷「整體不確定」，但無法判斷「哪個子問題需要 retrieval」，且拆解錯誤會直接讓後續 SC 判斷失效。

---

## 外部基準比較（deeprag_compare 實驗）

> 使用 Meta-Llama-3-8B-Instruct，DRAGIN 8-shot CoT 格式，top-1000 dev，BM25 wiki index

詳見 `/home/S113065528/Research/deeprag_compare/results/comparison_table.md`

DeepRAG Table 2 基準（Llama-3-8B-Instruct）：

| 方法 | HotpotQA EM/F1 | 2Wiki EM/F1 | MuSiQue EM/F1 |
|------|---------------|-------------|---------------|
| CoT (paper) | 27.2 / 37.8 | 28.2 / 34.9 | 13.7 / 23.0 |
| DRAGIN (paper) | 27.6 / 38.1 | 29.1 / 35.7 | 11.8 / 20.0 |
| UAR (paper) | 29.7 / 40.7 | 34.8 / 42.4 | 19.1 / 28.4 |
| DeepRAG (paper) | 40.7 / 51.5 | 48.1 / 53.3 | 22.5 / 30.4 |

我們的方法（實驗跑完後更新）：見 comparison_table.md
