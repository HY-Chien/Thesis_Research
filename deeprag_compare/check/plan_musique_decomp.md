# 研究規劃 — MuSiQue Gold Decomposition 突破 RAG Ceiling

**作者**：S113065528
**日期**：2026-05-12
**對應前期成果**：[progress_report_2weeks.md](progress_report_2weeks.md)、[progress_slides_outline.md](progress_slides_outline.md)
**預計執行期**：3 週

---

## 1. 出發點（為什麼做這個）

### 1.1 兩週進度的結論
- ✅ **Single-signal gate 在 4/4 dataset 都有效**（PopQA +10.3pp、HotpotQA +3.6pp、2WikiMHQA +2.9pp、MuSiQue +3.3pp）
- ❌ **加子類型訊號（Step A LLM judge）幾乎沒幫助**：E (subtype × neg_max_ent) 比 C (neg_max_ent) 只多 +0.001 ~ +0.007 EM
- ⚠️ **真正的瓶頸暴露**：per-label help_rate **全部 < 19%**（HI_correct 14.8%、B1 18.5%、B2 17.6%、B3 0%、Other 18.7%）

### 1.2 瓶頸詮釋
即使是「應該救」的 B1/B2/Other 子類型，**RAG 也只能救其中 17–19% 的題**。
→ **問題不在 gate**（已逼近 Oracle 上限的 76%），**問題在 RAG 內容本身對多數題目沒幫助**。

### 1.3 一個合理的猜想
為什麼 RAG retrieval 對多跳題撈不到對的內容？
**假設**：retriever 拿整題 2-hop question 當 query，會混在一起檢索 → 取回的 paragraph 對 hop1 和 hop2 都不夠精確。
→ 如果**先拆題、再分別 retrieve**，retrieval 命中率應該大幅提高，help_rate 應該突破 17% 上限。

這正是 DeepRAG 用 RL 訓 decomposer 的動機；但 DeepRAG 需要訓練資料。**我們的目標：在不訓練的前提下驗證這個假設。**

---

## 2. 為什麼選 MuSiQue（資料層面的關鍵突破）

### 2.1 MuSiQue 是唯一 native 提供 gold decomposition 的多跳 dataset

| 資源 | HotpotQA | 2WikiMHQA | **MuSiQue** | PopQA |
|---|---|---|---|---|
| 多跳 | 2-hop | 2-4 hop | **2-4 hop** | 單跳 |
| Gold sub-questions | ✗ | ✗ | **✓** `question_decomposition` 欄位 | N/A |
| Gold paragraph per hop | sentence-level facts | 部分 | **✓** `paragraph_support_idx` | N/A |
| `#N` 變數綁定 | ✗ | ✗ | **✓**（e.g. "When was #1 founded?"） | N/A |

### 2.2 MuSiQue 也是「最難、RAG 最沒幫助」的 dataset
從現有 Exp-A 數據（[exp_a_summary.md](exp_a_output/exp_a_summary.md)）：

| Dataset | always-NR | always-RAG | Type-A share | RAG save (LOW) |
|---|---|---|---|---|
| HotpotQA | 0.250 | 0.322 | 0.332 | 0.170 |
| 2WikiMHQA | 0.320 | 0.377 | 0.276 | 0.177 |
| **MuSiQue** | **0.115** | **0.192** | **0.399** | **0.123** |
| PopQA | 0.199 | 0.416 | 0.290 | 0.380 |

→ MuSiQue 是 4 dataset 裡：
1. **NR 表現最差**（0.115）
2. **RAG 也最沒救**（0.192，only +7.7pp gain）
3. **Type-A share 最高**（39.9% 的錯誤是 RAG 救不回來的）
4. **LOW-conf 時 RAG save rate 最低**（12.3%）

**意涵**：如果拆題能在 MuSiQue 這個最難的 dataset 上 work，那就有強說服力可以推廣。

### 2.3 為什麼不直接做 HotpotQA？
HotpotQA 雖然 1000Q 已經有完整 Step A subtype 標籤（HI_correct 264 / B1 178 / B2 443 / B3 6 / Other 107），但**它沒有 gold sub-question 拆解**。
→ 如果要在 HotpotQA 做拆題實驗，必須**自己訓 / prompt 一個 decomposer**，這會引入混淆變因（拆得好不好 vs 拆題本身有沒有用）。
→ MuSiQue 內建一個**完美 oracle decomposer**，可以**先排除「拆題品質」這個變因**，乾淨地驗證假設。

---

## 3. 核心假設

| 編號 | 假設 | 預測結果 | 怎麼解讀 |
|---|---|---|---|
| **H1** | 拆題後分別 retrieve > 整題單次 retrieve | C2 EM > C1 EM | 拆題本身有效 |
| **H2** | 拆題 retrieval 接近 oracle paragraph 上限 | C2 ≈ C3（gap < 5pp） | retrieval 不是 bottleneck，拆題能補上 |
| **H3** | 改善主要來自 B2 / Other 子類型 | per-subtype help_rate 在 B2/Other 顯著上升 | B1 是真知識錯（拆題救不了），B2 是 retrieval miss（拆題能救） |
| **H4** | B3 仍然救不了 | B3 help_rate ≈ 0% | B3 是 prompt 解析錯誤，跟 retrieval 無關 |

---

## 4. 實驗設計

### 4.1 受測 Pipelines（5 條 condition）

| ID | 名稱 | Retrieval query | Answer 用什麼 context | 角色 |
|---|---|---|---|---|
| **C0** | No-RAG | — | — | Baseline (lower bound) |
| **C1** | Single-pass RAG | 原始 2-hop question | top-k retrieved paragraphs | 現有 baseline |
| **C2** | **Gold Sub-Q Retrieval** | Sub-Q1, 然後代入 #1 後的 Sub-Q2 | C2 retrieved paragraphs | **主實驗** |
| **C3** | Oracle Paragraph (single-pass) | — | gold `paragraph_support_idx` 兩篇 | Retrieval upper bound |
| **C4** | Oracle Paragraph + Sub-Q Answer | — | gold paragraphs，但 sequential 回答 sub-Q1 → sub-Q2 | 純拆題上限 |

### 4.2 C2 詳細 pipeline

```
Input: 2-hop question Q with question_decomposition = [SubQ1, SubQ2(#1)]

Step 1: retrieve(SubQ1) → paragraphs P1 → LLM(SubQ1 | P1) → A1
Step 2: SubQ2_resolved = SubQ2.replace("#1", A1)
        retrieve(SubQ2_resolved) → paragraphs P2 → LLM(SubQ2_resolved | P2) → A2
Step 3: Final answer = A2
```

**關鍵實作細節**：
- A1 用 `clean_pred` 後做字串替換（避免 LLM 自由生成的雜訊干擾）
- 若 SubQ2 含多個 `#N`（3-hop+ 題目），遞迴展開
- LLM 溫度 T=0，與既有實驗一致

### 4.3 規模 / 模型 / 訊號

| 項目 | 設定 |
|---|---|
| Dataset | MuSiQue 2-hop subset, 1000Q (與 Step A 同 sample) |
| 受測模型 | Llama-3-8B-Instruct（與既有實驗一致，無 CoT） |
| Retriever | Elasticsearch BM25 + dense (deeprag_compare 既有 setup) |
| Top-k | k=5（與 C1 baseline 一致） |
| Sampling | T=0 greedy，1 run/題 |
| 評估 | EM, F1（與 Adaptive-RAG 標準一致） |

### 4.4 子類型切分分析（如果完成 4.5 預備工作）
若先跑完 Step A LLM judge 在 MuSiQue 上，可額外做：
- **Per-subtype EM**：每條 condition 在 HI_correct / B1 / B2 / B3 / Other 各自的 EM
- **Per-subtype help_rate**：`P(condition correct | NR wrong)` 在每個 subtype 上
- **核心對照**：C2 vs C1 的 per-subtype delta，驗證 H3

### 4.5 預備工作（前置步驟）

| 步驟 | 描述 | 是否必須 |
|---|---|---|
| P1 | 確認本地 MuSiQue raw 含 `question_decomposition` 與 `paragraph_support_idx` 欄位 | **必須** |
| P2 | 確認 Elasticsearch index 已 index MuSiQue 所有 paragraph corpus | **必須** |
| P3 | 在 MuSiQue 1000Q 上跑 Step A LLM judge 取得 subtype 標籤 | 選用（為 H3/H4 提供切分） |
| P4 | 在 MuSiQue 1000Q 上跑 Exp-A 六格矩陣（取得 Type-A 比例與 first_ent_min_lp gate 訊號） | 選用（接 Step A） |

---

## 5. 預期結果與三種劇本

### 劇本 A：拆題有效（C2 顯著 > C1，且 C2 ≈ C3）
- **主張**：MuSiQue 上 RAG ceiling 主因是「整題 retrieval 失敗」
- **論文方向**：no-train decomposition gate — LLM 自信度當 gate，gold decomposition 當 actionable intervention
- **預期 EM**：C1 ≈ 0.19 → C2 預期 0.25–0.30
- **論文貢獻**：第一篇用「不訓 RL、不訓 decomposer」逼近 DeepRAG 的工作

### 劇本 B：拆題部分有效（C2 > C1 但 C2 ≪ C3）
- **主張**：拆題能補一些，但 retrieval 本身還有 gap
- **論文方向**：拆題 + retriever 改進的組合
- **後續實驗**：換 dense retriever、加 query rewriting

### 劇本 C：拆題無效（C2 ≈ C1）
- **主張**：retrieval 本身是 bottleneck，跟拆不拆題無關
- **論文方向**：改 reframe 成 "characterizing retrieval limitation in confident hallucination"
- **後續實驗**：refusal mechanism、retriever 替換

→ **三個劇本都有 publishable 故事**，不是 win-or-lose 的賭注。

---

## 6. Timeline（3 週）

### Week 1（5/13 – 5/19）：資料 + Pipeline
- [ ] P1: 確認 MuSiQue raw 在本地（HF dataset `dgslibisey/MuSiQue` 或既有 download）
- [ ] P2: 驗證 ES index 涵蓋 MuSiQue paragraph corpus（必要時重 build）
- [ ] 寫 C2 pipeline script（sub-Q retrieval + 變數綁定 + 序列回答）
- [ ] 寫 C3/C4 pipeline script（oracle paragraph 灌入）
- [ ] **里程碑**：5 條 condition 在 50Q smoke test 上能跑通

### Week 2（5/20 – 5/26）：執行 + 分析
- [ ] 跑 5 條 condition × MuSiQue 1000Q
- [ ] 計算 aggregate EM / F1
- [ ] 若 P3/P4 已完成 → per-subtype 切分
- [ ] **里程碑**：產出主表（5 condition × EM/F1）+ 1 張 per-subtype 圖

### Week 3（5/27 – 6/2）：寫作 + Cross-check
- [ ] 撰寫 short report（3–5 頁）
- [ ] 在 HotpotQA 用 LLM-prompted decomposer 做 partial replication（驗證 MuSiQue 結論可否推廣）
- [ ] **里程碑**：交給老師的 progress report v2

---

## 7. 風險與 Fallback

| 風險 | 機率 | Fallback |
|---|---|---|
| 本地 MuSiQue 沒有 decomposition 欄位 | 中 | 從 HF `dgslibisey/MuSiQue` 重 download，用 question_id join |
| ES index 沒有 MuSiQue paragraph | 中 | 用既有 retriever_server build script 重 index |
| Sub-Q2 變數綁定錯誤（A1 含多種 alias） | 中 | 用 gold A1（從 decomposition 欄位）取代 LLM 預測 A1，作為「拆題上限」對照 |
| C2 ≈ C1（劇本 C） | 中 | 改寫故事為 "retrieval limitation"，仍可發表 |
| Step A LLM judge 在 MuSiQue 上 PARSE_ERROR 過高 | 低 | 沿用 HotpotQA 的 GPT-OSS-20B + JSON fallback parser |

---

## 8. 與 DeepRAG 的關係（論文定位）

| | DeepRAG | **本工作** |
|---|---|---|
| Gate | RL-trained | **LLM 內部訊號（self-certainty / first-entity entropy）** |
| Decomposer | RL-trained | **MuSiQue gold decomposition（oracle）→ 接 LLM-prompted decomposer** |
| 訓練成本 | 需要訓練資料 + RL pipeline | **零訓練** |
| 評估 dataset | 多跳 QA | 同 |

**論文一句話 pitch**：
> 我們證明 DeepRAG 用 RL 學到的兩個元件（gate + decomposer），可以分別用「LLM 自信度訊號」和「gold / prompted decomposition」替代，在 MuSiQue 上逼近 RL-trained DeepRAG 的表現。

---

## 9. 與兩週進度報告的銜接

- 兩週進度報告的最後一個結論：「真正的瓶頸是 RAG content 本身對多數題沒幫助」
- 這份規劃直接回應該結論：**拆題是不是能突破這個 17% ceiling？**
- 答案無論正反，都讓兩週的 negative result（Step A subtype gate 失敗）變成有意義的 motivation。

---

## 附錄 A：相關檔案
- 既有實驗位置：[/home/S113065528/Research/deeprag_compare/check/](.)
- 兩週進度：[progress_report_2weeks.md](progress_report_2weeks.md)
- 投影片大綱：[progress_slides_outline.md](progress_slides_outline.md)
- Exp-A 六格矩陣：[exp_a_output/exp_a_summary.md](exp_a_output/exp_a_summary.md)
- Step 2 perturbation：[step2_output/sensitivity_table.md](step2_output/sensitivity_table.md)
- Step A gate 對比：[step4_output/gate_comparison.md](step4_output/gate_comparison.md)

## 附錄 B：MuSiQue 範例（佐證 gold decomposition 可用）
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
→ 每題附 2 個 sub-Q、各自的 gold answer、各自的 gold paragraph index。
這就是「免訓 decomposer」的基礎。
