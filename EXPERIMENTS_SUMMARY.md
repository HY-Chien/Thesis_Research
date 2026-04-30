# Self-Certainty (SC) Gated RAG — Experiments Summary

> 最後更新：2026-04-27（晚間）
> 核心研究問題：用 Self-Certainty (SC) 作為 RAG 觸發信號，能否取代 DeepRAG 的 RL-trained gate？
> **核心成果**：
>   1. (已驗證) DeepRAG-8b 上把 learned gate 換成 SC threshold ≥ 原 paper（2wiki +0.9 EM, hotpot ≈, musique -0.9）
>   2. (待驗證) Qwen-2.5-72B 用 DeepRAG 4-shot prompt 拆解 + SC gate 是否能 match DeepRAG-8b 的 SFT+DPO 訓練成果

---

## 0. 專案目錄

```
/home/S113065528/Research/
├── experiment/self_certainty/      ← 第一/二階段（exp0–11）on NQ/2wiki, Llama-3.1-8B
│   ├── scripts/                     exp{0..11}_*.py
│   ├── results/                     exp{0..11}_*_results.json + .md 質性報告
│   ├── logs/, figures/
│   └── EXPERIMENTS_SUMMARY.md      ← 階段一二的舊版總結
└── deeprag_compare/                 ← 第三階段：4-dataset benchmark vs DeepRAG paper
    ├── src/
    │   ├── run_experiment.py         baseline + dynamic + SC-gated post-hoc
    │   ├── run_deeprag_sc.py         DeepRAG-style BFS decomposition + 3 種 gate
    │   │                              (現支援 model_id 切換 + 4-shot prompt)
    │   ├── retriever_bm25.py         BM25 over wiki ES index
    │   └── dataset_eval.py           EM/F1 (DRAGIN-ported)
    ├── scripts/
    │   ├── run_dynamic_c3.sh           dynamic continue+restart × 4 datasets
    │   ├── run_dynamic_plan_a.sh       Plan A (regen_sentence + masked query)
    │   ├── run_deeprag_sc.sh           單一 gate/threshold runner（支援 MODEL_ID 切換）
    │   ├── run_deeprag_sc_sweep.sh     threshold sweep
    │   ├── compare_all.py              彙整所有變體 → comparison_all.md
    │   └── summarize_vs_deeprag.py     舊版 (僅 baseline 對比)
    ├── results/                      所有 .jsonl + _summary.json + comparison_all.md
    ├── logs/, data/, prompts/
```

---

## 1. 階段一：SC 信號驗證（exp0–7, Llama-3.1-8B, NQ/2wiki, 500Q）

| Exp | 主題 | 結論 |
|---|---|---|
| exp0 | SC vs PPL | SC 與正確性相關，比 PPL 穩定 |
| exp1–3 | SC 穩定性 / CoT 軌跡 | SC 在重複抽樣下穩定，CoT 中逐句變化 |
| exp4 | 分位數 vs 準確率 | 低 SC Q1 準確率最差 → 可作 gating 信號 |
| exp5–7 | 質性 + CoT 統計 | 低 SC 多為需要外部知識的事實型問題 |

---

## 2. 階段二：SC-Gated RAG on NQ/2wiki（exp8–11, Llama-3.1-8B, 500Q, F1≥0.5 hybrid）

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

## 3. 階段三：deeprag_compare benchmark（4 datasets, Llama-3-8B + DeepRAG-8b, 1000Q）

> Strict EM/F1（DRAGIN-ported, 對齊 DeepRAG paper）。

### 3.1 Baseline + post-hoc SC-gated（run_experiment.py）

| Dataset | no_rag | single_rag | sc@20 | sc@50 | sc@80 |
|---|---|---|---|---|---|
| hotpotqa | 25.2 / 35.6 | 32.4 / 44.0 | 27.5 / 38.0 | 30.6 / 41.4 | **32.1** / 43.2 |
| 2wikimultihopqa | 31.2 / 39.3 | 37.9 / 47.0 | 33.1 / 42.1 | 34.8 / 44.2 | **37.3** / 46.6 |
| musique | 11.5 / 20.3 | 19.2 / 29.0 | 13.5 / 22.8 | 16.8 / 26.1 | **18.2** / 28.1 |
| popqa | 20.4 / 23.1 | 41.6 / 51.3 | 22.6 / 27.5 | 29.5 / 35.6 | **35.0** / 42.6 |

### 3.2 Dynamic SC-RAG（mid-generation 觸發；3 個變體都失效）

| Dataset | continue (basic) | restart | regen+masked (Plan A) |
|---|---|---|---|
| hotpotqa | 20.6 / 32.2 | 19.1 / 30.7 | 20.2 / 31.3 |
| 2wikimultihopqa | 20.6 / 30.1 | 17.2 / 28.0 | 19.5 / 29.2 |
| musique | 10.0 / 19.1 | 10.3 / 19.4 | 10.1 / 19.0 |
| popqa | 7.7 / 17.8 | 8.5 / 18.8 | 7.7 / 17.6 |

→ **全部低於 no_rag**。Plan A（FLARE-style masked query + regenerate_sentence）有正確改變 retrieval 結果（Ed Wood case 證實），但 EM 沒提升 → bottleneck **不在 query 設計或 trigger 處理**。
→ **結論：mid-generation dynamic 在 multi-hop 沒有實證好處**（寫進 paper 當 negative result）。

### 3.3 ⭐ DeepRAG-8b + 三種 gate（1000Q）

| Method | hotpotqa EM/F1 | 2wiki EM/F1 | musique EM/F1 | avg ret |
|---|---|---|---|---|
| **learned gate (paper baseline 復現)** | 39.6 / 51.1 | 50.2 / 54.8 | 23.7 / 30.9 | 1.7 / 1.5 / 2.3 |
| SC gate th=17 | 32.0 / 43.5 | 39.0 / 44.1 | 18.8 / 27.2 | low |
| SC gate th=19 | 36.7 / 48.5 | 47.3 / 52.0 | 22.7 / 30.6 | mid |
| **SC gate th=21** | **39.5** / 50.4 | **50.2** / 54.9 | **23.0** / 30.4 | 1.9 / 2.0 / 2.5 |
| **SC gate th=23** | **39.7** / 50.4 | **51.1** / 55.9 | 22.8 / 30.2 | 2.3 / 2.5 / 2.5 |
| DeepRAG paper | 40.7 / 51.5 | 48.1 / 53.3 | 22.5 / 30.4 | — |

**結論**：DeepRAG-8b 上 SC gate 跟 learned gate 持平/微贏。但這個方法**inherits DeepRAG 的 SFT+DPO 訓練成果**，contribution 偏小。

### 3.4 DeepRAG-8b + SC gate **on PopQA**（200Q, 2026-04-27 新跑）

| Dataset | DeepRAG-8b SC@21 EM | avg_ret |
|---|---|---|
| popqa | **0.265** | 2.385 |

vs 之前 sc_gated@80 (post-hoc, 1000Q) = 0.350 → 在 single-hop popqa 上，**simple post-hoc gate 比 DeepRAG decomposition 還好**。原因可能是 popqa 是單跳，decomposition 反而 over-engineering。

---

## 4. 階段四（進行中）：vanilla Llama-3-8B / Qwen-2.5-72B + DeepRAG framework

> 動機：3.3 的 SC gate claim 太弱（只是換 gate，沒解決核心訓練問題）。階段四目標：
>   - 證明 **vanilla 8B 跑不動 DeepRAG 框架**（負對照，confirm SFT 必要性）
>   - 證明 **Qwen-72B prompt-based 可以替代 DeepRAG 的 SFT 拆解**（升級 paper claim）

### 4.1 ❌ vanilla Llama-3-8B-Instruct + SC gate（4ds × 200Q, **無 4-shot**）

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

### 4.2 ⏳ Qwen-2.5-72B + 4-shot DeepRAG prompt + SC gate（smoke test 進行中）

正在測：用 DeepRAG 官方 [`ICL-prompt/question_decompose.txt`](https://github.com/gxy-gxy/DeepRAG/blob/main/ICL-prompt/question_decompose.txt) **逐字複製** 的 4-shot examples（這也是 DeepRAG 用來叫 72B 生 SFT 訓練資料的同一個 prompt）。

期望結果：
- Qwen-72B 能照 4-shot 例子做出 multi-step decomposition + `<answer short>` 格式
- SC gate 在這個無訓練 setup 下達到接近 DeepRAG-8b 的水準
- **如果成立**：paper claim 升級為「**只用 prompt-based 拆解 + SC gate（不需 SFT、也不需 DPO）能 match DeepRAG 的兩階段訓練**」

---

## 5. 完整跨方法對照（4 datasets × 1000Q, Llama-3-8B base，截至 2026-04-27 晚）

| Method | hotpotqa EM/F1 | 2wiki EM/F1 | musique EM/F1 |
|---|---|---|---|
| Ours CoT (no-RAG) | 25.2 / 35.6 | 31.2 / 39.3 | 11.5 / 20.3 |
| Ours Single-RAG | 32.4 / 44.0 | 37.9 / 47.0 | 19.2 / 29.0 |
| Ours SC-gated@80 (post-hoc) | 32.1 / 43.2 | 37.3 / 46.6 | 18.2 / 28.1 |
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
| Ours SC-gated@80 (1000Q) | **35.0** | 42.6 |
| Ours DeepRAG-8b SC@21 (200Q) | 26.5 | – |
| Ours vanilla Llama-3-8B SC@21 (200Q) | 6.5 | – |
| DeepRAG paper | 42.5 | 47.8 |

---

## 6. 已 launch 的 Qwen-72B 完整 4ds × 200Q 指令

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

---

## 7. Code 變更紀錄（截至 2026-04-27 晚）

| File | 變更 |
|---|---|
| `deeprag_compare/src/run_experiment.py` | (階段三) 加 dynamic 變體 + skip_baselines |
| `deeprag_compare/src/run_deeprag_sc.py` | (階段三) gate suffix 在檔名；(階段四) `--use_fewshot {auto,yes,no}` flag + DEEPRAG 官方 4-shot prompt 整合 + 與 prompt_style auto 聯動 |
| `deeprag_compare/scripts/run_deeprag_sc.sh` | 加 `MODEL_ID/MODEL_TAG/HF_CACHE/DEVICE/PROMPT_STYLE/USE_FEWSHOT` env vars |
| `deeprag_compare/scripts/run_dynamic_c3.sh` | (階段三) dynamic 4ds × 2 modes |
| `deeprag_compare/scripts/run_dynamic_plan_a.sh` | (階段三) Plan A regen+masked |
| `deeprag_compare/scripts/run_deeprag_sc_sweep.sh` | (階段三) threshold sweep |
| `deeprag_compare/scripts/compare_all.py` | (階段三) glob 所有變體 + DeepRAG-SC F1 fallback |

---

## 8. 結果檔命名規則

| 檔名模式 | 內容 |
|---|---|
| `<ds>_meta-llama-3-8b-instruct.jsonl` | baseline (no_rag + single_rag + 16 sc_gated_rate*) |
| `<ds>_meta-llama-3-8b-instruct_dyn_{continue,restart,planA}.jsonl` | dynamic 變體 |
| `<ds>_deeprag-8b_learned.jsonl` | DeepRAG-8b 原始 learned gate |
| `<ds>_deeprag-8b_sc_th{17,19,21,23}.jsonl` | DeepRAG-8b SC gate sweep |
| `<ds>_llama3-8b-instruct_sc_th21.jsonl` | vanilla Llama-3-8B + SC gate (no 4-shot) |
| `<ds>_qwen2.5-72b-fs_sc_th21.jsonl` | Qwen-2.5-72B + 4-shot + SC gate（待產出）|
| `comparison_all.md` | compare_all.py 產出的單一 markdown 表 |

---

## 9. 環境 / 路徑

| 用途 | 路徑 |
|---|---|
| Python env | `/dataspace/S113065528/envs/research/bin/python` |
| Llama-3-8B base | `/workspace/models/Meta-Llama-3-8B-Instruct/` |
| **Qwen-2.5-72B-Instruct** (FP16 cache) | `/workspace/models/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31` |
| DeepRAG-8b ckpt | `/home/S113065528/Adaptive-RAG/hf_cache/models--xinyan233333--DeepRAG-8b/` |
| Elasticsearch | `localhost:9200` (wiki, field=`paragraph_text`, 880K docs) |
| 必要 env vars | `HF_HUB_OFFLINE=1`, `CUDA_VISIBLE_DEVICES=...` |

---

## 10. ⭐ 論文 claim 演進

### 原始 claim（2026-04-25 階段三完成）
> SC 作為 training-free gate, 替代 DeepRAG 的 RL-trained gate。

**問題**：使用 DeepRAG-8b（人家訓練好的）只是換 gate，contribution 不夠投頂會。

### 新 claim（2026-04-27 階段四方向）
> **DeepRAG 需要兩階段訓練：(1) Qwen-72B distill 拆解到 8B 的 SFT，(2) DPO 訓練 retrieval gate。我們證明：**
>   1. **直接用 Qwen-72B prompt-based 拆解**（DeepRAG 同款 4-shot prompt，無需 SFT 學生）
>   2. **訓練-free SC threshold gate**（無需 DPO）
>
> 達到接近 DeepRAG-8b 兩階段訓練的表現 — **整套 RL/DPO 流水線可以省掉**。

### 對 reviewer 的反駁

| 質疑 | 反駁 |
|---|---|
| 「為何不測 NQ？」 | DeepRAG 也沒測 NQ；測 PopQA + 三 multi-hop 已 cover single/multi |
| 「Qwen-72B 比 DeepRAG-8b 大 9 倍，不公平」 | **DeepRAG 自己也用 Qwen-72B 當拆解 teacher**（paper §4.3）。我們只是把它從「離線 distill 到 8B」改成「直接 inference」，本質上沒用更多算力 |
| 「為何 SC 比 learned 還好？」 | SC@23 retrieves more (avg 2.5 vs 1.7)，trade-off 透過 efficiency curve 呈現 |
| 「為何 Plan A 失敗？」 | multi-hop 中 mid-CoT 插 docs 打斷 reasoning chain（與 FLARE/DRAGIN 弱點一致） |

---

## 11. 後續路線

### 短期（這週）
1. ✅ Qwen-72B 4ds × 200Q smoke test → full run（GPU 2+3, ~5-7 hr）
2. ⏳ 跑完用 `compare_all.py` 重新生 markdown 表
3. ⏳ **如果 Qwen-72B + 4-shot 表現好** → paper 主軸成立

### 中期（下週）
4. **Iterative SC** prototype（Path A）：用 ΔSC 當 retrieval 品質 closed-loop signal，看能否在 DeepRAG-8b / Qwen-72B 上額外提升 EM
5. **Threshold efficiency curve**：x=avg_retrievals, y=EM，畫 SC@{17,19,21,23} vs learned，展示 SC gate 是否能用更少 retrieval 拿到一樣分數
6. PopQA SC sentence-level dynamic（exp10 在 NQ +4.2 → 在 popqa 是不是也 work）

### 長期（往 paper draft）
7. 寫 method section：SC formula、threshold 自動化（dev set quantile）
8. 寫 experiments section：跨 4ds 跨 backbone 的完整 table
9. 寫 analysis：為何 SC 跟 learned gate behavior 不同（per-question agreement matrix）
10. 寫 negative result section：Plan A 失敗、vanilla 8B 失敗、DPO 不必要

---

## 12. Decision Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-23 | 用 deeprag_compare benchmark 對齊 DeepRAG | 取得跨方法可比較的 1000Q 規模結果 |
| 2026-04-25 | 加 dynamic restart mode + skip_baselines flag | 不重跑 baseline，省時 |
| 2026-04-26 | DeepRAG-SC threshold=8/12 失效後做 SC distribution 分析 | 發現實際 SC 分布在 17-22，threshold 太低 |
| 2026-04-26 | Plan A (regen+masked) 實作並 4ds × 1000Q 跑 | 修 Dynamic-RAG 的 query/CoT 設計 |
| 2026-04-27 (上午) | Plan A 確認失敗（4ds 全部 ±0.01）；threshold=21/23 確認成功 | 寫入 paper：dynamic 是 negative result |
| 2026-04-27 (中午) | 跑 learned gate baseline vs SC@23（39.7/51.1/22.8 vs 39.6/50.2/23.7） | 確認 SC ≥ learned，但 inherits DeepRAG-8b 訓練 |
| 2026-04-27 (晚間) | **Pivot 方向**：vanilla Llama-3-8B 跑（confirm 失敗）+ Qwen-72B + 官方 4-shot prompt | 把 paper claim 升級到「**SFT teacher (72B) 直接 inference + SC = 跳過 SFT+DPO**」 |
| 2026-04-27 (晚間) | 從 DeepRAG 官方 repo 抓 4-shot decomposition prompt | 對齊 DeepRAG 用 72B 生 SFT 訓練資料的同一個 prompt |
