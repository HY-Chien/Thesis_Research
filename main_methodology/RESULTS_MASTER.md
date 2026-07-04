# SEEK — 實驗結果總表(Master)

> **單一彙整檔。** 整併自(已移除的)`RESULTS_SEEK.md`、`RESULTS.md`、
> `results/{deeprag_same_pool_comparison, wiki21M_performance_memory,
> wiki21M_router_round2, wiki21M_sweep_round1, wiki21M_verifier_v1_summary,
> wiki21M_verifier_v2_entropy_aware_summary, wiki21M_oracle_direct_decompose,
> wiki21M_oracle_direct_decompose_full, decompose_vs_direct_2x2, recall_ab_m3}.md`。
> 原始 JSONL 結果仍在 `results/`,本檔只整併文字彙整。
>
> 最後更新:2026-07-03。2026-07-03 新增跨資料集 decomposition failure taxonomy
> (`results/decomp_failure_taxonomy.md`)與 PopQA confidence-popularity bias 分析
> (`experiment/maxent_correctness/confidence_popularity_bias.md`);
> 2026-07-01 新增 efficiency 成本框架:
> SEEK 的主要效率主張改為 training-free adaptive routing,不宣稱推論檢索次數一定少於 DeepRAG；
> 2026-06-27 整理 6/25 thesis presentation 的最新敘事記憶；
> 2026-06-23 新增 HotpotQA failure analysis、conservative date-comparison synthesis guard、original-question evidence union ablation；
> 主表 headline JSONL 尚未覆寫者維持原 canonical。
> 前置階段(SC-gated / confident-hallucination,
> `Research/EXPERIMENTS_SUMMARY.md`、`deeprag_compare/`、`experiment/`)是**另一個主題**,
> 不在本檔範圍。資料集說明見 `DATASETS.md`。

---

## 0. Canonical 摘要 / 最終選定法

**最終選定法(2026-06-17)= training-free entropy-aware verifier v2(no evidence)+ path-aware title 檢索**。

```text
Base model        : Llama-3-8B-Instruct
Adapter           : checkpoints/llama3-8b-decomposer-musique-2wiki-popqa  (12K three-source: MuSiQue+2Wiki+PopQA)
Inference mode    : Mode C(decompose & answer 都用 LoRA),synthesis LoRA-off(synthOFF)
Synthesis guard   : value→entity guard + conservative date-comparison guard
                    (2026-06-23 offline replay;正式 JSONL headline 待重算/套用)
Gate              : token max-entropy, τ=0.3
Verifier          : query-level, fallback=auto, direct_entropy_tau=0.4, decomp_post_entropy_tau=0.8
Evidence support  : 關閉(只當 ablation,見 §3.2)
Retrieval(honest): wiki-21M BM25(title+text,對齊 DeepRAG), top_k=8;PopQA top_k=3
                    PATH-AWARE title:decompose 短 sub-q 開 title、direct 長原問題關 title
                    (單一 dataset-agnostic 規則,見 §4.4)
```

最終選定法 @ honest wiki-21M(§3.1 path-aware 表):

| Dataset | path-aware(final)EM/F1 | DeepRAG paper | ΔEM | verdict |
|---|---:|---:|---:|---|
| PopQA   | 51.8 / 54.2 | 42.5 / 47.8 | **+9.3** | win |
| MuSiQue | 24.6 / 33.5 | 22.5 / 30.4 | **+2.1** | win |
| 2Wiki   | 46.9 / 53.5 | 48.1 / 53.3 | −1.2 | close(F1 **+0.2 贏**) |
| HotpotQA| 31.8 / 41.2 | 40.7 / 51.5 | −8.9 | lose(檢索本質限制,§4.4) |

**一句話**:在 DeepRAG 自己的 21M pool、對齊其 title+text 檢索下,單一統一配置(training-free
verifier + path-aware title)—— **PopQA / MuSiQue 兩勝、2Wiki 近平(F1 勝、EM −1.2)、HotpotQA 仍落後
(−8.9,M3 §4.4 證明為 BM25 第二跳 lexical 上限,非方法問題)**。比 2026-06-11 舊 canonical(45.2/22.5
title-off)再進一步:MuSiQue 由平轉勝、2Wiki gap −2.9→−1.2。distractor pool 的亮眼數字(§2)是另一個
protocol,不可與 honest pool 直接相比。

**2026-06-23 incremental update(不覆蓋主表):** 新增 HotpotQA failure analysis
`results/hotpotqa_failure_analysis.md`。結論:HotpotQA 與 2Wiki/MuSiQue 不同, direct-only
(151) > decomp-only(98), verifier 只抓到 77/151 direct-only,而 599 題 two-path both-wrong
仍是最大瓶頸。已在 `seek/pipeline.py::_rescue_value_answer` 與
`scripts/rescue_value_answers.py` 加入 conservative date-comparison guard,修正「step date 已足夠但
synthesis 選錯 entity」的乾淨錯誤。既有 path-aware trace 的 offline replay 估計:
HotpotQA **31.8/41.2 → 32.1/41.7**, 2Wiki **46.9/53.5 → 47.8/54.4**,
MuSiQue/PopQA 不變。此為小幅 synthesis fix。

**2026-07-03 incremental update(offline replay,不改主表):** 新增
`results/decomp_failure_taxonomy.md`(`scripts/analyze_decomp_failure_taxonomy.py`),
把 HotpotQA failure analysis 的 2x2 + coverage cross-tab + 失敗分類泛化到
2Wiki/MuSiQue(HotpotQA 重放數字與原分析完全一致,自洽)。跨資料集結論:
(1) decomposition-path 錯誤的最大宗都是 `retrieval_gap_both`(兩路 full-coverage@10
皆未達成):HotpotQA 47% / 2Wiki 64% / MuSiQue 68%——瓶頸是 BM25 檢索,非 decomposer;
(2) `constraint_loss`(sub-q 丟失原題 disambiguating 限制)幾乎是 HotpotQA 專屬
(10% vs 2Wiki 1% / MuSiQue 2%),對應 evidence union 對 HotpotQA 最有效(+5.5 EM);
(3) `synthesis_failure`(step 已含 gold 但 synthesis 選錯)跨三資料集穩定 15–18%,
是 synthesis guard 的目標質量;(4) verifier 結構性偏 decompose:decomp-only 抓到
91–95%、direct-only 只 24–51%(2Wiki/MuSiQue 是正確取捨,HotpotQA 相反)。
另新增 PopQA confidence-popularity bias 分析(`experiment/maxent_correctness/
confidence_popularity_bias.md`):信心值與 entity popularity 無相關(bias 假說否證,
Spearman |rho|<0.05 n.s.,wrong-only 子集亦然),AUROC(ent→wrong) 跨 popularity 分層
穩定 0.72–0.75;confident hallucination 的本質是 deterministic wrong path,
非 popularity bias;唯一 popularity-skewed 的是 metric artifact(熱門實體的
zero-entropy「錯誤」28% 可被 alias 救回,稀有實體 0%)。

**2026-06-23 original-question evidence union(正式 1000Q,四資料集全跑；修正版 ablation 口徑):**
每個 decomposed sub-question 的 evidence = sub-q top-k ∪ original-question top-k(去重),
保留原題 disambiguating clues。這個設計只改 **decomposition path 的 step context**；direct path
本身沒有 union。因此 union 的乾淨 ablation 不應把 final verifier 選 direct 的題一起混進同一張表,
而應比較同一批題目的 **decomposed candidate**:path-aware baseline vs `--direct-context-union`。

| Dataset | decomp w/o union EM/F1 | decomp + union EM/F1 | ΔEM/F1 | Interpretation |
|---------|-------------------------:|-----------------------:|-------:|---|
| HotpotQA| 25.00 / 33.42 | **30.50 / 39.49** | **+5.50 / +6.07** | 原題 comparison / entity clues 對 sub-step 最有幫助 |
| 2Wiki   | 47.20 / 53.63 | **50.20 / 56.65** | **+3.00 / +3.03** | sub-q title retrieval + 原題約束互補 |
| MuSiQue | 24.70 / 33.10 | **25.50 / 33.40** | +0.80 / +0.30 | 小幅補 context loss |
| PopQA   | 51.50 / 53.68 | **52.90 / 55.20** | +1.40 / +1.52 | 單跳題也不傷 decomp candidate |
| **AVG** | **37.10 / 43.46** | **39.78 / 46.19** | **+2.68 / +2.73** | union 的 path-local 效果 |

**答辯/論文口徑:** 這不是「拿 direct path 的答案去幫 decompose path」,而是
**original-question evidence preservation**。上路 direct candidate 仍是「原問題→直接答案」;
下路 decompose candidate 仍是「原問題→sub-questions→逐步回答→synthesis」。兩路會共享
`r(q)` 取回的原問題文章,但下路另外有 sub-question title-on retrieval、step answers 與 final
synthesis,所以答案生成路徑仍不同；共享的是證據來源,不是 decision path 或 final answer。

Full-system 成績可作為 final candidate 結果,但不要拿來當 union 的唯一 ablation 證據:

| Dataset | path-aware verifier | + union verifier | + union + type-aware final |
|---------|--------------------:|-----------------:|---------------------------:|
| HotpotQA| 31.8 / 41.2 | 34.3 / 44.3 | **35.5 / 46.1** |
| 2Wiki   | 46.9 / 53.5 | 49.9 / 56.4 | **50.0 / 56.6** |
| MuSiQue | 24.6 / 33.5 | 24.8 / 32.9 | **24.7 / 33.1** |
| PopQA   | 51.8 / 54.2 | 52.0 / 54.8 | **51.9 / 54.7** |
| **AVG** | **38.8 / 45.6** | **40.2 / 47.1** | **40.5 / 47.6** |

Final candidate 的 verifier 實際選路比例(老師建議補的「上路/下路比例」):

| Dataset | direct path selected | decompose path selected |
|---------|---------------------:|------------------------:|
| HotpotQA| 274 / 1000 = **27.4%** | 726 / 1000 = **72.6%** |
| 2Wiki   | 65 / 1000 = **6.5%** | 935 / 1000 = **93.5%** |
| MuSiQue | 182 / 1000 = **18.2%** | 818 / 1000 = **81.8%** |
| PopQA   | 37 / 1000 = **3.7%** | 963 / 1000 = **96.3%** |
| **Pooled** | 558 / 4000 = **14.0%** | 3442 / 4000 = **86.1%** |

**Slide 44 修正版(針對截圖 `Original-question Evidence Union Ablation`):** 這頁現在的表是
full-system before/after,會混入 verifier 最後選 direct 的題,因此不適合作為 union ablation 主證據。
建議改成以下版面:

- Goal: Test whether the decomposition path should preserve original-question evidence.
- Evidence union: for each sub-question, answer with `sub-question top-k ∪ original-question top-k`.
- Main table title: **Decomposition-path effect** 或 **Decomposed candidate only**。
- 主表放 decomp candidate EM/F1:HotpotQA `25.00/33.42→30.50/39.49(+5.50/+6.07)`,
  2Wiki `47.20/53.63→50.20/56.65(+3.00/+3.03)`,
  MuSiQue `24.70/33.10→25.50/33.40(+0.80/+0.30)`,
  PopQA `51.50/53.68→52.90/55.20(+1.40/+1.52)`,
  AVG `37.10/43.46→39.78/46.19(+2.68/+2.73)`。
- 若想保留原本截圖的 `31.8→34.3 / 46.9→49.9` 等數字,放在右下角小字:
  "Full-system verifier score also improves, but it includes final path selection and is not the clean
  union ablation." 不要把舊 full-system F1 誤放進主 ablation 表。

檔:`results/{hotpotqa,2wiki,musique,popqa}_1000_verify_v2_pathaware_directctx.jsonl`,
`results/*_pathaware_directctx_typeaware_replay.jsonl`;跑法:
`scripts/run_{ds}_union_ablation.sh`(`run_seek_dual.py --direct-context-union`)。

**2026-06-23 type-aware verifier offline replay:**針對 `auto_fallback_decomp`
過度偏向 decompose 的問題,新增 `scripts/replay_typeaware_verifier.py` 做不重跑 LLM 的
direct/decomp 選擇重評估。保守規則(只在 answer type、direct entropy、decomp post-entropy
與 entity-fragment 訊號足夠時改選 direct)在 original-question evidence union trace 上:
HotpotQA **34.3/44.3 → 35.5/46.1**(+1.2 EM,+1.9 F1),2Wiki
**49.9/56.4 → 50.0/56.6**(+0.1 EM,+0.2 F1),MuSiQue
**24.8/32.9 → 24.7/33.1**(-0.1 EM,+0.2 F1),PopQA
**52.0/54.8 → 51.9/54.7**(-0.1 EM,-0.1 F1)。final candidate 平均
**40.5/47.6**,高於 DeepRAG 平均 **38.5/45.8**。檔:`results/typeaware_verifier_replay.md`、`results/final_performance_table.md`。

> ⚠️ **兩個 pool 不可混表**:distractor(per-question 10–20 paras,gold 必在候選裡;
> Adaptive-RAG/IRCoT 協定)會把多跳 EM 灌水 +16~24;honest wiki-21M(psgs_w100, 21M)
> 才是 DeepRAG 的真實設定。論文若要宣稱「贏 DeepRAG 多跳」必須用 honest pool 並分表。

---

## 1. Decomposer 內在品質(intrinsic eval)

來源:`RESULTS.md`(2026-05-21)。評的是 decomposer 輸出的結構正確性,非端到端 QA。

### 1.1 v1(MuSiQue gold 4000 筆 LoRA)— `checkpoints/llama3-8b-decomposer-musique/`

MuSiQue dev(2417):parse_ok **100.0%** · deps_valid **100.0%** · count_match **82.2%** · topo_match **95.4%**。

| hop | n | count% | topo% |
|---|---:|---:|---:|
| 2hop | 1252 | 97.6 | 99.8 |
| 3hop1 | 568 | 71.8 | 96.1 |
| 3hop2 | 192 | 80.2 | 72.1 |
| 4hop1 | 246 | **46.3** | 98.2 |
| 4hop2 | 64 | **40.6** | 76.9 |
| 4hop3 | 95 | 65.3 | 64.5 |

格式零後處理(100% 合法 `#N`);弱點 = 深層題步數崩塌(4hop 只 40–46% count,自然分佈 4hop 僅 ~6%)。明顯優於原始 OSS-20B(count 82.2 vs 66.1)。

### 1.2 v2(MuSiQue+2Wiki 8000 筆)— `checkpoints/llama3-8b-decomposer-musique-2wiki/`

| dev | parse | deps | count | topo |
|---|---|---|---|---|
| MuSiQue (2417) | 100.0 | 99.9 | 81.6 | 94.3 |
| 2Wiki (12576) | 100.0 | 100.0 | **98.7** | 92.5 |

合併 2Wiki 後 MuSiQue 幾乎無損(−0.6/−1.1,雜訊內)。2Wiki count 98.7(結構規律)。弱點同 v1(4hop 步數崩塌;2Wiki bridge_comparison 4 步雙鏈拓樸偶爾連錯)。

### 1.3 對照基準 GPT-OSS-20B(MuSiQue train 全量)

parse_rate 93.5% · count_match 66.1%(of parsed)· step_answer EM 65.1% · psi_acc 78.4%。

> 註:最終 canonical decomposer 是 **12K three-source**(加 PopQA),用於所有端到端結果。

---

## 2. 端到端 QA — distractor pool(per-question candidate paragraphs)

來源:`RESULTS_SEEK.md`(2026-05-25)。Retrieval = per-question ~20 paras BM25(Adaptive-RAG/IRCoT 協定,gold 通常在候選裡)。

> ⚠️ **本節全部是 distractor pool 數字,與 DeepRAG 不是同一個 pool,不可並列比較。**
> DeepRAG 跑 global wiki(psgs_w100, 21M);此處 gold 必在 10–20 候選裡,會把多跳 EM
> 灌水 +16~24。**與 DeepRAG 的對外比較一律只看 §3(honest wiki-21M 同 pool)。**
> 本節保留作為 SEEK 自身的開發/消融紀錄(pipeline、失敗診斷、τ/mode、per-hop)。

### 2.1 評估設定(distractor pool;為何不與 DeepRAG 並列)

- DeepRAG `inference/run.sh` 用 `--sample 1000` 無 `--shuffle` → 取 dev **前 1000 題**(`src/main.py:178-183` 確認)。
- MuSiQue dev 按 hop 排序,前 1252 全是 2hop → DeepRAG MuSiQue 22.5 EM 是**全 2hop 子集**,不是 full dev。對齊比較須用同「dev 前 1000」protocol。
- Dev sizes:MuSiQue 2417 / 2Wiki 12576 / HotpotQA 7405。
- Corpus 差異:DeepRAG = global Wikipedia(DPR dump);我方 distractor = per-question 20 候選(命中率天花板較高,偏向我方)。

### 2.2 主結果(distractor pool;dev 前 1000,mode C,max_ent τ=0.3,+ synthesize prompt fix + option C)

> distractor pool,非 DeepRAG 同 pool（DeepRAG 欄已移除;對 DeepRAG 比較見 §3）。

| Dataset | n | SEEK EM/F1(distractor) |
|---|---:|---:|
| **MuSiQue** | 1000 | **48.0 / 57.1** |
| **2Wiki** | 1000 | **45.0 / 53.7** |
| **HotpotQA** | 1000 | **39.6 / 52.4** |
| **PopQA (+option C)** | 1000 | 30.3 / 40.2 |

亮點(distractor 內部):synthesize prompt fix 後三個多跳大幅提升;PopQA 為結構性短板。這些數字因 gold 必在候選裡而偏高,**不代表對 DeepRAG 的勝負**。

兩項技術改進(無重訓):
1. **Synthesize prompt fix**(只改 `_synthesize` prompt:comparison 回 entity、yes/no 看一致性、加 2 few-shot)→ 2Wiki +9.1 EM、MuSiQue +2.1、HotpotQA +1.8、PopQA 無影響。free win、零 regression。
2. **Option C entropy-probe skip-decompose**(`skip_decompose_tau=2.0`)→ PopQA 17.9→30.3(+12.4,skip rate 79.7%)。**但多跳會崩(見 §5 限制),最終 distractor headline 的 PopQA 取無 C 版 17.9**。

改進前後(distractor):

| Dataset | 原始 SEEK | +synth fix | +synth fix +option C |
|---|---:|---:|---:|
| MuSiQue | 45.9 | 48.0 | — |
| 2Wiki | 35.9 | 45.0 | — |
| HotpotQA | 37.8 | 39.6 | — |
| PopQA | 17.9 | 17.9 | **30.3** |

### 2.3 2Wiki per-type 細分 + bridge_comparison 診斷(τ=0.3,synth fix 前)

| type | n | EM | 備註 |
|---|---:|---:|---|
| inference | 128 | **58.6** | 強項 |
| compositional | 428 | 46.3 | 主流 |
| comparison | 216 | 29.6 | yes/no norm 問題 |
| bridge_comparison | 228 | **9.6** | ← 拖垮平均 |
| TOTAL | 1000 | 35.9 | |

**根因(實證)**:不是訓練量、不是 decomposition 錯、不是 retrieval 錯,而是 **synthesize 步驟壞掉**。反證:inference 訓練只佔 2.6% 卻 58.6 EM;bridge_comparison 訓練佔 20.7% 卻 9.6。典型 fail:decompose 正確 4 步、每步 sub-answer 對,但 synthesize 輸出 date 字串或 step 格式(`X >> date of birth`),沒做「比兩年份→選片名」的 cross-step reasoning。→ 由 §2.2 的 synthesize prompt fix 修補(2Wiki +9.1)。

### 2.4 gate τ 對照 + Mode B vs C(MuSiQue 200,早期)

τ=0.0 vs 0.3:三個多跳都省 7–9% retrieval、EM 損失 ≤0.5。MuSiQue 200 題 τ 全掃描甜蜜點 = τ=0.3(retrieval 90.8%、EM 47.0);80% 觸發率以下急速崩塌。

Mode:B(decompose LoRA、answer base)35.0/47.6 vs **C(都 LoRA)40.5/53.3** → C 勝 5.5 EM,**後續全用 mode C**。

### 2.5 SEEK vs 我方自跑 baselines(同一 distractor pool 內部對照,1000Q,synth fix,no option C)

> ⚠️ **全部在同一個 distractor pool**(SEEK 與下列 baselines 皆我方自跑,可同表比較)。
> **DeepRAG (paper) 列已移除——那是 global wiki 21M pool,不同基準,不可同表;對 DeepRAG 比較見 §3。**

| Method | HotpotQA | 2Wiki | MuSiQue | PopQA |
|---|---|---|---|---|
| Ours CoT (no-RAG) | 25.2/35.6 | 31.2/39.3 | 11.5/20.3 | 20.4/23.1 |
| Ours Single-RAG | 32.4/44.0 | 37.9/47.0 | 19.2/29.0 | 41.6/51.3 |
| Ours `neg_max_ent@-0.80` | 32.7/44.3 | 36.2/45.4 | 19.0/29.1 | 42.4/51.6 |
| Ours DeepRAG-8b + SC(我方重跑) | 39.7/50.4 | **51.1/55.9** | 22.8/30.2 | – |
| **SEEK (本研究)** | **39.6/52.4** | **45.0/53.7** | **48.0/57.1** | 17.9/29.9 |

> distractor 內部解讀:SEEK 在 MuSiQue 大幅領先自跑 baselines;PopQA 為結構性短板。此表不含 DeepRAG，不可推論對 DeepRAG 勝負。

### 2.6 MuSiQue full dev 各 hop(2417,τ=0.3)

| hop | n | EM |
|---|---:|---:|
| 2hop | 1252 | **43.0** |
| 3hop1 | 568 | 36.6 |
| 3hop2 | 192 | 15.1 |
| 4hop1 | 246 | 25.6 |
| 4hop2 | 64 | 14.1 |
| 4hop3 | 95 | 9.5 |
| **TOTAL** | **2417** | **35.4 / 47.5** |

前 200/1000 全 2hop → 對齊 protocol 的 48.0 EM 高估了絕對難度;full dev 35.4(distractor pool)是更嚴格的數字,深 hop 是主短板(對應 §1 decomposer 4hop count 崩塌)。(此為 distractor pool;與 DeepRAG 的比較見 §3。)

---

## 3. 端到端 QA — honest wiki-21M pool(DeepRAG 同 pool,canonical)

來源:`deeprag_same_pool_comparison.md` + `wiki21M_*.md`。Pool = 共享 ES `wiki`(psgs_w100, 21M)= DeepRAG 真實 corpus。12K 三源 decomposer、mode C、max_ent τ=0.3、synthOFF。dev 前 1000 無 shuffle。所有數字已套用 2026-06-11 synthesis value→entity guard(`seek/pipeline.py::_rescue_value_answer`,離線 `scripts/rescue_value_answers.py`;原值留 `pred_orig`/`em_orig`)。

### 3.1 Best unified config = entropy-aware verifier v2(final operating point)

verifier 以 per-query 路由(single-step vs decompose),threshold `fallback=auto, direct_entropy_tau=0.4, decomp_post_entropy_tau=0.8`,no evidence。PopQA top-3,其餘 k=8。

| Dataset | all-decompose k8(no verifier) | **verifier v2(final)** | DeepRAG | ΔEM | ΔF1 | verdict |
|---|---|---|---|---:|---:|---|
| PopQA   | 50.7/52.9 | **50.7/52.9** | 42.5/47.8 | +8.2 | +5.1 | win |
| MuSiQue | 22.7/30.5 | **22.5/30.5** | 22.5/30.4 | +0.0 | +0.1 | ≈ tie |
| 2Wiki   | 45.6/51.5 | **45.2/51.3** | 48.1/53.3 | −2.9 | −2.0 | close |
| HotpotQA| 25.7/33.9 | **31.9/41.2** | 40.7/51.5 | −8.8 | −10.3 | lose(gap −15.0→−8.8) |

**verifier 淨效果**:+6.2 EM HotpotQA(per-query single-step 路由)、其餘三個 ≤0.4 EM 代價。verifier default v2 = HotpotQA 31.7/41.0、2Wiki 45.5/51.5、MuSiQue 22.4/30.4、PopQA 50.7/52.9;tuned 為上表。

**上表為 title-off(legacy handicap,§4.4 發現)。套 PATH-AWARE title(decompose 短 sub-q 開 title、direct 長原問題關 title;單一 dataset-agnostic 規則)後 → 新的 final(2026-06-17):**

| Dataset | verifier v2 title-off | 全域 title-on | **path-aware(final)** | DeepRAG | ΔEM | ΔF1 |
|---|---|---|---|---|---:|---:|
| 2Wiki   | 45.2/51.3 | 45.9/52.5 | **46.9/53.5** | 48.1/53.3 | −1.2 | **+0.2(F1 贏)** |
| HotpotQA| 31.9/41.2 | 30.5/39.9 | **31.8/41.2** | 40.7/51.5 | −8.9 | −10.3 |

**相容實驗完成(PopQA top-3 + MuSiQue k8,2026-06-17)→ path-aware 四資料集全數不倒退,3/4 還上升。新全域 canonical:**

| Dataset | 舊 canonical(title-off) | **path-aware(new final)** | DeepRAG | ΔEM | ΔF1 | verdict |
|---|---|---|---|---:|---:|---|
| PopQA   | 50.7/52.9 | **51.8/54.2** | 42.5/47.8 | +9.3 | +6.4 | win(更大) |
| MuSiQue | 22.4/30.4 | **24.6/33.5** | 22.5/30.4 | +2.1 | +3.1 | **win(原為 tie)** |
| 2Wiki   | 45.2/51.3 | **46.9/53.5** | 48.1/53.3 | −1.2 | **+0.2(F1 贏)** | close |
| HotpotQA| 31.9/41.2 | **31.8/41.2** | 40.7/51.5 | −8.9 | −10.3 | lose(檢索本質限制,§4.4) |

per-path EM(verifier 內部):PopQA direct 37.1/decomp 51.5/verifier 51.8/oracle 55.4(decomp 945+direct 55);MuSiQue direct 10.2/decomp 24.7/verifier 24.6/oracle 28.4;2Wiki direct 18.0/decomp 47.2/verifier 46.9/oracle **52.7(>DeepRAG)**;HotpotQA direct 30.3/decomp 25.0/verifier 31.8/oracle 40.1。**PopQA 沒被「direct 關 title」傷到,因為 verifier 945/1000 路由到 decompose(sub-q entity-centric→title-on 受益);direct 只用 55 次。**

**淨效果**:path-aware 是單一 dataset-agnostic 規則(看 query 是 sub-q 還是原問題,與資料集無關 → 合法統一配置,非 per-dataset cherry-pick),且對齊 DeepRAG 的 title+text 檢索。score card:PopQA/MuSiQue 兩明確勝、2Wiki 近平(F1 勝、EM −1.2)、HotpotQA 仍負(−8.9,M3 §4.4 證明是 BM25 第二跳 lexical 上限,非 gate/answering)。檔:`results/{ds}_1000_verify_v2_pathaware.jsonl`。**注意 §3.1 主表上方仍列 title-off 為歷史對照;此表為現行 final。**

### 4.5 Title-handling ablation(2026-06-20,支撐 path-aware 的設計選擇)

完整 3×4 ablation:同一條 pipeline、同 verifier 設定(no-evidence, direct_τ=0.4, decomp_τ=0.8, fallback auto, k8/PopQA k3),**只差 `--title-mode`**(`scripts/run_seek_dual.py --title-mode {off,global,pathaware}`,實作於 `seek/pipeline.py` `title_direct`/`title_subq`)。fresh title-off 重現舊 canonical → 整組自洽。

| Dataset | (a) title-off | (b) global title-on | (c) **path-aware** | DeepRAG |
|---|---|---|---|---|
| PopQA   | 51.2/53.5 | 51.5/53.9 | **51.8/54.2** | 42.5/47.8 |
| MuSiQue | 22.5/30.5 | 24.5/33.1 | **24.6/33.5** | 22.5/30.4 |
| 2Wiki   | 45.2/51.3 | 45.9/52.5 | **46.9/53.5** | 48.1/53.3 |
| HotpotQA| 31.9/41.2 | 30.5/39.9 | **31.8/41.2** | 40.7/51.5 |

**path-aware 在四個資料集都 ≥ 另外兩種(weakly Pareto-dominant)。** 兩個方向:

- **開 title 的價值((a)→(c))**:2Wiki **+1.7**、MuSiQue **+2.1**、PopQA +0.6、HotpotQA −0.1。title 比對對 decompose 的 entity-centric sub-q 有實質 recall 助益。
- **不做 path-aware 的代價((b) global vs (c))**:HotpotQA **−1.3/−1.3**、2Wiki **−1.0/−1.0**、PopQA −0.3、MuSiQue −0.1。global「長 query 也開 title」會把 stray title hit 灌進原問題檢索;path-aware 對長原問題關 title 收回此損失,且不傷任何資料集。

**設計理由(寫進論文)**:title 比對對「短、≈ 實體名」的 decompose sub-q 有利、對「長、多 token」的原問題有害(§4.4 實例:長 query 開 title 會把 `Polish Film School` 等 stray title 拉進前段)。path-aware 依 query 型態(非資料集)切換 → 各取所需。**主結果用 (c) path-aware;(a)(b) 為此 ablation 的對照,量化「開 title」與「path-aware 精修」各自的貢獻。** 檔:`results/{ds}_1000_verify_v2_title{off,global}.jsonl` + `..._pathaware.jsonl`。

### 3.2 Verifier evidence-support ablation(v2+evidence、v3 chain)— 非最終法

`support_margin=0.35, corpus=wiki`。v3 額外把每個 decomposition step 答案對該步 passages 評分,取 final 與 chain-level 的 max。

| Dataset | v2 no evidence(final) | v2 + final evidence | v3 + chain evidence |
|---|---|---|---|
| HotpotQA | 31.9/41.2 | 32.4/41.8 | 32.4/41.6 |
| 2Wiki | 45.2/51.3 | 41.0/46.9 | 42.7/48.5 |
| MuSiQue | 22.5/30.5 | 21.5/29.0 | 21.5/29.1 |
| PopQA | 50.7/52.9 | 50.4/52.8 | 50.4/52.9 |

evidence support 微幫 HotpotQA(+0.5)但傷多跳(lexical 命中不保證 reasoning chain 成立)。v3 chain-level 修回部分 2Wiki 傷害(−4.2→−2.5)仍輸。**無單一 training-free 全域 evidence rule 能同時改善四個 → 只當 ablation/failure analysis**。檔:`results/{ds}_1000_verify_v3_chain_evidence.jsonl`。

### 3.3 Router round-2 — 為何單一全域 τ 不是答案

`--skip-decompose-tau τ`:probe 原問題,max_ent<τ 則 skip(single-step)。

(A) **強制 single-step**:HotpotQA +4.6 EM(贏),但 2Wiki −27.6、MuSiQue −12.5(崩)。decompose-vs-single 強烈 dataset-dependent。

(B) **單一全域 τ**:EM(skip%),ref = all-decompose / all-single。

| Dataset | τ=0.5 | τ=0.8 | τ=1.2 | all-decomp | all-single |
|---|---|---|---|---|---|
| HotpotQA | **33.0 (40%)** | 32.8 (58%) | 32.2 (72%) | 25.7 | 30.3 |
| 2Wiki | 41.2 (22%) | 35.5 (40%) | 29.3 (57%) | **45.6** | 18.0 |
| MuSiQue | 20.8 (20%) | 19.6 (32%) | 17.2 (49%) | **22.7** | 10.2 |
| PopQA | 47.7 (42%) | 43.1 (61%) | 40.1 (76%) | 50.7(k3) | — |

HotpotQA:router 33.0 > all-single 30.3 > all-decompose 25.7 —— query-level entropy routing 在這裡真的有效。但 2Wiki/MuSiQue/PopQA:**任何 skip 都傷**(單調),其低 entropy 題是 confident-but-wrong 不是 easy。→ **單一全域 τ 無法跨資料集**(HotpotQA 要高 skip、其餘要零 skip)。§3.1 的 entropy-aware **verifier** 才是解:per-query 拿到 HotpotQA gain 而不傷其餘三個。

### 3.4 top_k sweep(decompose path,honest)+ oracle-context note

| Dataset | k=3(floor) | k=5 | k=8(headline) | k=10 |
|---|---|---|---|---|
| HotpotQA | 25.7/33.5 | 24.8/33.2 | 25.7/33.9 | 25.8/34.2 |
| 2Wiki | 41.1/47.0 | 43.3/49.2 | **45.6/51.5** | — |
| MuSiQue | 20.3/26.7 | 21.1/28.3 | **22.7/30.5** | — |
| PopQA | 50.7/52.9 | — | — | — |

k=8 對三多跳最佳;**HotpotQA 飽和**(k8≈k10)→ 瓶頸是 query 品質(KG-triple sub-q),非 recall 深度。k=3 是嚴格 floor(對齊 DeepRAG per-step top-3),k=8 是 headline 操作點。

### 3.5 對比:distractor pool(我方 headline,非 DeepRAG pool)

| Dataset | Ours synthOFF EM/F1 | vs wiki k3 |
|---|---|---|
| 2Wiki | 65.2/73.1 | +24.1 |
| HotpotQA | 41.9/53.5 | +16.2 |
| MuSiQue | 44.3/53.7 | +24.0 |
| PopQA | 48.7/51.2 | −2.0 |

distractor 把多跳灌水 +16~24 EM(gold 保證在 10–20 候選裡);PopQA 單跳 ≈ 持平。

### 3.6 Caveats（honest pool 比較）

- DeepRAG 數字是其 paper 的,非同 harness 重跑(EM/F1 normalize 可能小差)。
- DeepRAG PopQA 42.5/47.8 provenance = 舊 §6;需對 DeepRAG paper 再核(PopQA 可能不在其主表)。
- DeepRAG = full DPO + SC-trained gate;SEEK = LoRA-SFT decomposer + training-free entropy verifier(~1/4 訓練量)。
- 我方 open-domain 是 plain BM25 top-k → 多跳差距部分是 **retrieval 品質**差,非純 answering 差(見 §4.3 M3,但 M3 在 distractor pool 測,open-domain M3 待跑)。
- **論文待決定**:(a) honest pool 當主表(誠實但 HotpotQA 輸)並解釋;(b) 先補 open-domain retrieval 差距;(c) distractor 當主表但分 protocol 列表、停止宣稱多跳贏 DeepRAG。verifier 縮小但未消除 HotpotQA 差距,此決定仍開放。

---

## 4. 支援分析

### 4.1 Oracle direct-vs-decompose 上界(honest wiki-21M)

來源:`wiki21M_oracle_direct_decompose_full.md`。oracle 用 gold EM 在 forced-direct / forced-decompose 間挑對的(上界,非可部署)。HotpotQA/2Wiki/MuSiQue k=8、PopQA k=3。

| Dataset | Direct | Decomp | Best router(τ0.5) | **Oracle** | gain vs best pure | directOnly | decompOnly | bothWrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | 30.3 | 25.7 | 33.0 | **39.8** | +9.5 | 141 | 95 | 602 |
| 2Wiki | 18.0 | 45.6 | 41.2 | **50.6** | +5.0 | 50 | 326 | 494 |
| MuSiQue | 10.2 | 22.7 | 20.8 | **26.3** | +3.6 | 36 | 161 | 737 |
| PopQA | 37.1 | 50.7 | 47.7 | **53.5** | +2.8 | 28 | 164 | 465 |

`directOnly`/`decompOnly` = verifier 最多可救的兩類;`bothWrong` 兩路徑都救不了(需更好 retrieval/decomp/synthesis)。HotpotQA oracle 上界 39.8 ≈ DeepRAG 40.7 → **理想 verifier 幾乎追平 DeepRAG**,§3.1 verifier v2 (31.9) 仍有 +7.9 headroom。

### 4.2 Decompose-vs-direct 2x2 一致性(honest,phi)

來源:`decompose_vs_direct_2x2.md`。decompose = `*_3ds_terse.jsonl`,direct = `*_alwaysskip.jsonl`(同 12K adapter)。已套 value→entity guard(2Wiki dirOnly 83→39,44 個是 value-return 型;hotpotqa 197→180)。

| dataset | bothC | bothW | dirOnly | decOnly | phi |
|---|---|---|---|---|---|
| 2wiki | 172 | 478 | 39 | **311** | 0.344 |
| musique | 161 | 473 | 65 | **301** | 0.271 |
| popqa | 272 | 563 | 39 | 126 | 0.654 |
| hotpotqa | 288 | 435 | **180** | 97 | 0.444 |

2Wiki/MuSiQue:decOnly ≫ dirOnly → decompose 主導。HotpotQA:dirOnly(180) > decOnly(97) → direct 反而救更多(對應 §3.3 single-step 贏 HotpotQA)。

### 4.3 M3 — decomposition 是否比原問題更會 retrieve gold?(distractor pool)

來源:`recall_ab_m3.md`。BM25 over per-question pool,measure gold(`is_supporting`)recall。A=原問題單一 query;B-match=sub-q best-match,同 k budget(公平對照)。

recall@k / full-coverage@k(1000q):

| dataset | k | A recall | B-match | Δrecall | A full-cov | B-match |
|---|---|---|---|---|---|---|
| 2Wiki | 3 | 63.8 | **82.6** | +18.9 | 34.5 | **63.4** |
| 2Wiki | 5 | 75.7 | **95.4** | +19.7 | 51.6 | **89.6** |
| HotpotQA | 3 | 71.0 | **78.2** | +7.2 | 46.1 | **60.3** |
| HotpotQA | 5 | 81.4 | **87.1** | +5.7 | 64.8 | **76.2** |
| MuSiQue | 3 | 46.4 | **70.0** | +23.6 | 15.5 | **53.0** |
| MuSiQue | 5 | 56.1 | **78.0** | +21.9 | 27.0 | **64.5** |

結論:**同 budget 下 decomposition 顯著提升 gold recall**(+5.7~+23.6),效果隨 hop-bridging 難度遞增(MuSiQue ≫ 2Wiki ≫ HotpotQA;HotpotQA 第二跳常與原問題共享 surface entity,單 query 已能 retrieve → decomposition 幫最少)。PopQA N/A(單跳無 supporting label)。Caveat:在 distractor pool 測,open-domain 21M recall 待跑。重現:`bash scripts/run_m3_recall.sh`。

### 4.4 M3 — honest wiki-21M open-domain recall(2026-06-17)

來源:`scripts/recall_ab_wiki.py`(distractor M3 的 honest-pool 版)。查共享 ES `wiki`(psgs_w100, 21M)= DeepRAG 真實 corpus,gold 改用 **article title 比對**(IRCoT/DeepRAG 標準 open-domain recall)。`full-coverage@k` = 整題所有 gold title 都被撈到(多跳要全中才答得對,這才是關鍵指標)。A=原問題單一 query;B-union=每個 sub-q 各自 top-k 取聯集(SEEK decompose 實際看到的)。trace = `*_3ds_terse.jsonl`。

**重大發現:SEEK 的 open-domain 檢索一直 handicap——只比對 `paragraph_text`,沒比對 `title` 欄位。** Adaptive-RAG 的 `ElasticsearchRetriever.retrieve_paragraphs` 有 `query_title_field_too`(預設 `False`),而 `OpenDomainBM25.topk` 從沒傳 → 只 match body。DeepRAG 的 BM25 兩個欄位都 match。title 是找對「文章」最強訊號。2026-06-17 已將 `OpenDomainBM25(query_title_field_too=True)` 設為預設(`seek/retriever.py`)。

BM25 單篇命中天花板(query = gold 自己的 title,純檢索上界):

| dataset | text-only top3/5/10 | **title+text** top3/5/10 |
|---|---|---|
| HotpotQA | 67.5 / 72.9 / 78.9 | **86.5 / 86.8 / 87.3** |
| 2Wiki | 53.5 / 56.5 / 59.5 | **64.2 / 64.7 / 65.1** |

→ 打開 title:HotpotQA 單篇上界 +19、2Wiki +11。但 2Wiki 即使開 title 也只 64%(實體易撞名,BM25 本質限制)。entity-only query rewriting(砍掉 `>> relation`)幾乎沒用(HotpotQA ~0、2Wiki/MuSiQue +1~2)→ 排除「改 query 寫法」這條路;瓶頸是欄位 match 與 BM25 lexical 上限。

full-coverage@k(1000q),**title-off → title-on**(B-union):

| dataset | k=3 | k=5 | k=10 |
|---|---|---|---|
| HotpotQA | 27.3 → **35.8** | 33.7 → **38.7** | 42.0 → **43.9** |
| 2Wiki | 21.2 → **32.1** | 24.9 → **33.8** | 29.3 → **36.3** |
| MuSiQue | 14.4 → **19.3** | 18.7 → **21.5** | 28.5 → 26.2 |

per-type full-cov@10(title-on,A / B-union):HotpotQA bridge(807)16/**38**、comparison(193)60/**69**;2Wiki bridge_comparison(228)**0/13**(需 4 篇,BM25 幾乎無望)、compositional(428)11/**46**、comparison(216)30/**37**、inference(128)9/**45**。

**結論**:(1) honest pool 的瓶頸是**檢索 full-coverage**,不是 gate——decompose 也只有 HotpotQA ~42% / 2Wiki ~36%@k10,遠低於 distractor 的 76%/90%;§4.1 oracle routing 上界本身被此天花板壓住。(2) title-field 修復是**免費的 config fix**,B-union full-cov 在 k=3 回升 +8.5(HotpotQA)/+10.9(2Wiki),decompose vs single 的優勢更被放大(Δfull +24~27@k3)。(3) 注意 title match 對**長 query(arm A 原問題,direct path)反而微傷**(把 stray title hit 拉進來),只對短 sub-q(decompose path)有利 → 對 verifier 的 direct 分支可能要分開設定。(4) 2Wiki bridge_comparison 仍近乎無解(BM25 上限)。重現:`bash scripts/run_m3_recall.sh`(distractor)/ `python scripts/recall_ab_wiki.py ...`(honest)。

**QA 端到端 re-run(套 title fix,all-decompose k8,post value-guard,2026-06-17):**

| Dataset | baseline k8 | **title-fix k8** | ΔEM | DeepRAG | gap |
|---|---|---|---|---|---|
| 2Wiki | 45.6 | **47.2** | **+1.6** | 48.1 | −2.9 → −0.9 |
| HotpotQA | 25.7 | 25.0 | −0.7(flat) | 40.7 | 不變 |

符合 M3 預測:title fix 對 **2Wiki decompose path 真實 +1.6 EM**(DeepRAG gap 砍到 −0.9,接近持平但未翻盤);**HotpotQA 持平**(其 decompose path 是 query-quality-bound 不是 coverage-bound,且 title match 微傷 direct path)。2Wiki 殘留缺口 = bridge_comparison 檢索死區(BM25 上限,title 也救不了)。檔:`results/{ds}_1000_wiki_k8_titlefix.jsonl`。**注意:value→entity guard 現已 inline 在 pipeline(`_rescue_value_answer`),離線 rescue script 對新檔 chg=0。**

**verifier v2 canonical re-run(套 title fix,`--no-evidence`,2026-06-17)→ 揭示 title-match 必須分路徑:**

| Dataset | 舊 v2(title-off) | 新 v2(全域 title-fix) | all-decompose(title-fix) | DeepRAG | gap(新 v2) |
|---|---|---|---|---|---|
| 2Wiki | 45.2/51.3 | **45.9/52.5** | **47.2** | 48.1/53.3 | −2.2/−0.8 |
| HotpotQA | 31.9/41.2 | **30.5/39.9** | 25.0 | 40.7/51.5 | −10.2 |

verifier 內部各路徑 EM:2Wiki direct 16.7 / decomp **47.2** / verifier 45.9 / **oracle 52.2**;HotpotQA direct 28.7 / decomp 25.0 / verifier **30.5** / oracle 38.5。choices:2Wiki decomp 938+direct 62、HotpotQA decomp 859+direct 141。

**兩個關鍵**:(1) **2Wiki verifier 反而比純 all-decompose 差**(45.9<47.2,誤路由 62 題到 16.7-EM 的 direct)→ 2Wiki 最佳配置 = **純 all-decompose 47.2**(gap −0.9);oracle 52.2>DeepRAG 48.1 表示路由 headroom 仍 +6.3(training-free 吃不滿,且**不訓練 router 分類器**見 memory)。(2) **HotpotQA verifier 路由有效**(30.5>28.7>25.0)**但全域 title-fix 拉低 direct path**(長 query 被 stray title hit 干擾)→ 比舊 v2 跌 −1.4。**結論:title-match 全域開是 wash(2Wiki +0.7、HotpotQA −1.4);正解 = 分路徑——decompose 短 sub-q 開 title、direct 長原問題關 title。** 檔:`results/{ds}_1000_verify_v2_titlefix.jsonl`。下一步:path-aware title 開關(預期 2Wiki +1.6 而 HotpotQA direct 不受傷)。

---

### 4.6 拆解後檢索必要性 + 逐步信心閘門效用(never/gate/always 三方,honest wiki-21M,1000Q)

來源:`scripts/run_3way_ablation.sh` + `scripts/ablate_rag_usefulness.py`(2026-06-21)。三條件**同一條 run_seek.py**(mode C、max_ent、path-aware title、top-k 8、PopQA k3、corpus wiki),只差 `--tau`:never=1e9(永不檢索)、gate=0.3(canonical)、always=0(每步檢索)。fresh gate 欄與 §3.1/§4.5 canonical **完全重現**(2Wiki 47.2、HotpotQA 25.0、MuSiQue 24.7、PopQA 51.5)→ 三方自洽且對齊主表。

| Dataset | never-RAG(只拆解) | gate τ=0.3 | always-RAG | ① never→gate | ② gate vs always | gate 檢索率 |
|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | 15.7 | 25.0 | 25.1 | **+9.3** | −0.1 | 92.7% |
| 2Wiki | 29.4 | 47.2 | 47.1 | **+17.8** | +0.1 | 95.2% |
| MuSiQue | 9.6 | 24.7 | 25.0 | **+15.1** | −0.3 | 90.8% |
| PopQA | 24.2 | 51.5 | 51.4 | **+27.3** | +0.1 | 99.3% |

**① 拆解後 RAG 是必要的(四資料集全成立)**:只拆解不檢索 EM 僅 9.6–29.4;加檢索 +9.3~+27.3 EM。拆解本身不產生事實,檢索才把 parametric 不知道的事實補進來。把 §2.4(舊只有 MuSiQue 200 distractor)的結論釘到 canonical honest pool。

**②(a) 逐步信心閘門 = 效率論證(非準確率)**:gate vs always 四資料集皆 ±0.3 EM 內,但少檢索 0.7–9.2%。**閘門省檢索成本而幾乎不損準確率,並非準確率贏 always**。honest pool 省檢索幅度小於 distractor(PopQA 僅省 0.7%——honest 單跳幾乎全高 entropy,閘門近全開)。

**②(b) entropy 作為「該不該檢索」訊號的辨別力強烈 dataset-dependent**——AUROC(entropy→never-RAG 答錯):

| Dataset | AUROC | 分位數 rescue 趨勢 | 解讀 |
|---|---:|---|---|
| MuSiQue | **0.809** | 遞減(高 entropy 是 retrieval 死區) | 訊號強 |
| PopQA | 0.749 | 遞增 | 訊號好 |
| HotpotQA | 0.696 | 低分位最低、其餘平 | 大致可用 |
| 2Wiki | **0.540** | **遞減 41.8%→29.5%** | **近隨機,confident-but-wrong** |

2Wiki entropy 幾乎無辨別力(0.540),其 parametric 錯答是 **confident-but-wrong**(低 entropy 卻錯),分位數 rescue 率**遞減**(與閘門假設相反)。這解釋:逐步閘門只能當**效率**(近全開幾乎不傷)、而**整題 confidence 路由必須謹慎**(§3.3 單一全域 τ 在 2Wiki/MuSiQue 任何 skip 都傷,正是 confident-but-wrong);MuSiQue/PopQA entropy 訊號良好但最高 entropy 段是 retrieval-ceiling 死區。檔:`results/{ds}_3way_{never,gate,always}.jsonl`。

### 4.7 逐步閘門 τ 的 cost-accuracy tradeoff(2026-06-24,supervisor request)

老師反映 canonical τ=0.3 檢索率 90%+、變化不明顯。兩個層級各做完整 τ 掃描:

**(a) decompose 單路徑(`run_seek.py`,11 個 τ 點 0→1e9)** — 檔 `results/cost_accuracy_tradeoff.{md,png}`,腳本 `scripts/run_tau_sweep.sh` + `plot_cost_accuracy.py`。
- τ≤0.5 段曲線平(檢索 89–99%、EM 幾乎不動);canonical 0.3 在此段內、非最佳(0.2 微高,雜訊內)。**在現在操作點細調 τ 無意義。**
- τ≈1.0 = 免費效率點(檢索 −7~14pp、EM ≤0.4);τ≥1.5 才明顯掉(壓到 ~50% 檢索損 3–9 EM)。honest pool 多跳真的需要檢索。

**(b) 完整 SEER 系統(`run_seek_dual.py --direct-context-union` + type-aware replay,主表口徑)** — 檔 `results/seer_tau_table.md`,腳本 `scripts/run_seer_tau_dual.sh` + `seer_tau_table.py`。τ=0.3 精確重現主表(40.52)。

EM / F1(皆 alias-aware、post-replay):

| Dataset | τ=0.3 (canonical) | τ=0.5 | τ=0.8 | τ=1.0 |
|---|---|---|---|---|
| HotpotQA | 35.50 / 46.11 | 35.40 / 46.10 | 35.50 / 46.17 | 35.50 / 46.12 |
| 2Wiki | 50.00 / 56.58 | 50.00 / 56.65 | 49.00 / 55.78 | 48.10 / 55.09 |
| MuSiQue | 24.70 / 33.06 | 24.50 / 32.61 | 23.70 / 31.84 | 22.60 / 30.91 |
| PopQA | 51.90 / 54.71 | 51.90 / 54.71 | 51.70 / 54.38 | 51.80 / 54.41 |
| **Avg** | **40.52 / 47.61** | **40.45 / 47.52** | **39.98 / 47.04** | **39.50 / 46.63** |

每資料集檢索率(decompose-path,cost):

| Dataset | τ=0.3 | τ=0.5 | τ=0.8 | τ=1.0 |
|---|---|---|---|---|
| HotpotQA | 92.07% | 88.44% | 81.79% | 77.58% |
| 2Wiki | 94.70% | 92.03% | 87.82% | 85.40% |
| MuSiQue | 90.55% | 86.02% | 79.28% | 74.31% |
| PopQA | 99.30% | 98.80% | 95.00% | 92.10% |
| **Avg** | **94.16%** | **91.32%** | **85.97%** | **82.35%** |

(τ=0.3 與主表 SEER (Ours) 40.52/47.61 逐格一致。F1 由 `seer_tau_table.py` 自 dev `answer_aliases` 重算,非單一 gold。)

**結論**:完整系統下閘門空間比單路徑更小,曲線單調且平滑。HotpotQA/PopQA 對 τ 免疫(verifier 的 direct 分支 + direct path 撐住,HotpotQA ~13% 選 direct);只有 2Wiki/MuSiQue 吃檢索(τ=1.0 各 −1.9/−2.1 EM)。τ=0.5 近乎免費但只降 2.9pp 檢索(無感);τ=0.8 降 8.2pp 檢索、損 0.54 avg EM(合理中間點);τ=1.0 降 12pp、損 1.0 avg EM。**canonical 維持 τ=0.3(EM 最佳),此表即 cost-accuracy 證據:現操作點已近最佳,閘門再緊就傷準確率。** 逐步閘門定位為效率旋鈕,與 §4.6 一致。

---


### 4.8 Efficiency: training-free adaptation, not lower inference retrieval(2026-07-01)

**修正後的效率主張:** SEEK 不應宣稱「推論時比 DeepRAG 省檢索」。在 honest wiki-21M、1000 題、canonical
verify_v2 / original-question evidence union 口徑下,SEEK 的每題檢索次數在 HotpotQA 與 2Wiki 反而高於 DeepRAG,
只有 MuSiQue 較低。更乾淨、可防守的主張是:SEEK 的 adaptive routing/gating/verifier 是
**training-free** 的,不需要 DeepRAG 式的額外 policy/router 訓練流程;推論成本則應誠實描述為
「與 DeepRAG 同量級,且資料集相依」。

每題檢索次數對照(1000Q, honest wiki-21M, canonical verify_v2;direct path retrieval = 0.000,
SEEK 檢索全來自 decomp path,且 100% 題目都跑 decomp path):

| Dataset | SEEK EM | SEEK retrievals/q | DeepRAG retrievals/q | Difference | Interpretation |
|---|---:|---:|---:|---:|---|
| HotpotQA | 34.3 | 2.078 | 1.716 | +21% | SEEK retrieves more |
| 2Wiki | 49.9 | 2.341 | 1.451 | +61% | SEEK retrieves more |
| MuSiQue | 24.8 | 1.840 | 2.273 | -19% | SEEK retrieves less |
| PopQA | 52.0 | 0.993 | — | — | DeepRAG paper no direct retrieval/q comparison |

**因此不可寫:** "SEEK reduces retrieval cost compared with DeepRAG" 或 "SEEK is cheaper at inference"。
三個多跳資料集中只有 MuSiQue 支持此說法;HotpotQA/2Wiki 都相反。此外,SEEK 還多跑一條
direct answer path(即使 direct retrieval 為 0),所以 wall-clock LLM forward cost 也不能假設更低。

**可寫、且應作為主效率論點:訓練成本(quantified)。** SEEK 的優勢不是「每題必定少查幾次」,而是
**大幅簡化的離線訓練 pipeline**。以下數字兩邊都可查證:SEEK 側為本專案實測(HF `train_results.json` /
`adapter_config.json`,見 `main_methodology/checkpoints/`);DeepRAG 側為其官方 repo(`~/DeepRAG`)的
訓練腳本與資料建構腳本所載之組態。

**重要澄清(資料盤點,2026-07-01):** 我們對 DeepRAG **只復現了推論**——用其釋出的模型
`xinyan233333/DeepRAG-8b` 在 `Research/deeprag_compare/` 跑評測(這就是 EM / avg_retrievals /
elapsed_min 的來源)。**我們從未跑過 DeepRAG 的訓練 pipeline**:全系統搜尋無任何 DeepRAG 訓練
checkpoint、無 `construct/` 輸出、無 trainer_state / train_results / training log。此外 `~/DeepRAG/data/`
內的 `sft_new.jsonl`(4K×2)、`dpo_new.jsonl`(1K×2) **是餵進 tree-search 的種子問題(question/
context/answer),不是 tokenized 訓練序列**;真正的訓練序列(tree-search trajectory,
`construct/sft/tokenized_dataset`)不在磁碟上。因此 DeepRAG 的**真實訓練 wall-clock 與精確 training-token
數都無法從現有資料取得**,除非實際照其腳本跑一次訓練;下表 DeepRAG 的 wall-clock / FLOPs 一律為
同 `total_flos` 口徑的**數量級估算並明標假設**,不是實測。

**(a) 梯度訓練對照**

| 項目 | SEEK decomposer(canonical) | DeepRAG policy |
|---|---|---|
| 微調方式 | **LoRA** (r=16, α=32, 7 個 proj 模組) | **全參數** (`finetuning_type full`, DeepSpeed ZeRO-3) |
| base model | LLaMA-3-8B-Instruct | **LLaMA-3-8B-Instruct**(同底,見下註) |
| 訓練階段 | 1 階段(SFT) | **2 階段(SFT + DPO)** |
| epochs | 3 | 1 (SFT) + **8 (DPO)** |
| 訓練樣本 | 12K decomposition(直接可訓練) | 8K 種子問題→tree-search trajectory(未釋出) + 2K 種子→DPO pairs |
| 硬體需求 | **單 GPU** | 多 GPU ZeRO-3 sharding |
| 可訓練參數 | LoRA adapter(≈ base 的 <1%) | 100%(整個 7B) |
| 產出檔 | 小型 adapter(可插拔) | 完整 7B checkpoint |
| **實測 wall-clock** | **37.9 分(2272.9 s,單卡)** | 未公布;全參數 7B 之 SFT(1ep)+DPO(8ep)為數 GPU-hours 量級 |
| **train FLOPs** | **2.40e17**(實測 HF total_flos) | ≈ 3.3e17(同口徑估算,SFT ~4.7e16 + DPO ~2.8e17) |

**梯度訓練本身兩邊同量級**(FLOPs 約 1.4×),因此**不可宣稱「SEEK 訓練快一個數量級」**。真正的
不對稱在下面兩點。

**註(base model / 模型角色):** 我們基準比較的釋出模型 `xinyan233333/DeepRAG-8b` 為
**Llama-3-8B-Instruct** 系(README 明列為主要 recipe、`scripts/launch/run.sh` 與 `scripts/inference/run.sh`
皆載入 `Meta-Llama-3-8B-Instruct`、`training/stage2.sh` 用 `--template llama3`)。**這與 SEEK decomposer
同一 base(Llama-3-8B-Instruct),故訓練成本為同底比較。** 注意 DeepRAG 資料建構用到兩個不同角色的模型:
(i) 被訓練的**主策略模型** = Llama-3-8B(`run.sh`, 8 ports);(ii) tree-search 的 **follow-up 子問題
生成器** = Qwen2.5-72B(`run-72b.sh`),與主模型是否 8B 無關,即走 Llama-3-8B 路線仍需 72B teacher →
72B 死結不因換底而消失。其 `training/stage1.sh` / filter 腳本內出現的 `Qwen2.5-7B-Instruct` 路徑研判為
repo 內殘留或另一 backbone 變體,非我們基準的那顆模型。

**(b) DeepRAG 獨有、SEEK 完全省掉的成本**

1. **偏好優化階段(DPO):** DeepRAG 需要額外一整個 8-epoch 全參數 DPO 階段來校準檢索決策
   (chain of calibration);SEEK 無此階段。
2. **Trajectory 資料建構(最昂貴、最具區別性):** DeepRAG 的 SFT 資料來自對每個訓練問題做
   **binary-tree search**(`srag-sample`),每題展開多條「retrieve / 不 retrieve」rollout 分支 →
   每題數次至數十次 LLM 生成;且其官方腳本需另外**架設 1 個 72B follow-up 模型 + 至多 8 個 7B
   vLLM server**(`scripts/launch/run-72b.sh`、`data-construct/stage1.sh` 開 8 個 port、
   `num_processes 64`)。相對地 **SEEK 的 SFT 資料 = 每題向 OSS teacher 取 1 次 decomposition**
   (`datasets/*/oss-120b`,`oss-20b`),無 tree search、無 preference pair 建構、無 serving 叢集。
3. **無檢索決策器重訓:** DeepRAG 訓練的是「何時檢索」的策略,換資料集需重跑 data-construct + 兩階段
   訓練;SEEK 的 sub-question retrieval gate、direct-vs-decompose selection、type-aware fallback
   全是 inference-time confidence/consistency rules,零額外訓練、免逐資料集重訓。

**一句話量化:** SEEK 的離線成本 = 一顆單卡、38 分鐘、<1% 參數的 LoRA decomposer;DeepRAG 的離線成本 =
一個需要 72B+8×7B 叢集跑 tree-search 建資料、再做全參數 SFT + 8-epoch 全參數 DPO 的兩階段 policy-learning
pipeline。差距不在梯度小時數(同量級),而在**省掉 DPO 階段 + 省掉 tree-search trajectory 資料建構與其
serving 叢集 + 免逐資料集重訓檢索決策器**。

**論文/答辯口徑:** SEEK is a lightweight adaptive RAG framework whose routing, gating, and verification are
training-free; its only trained component is a single LoRA decomposer (~38 min, single GPU, <1% params).
DeepRAG instead learns a retrieval *policy* via binary-tree-search trajectory construction (requiring a
72B + 8×7B serving cluster) followed by full-parameter SFT and 8-epoch DPO. The saving is not fewer
retrieval calls at inference (comparable, dataset-dependent) nor an order-of-magnitude fewer gradient
FLOPs (also comparable), but the elimination of the DPO stage, the trajectory data-construction pipeline,
and any per-dataset retraining of a retrieval decision-maker.

---


### 4.9 6/25 thesis presentation memory（答辯敘事版）

來源：`Research/Thesis/6_25_ThesisPresentation_FULL_speaker_script.md`，尤其 Part 4 Analysis 與 final Q&A notes。此節是 presentation-facing memory，用來固定答辯口徑；若與舊 canonical 小節不一致，以本節作為 6/25 簡報敘事的最新版本。

**核心 motivation：把 adaptive RAG 重新表述為 decomposition-routing problem。** Multi-step retrieval 的價值在於把複雜問題拆成較清楚的 sub-questions，使 retriever 更容易找到 intermediate evidence；但 decomposition 不是 always better。拆短後可能丟失原問題中的 comparison condition、entity pair、disambiguating clues，造成 retrieval drift 或 final synthesis 錯誤。因此 SEEK 的問題設定不是「是否要 decomposition」，而是「什麼時候相信 decomposition path，什麼時候保留 direct path」。

**SEEK 的 6/25 簡報定位：lightweight, interpretable, dual-path adaptive RAG。**
- Only trained component：12K three-source LoRA decomposer，學 structured sub-question plan。
- Sub-question retrieval gate：max entropy 決定每個 sub-question 是否 retrieve，定位為 efficiency knob，不宣稱勝過 always retrieve。
- Context preservation：original-question evidence union 把原問題檢索到的文章加回 decomposed steps，補回 decomposition 丟掉的全局線索；它共享的是 evidence，不共享 direct answer。
- Final selection：training-free verifier 同時看 direct answer 與 decomposed answer，用 confidence 與 consistency signals 選 final answer；不訓練額外 router，不做 DeepRAG 式 tree-search policy learning。

**6/25 presentation-facing main system：original-question evidence union + type-aware verifier replay。** 口頭簡報主表採用 SEER/SEEK final candidate：τ=0.3、original-question evidence union、type-aware verifier replay，平均 **40.52 EM / 47.61 F1**，高於 DeepRAG 平均 **38.45 EM / 45.75 F1**。逐資料集口徑：HotpotQA **35.50 / 46.11**（仍輸 DeepRAG 40.7 / 51.5）、2Wiki **50.00 / 56.58**（贏 DeepRAG 48.1 / 53.3）、MuSiQue **24.70 / 33.06**（贏 DeepRAG 22.5 / 30.4）、PopQA **51.90 / 54.71**（贏 DeepRAG 42.5 / 47.8）。

**Analysis slide 45：先講 final verifier 選路比例，再講兩路互補。** Final candidate 實際選 direct/decompose 的比例為 HotpotQA `27.4% / 72.6%`、2Wiki `6.5% / 93.5%`、MuSiQue `18.2% / 81.8%`、PopQA `3.7% / 96.3%`，pooled 為 `14.0% / 86.1%`。這說明系統大多數題仍走下路 decomposition；上路 direct 是保留給 HotpotQA 或 decomposition 不穩時的 context-preserving candidate。Answer-level 2x2 則顯示兩路互補：2Wiki、MuSiQue、PopQA 呈現 decomposition-dominant（2Wiki decompose-only `361` vs direct-only `39`；MuSiQue `184` vs `31`），HotpotQA direct-only `111`、decompose-only `113` 幾乎打平。Both-wrong 仍是大瓶頸：HotpotQA `584`、MuSiQue `714`，說明剩餘錯誤不能只靠 router 解決，仍需要更好的 retrieval、reranking、decomposition quality 或 final synthesis。

**Analysis slide 46：oracle gap 量化 routing headroom。** Direct EM 與 Decomp EM 是固定路徑，Current Router EM 是 SEEK verifier 選出的結果，Oracle EM 是事後用 gold answer 選路徑的不可部署 upper bound。HotpotQA direct `30.30`、decomp `30.50`、current router `35.50`、oracle `41.60`，剩餘 gap 為 `+6.10 EM`。2Wiki、MuSiQue、PopQA 也有約 `+3.8` 到 `+4.1 EM` 的 oracle gap。答辯重點：SEEK 並未宣稱 rule-based verifier 完全解決 routing；相反地，oracle gap 是 thesis novelty 的分析證據，說明 direct-vs-decompose routing 本身仍有明確 headroom。

**Analysis slide 47：entropy 有用，但不是 correctness guarantee。** Entropy 的合理用途是 step-level retrieval gating：higher entropy 通常 associated with higher error rate，pooled Spearman 約 `+0.326`。但 low entropy 不保證正確，模型可能 confident-wrong。因此 entropy 不能作為 whole-question direct-vs-decompose router 的唯一依據；SEEK 保留 dual-path design 與 verifier，就是為了避免單一 entropy threshold 支配所有決策。

**答辯 Q&A 固定口徑。**
- 「Decomposition 是否沒用？」不是。Decompose-only counts 與 retrieval recall 都顯示 decomposition 有用；精確說法是 decomposition 對很多 multi-hop questions 有用，但不是每題都最好，因此需要 routing。
- 「Oracle gap 是不是方法？」不是。Oracle 使用 gold answer 事後選 direct 或 decompose，不能部署，只用來量化 routing headroom。
- 「Entropy 會失敗為什麼還用？」因為 entropy 在 sub-question step-level gate 上低成本、可解釋且有效；失敗的是把 entropy 當成唯一 global router。
- 「HotpotQA 為什麼仍輸 DeepRAG？」HotpotQA direct-only 與 decompose-only 幾乎打平，且同時有 over-decomposition、context loss、retrieval drift、both-wrong bottleneck。Oracle 到 `41.60 EM`，但 current router 只有 `35.50 EM`，所以主要 limitation 是 routing 與 retrieval/synthesis bottleneck，而不是 decomposition 完全無效。

**一句話收束：** SEEK 的貢獻不是單純提出 question decomposition，而是把 decomposition 放進 direct-vs-decompose routing framework；用 entropy-gated sub-question retrieval、original-question context preservation 與 training-free verifier，在不訓練 policy router 的情況下取得可解釋、輕量且具競爭力的 adaptive RAG。

### 4.10 Gate signal 選擇的文獻核對與四資料集全訊號比較（2026-07-02）

詳見 `results/gate_signal_paper_fidelity_and_analysis.md`；數據 `results/gate_signal_replay_analysis.md`（4 datasets × 8 signals × n=1000，paired-bootstrap 95% CI）。

- **「MAQA/TARG 都說 entropy 差」是誤記**：MAQA (NAACL 2025 Findings) 結論 entropy 是 white-box 最佳（AUROC 71.18 vs margin 63.82）；只有 TARG (arXiv:2511.09803) 主張 margin > mean-entropy，且是在單跳、整題 20-token draft 的設定。
- **實作忠實度已核對**：`maqa_margin` = first-token top1−top2 softmax margin（同 MAQA）；`targ_unc` = prefix-mean exp(−logit_gap/β)，k=20、β=3（同 TARG 預設）。
- **結果**：max_ent macro AUROC(need) 0.689 為全訊號最佳；HotpotQA 上顯著贏所有 margin/TARG 訊號；PopQA（TARG 自用資料集）上 entropy 贏 targ_unc Δ=+0.106（p<0.001），直接反轉 TARG 結論；MuSiQue 上 min_lp≈max_ent（不顯著）；2Wiki 全訊號近隨機（資料集問題，非訊號選擇問題）。
- **論文口徑**：採 max_ent 有文獻（MAQA）與自家四資料集復現（含 TARG 忠實設定）雙重支撐。


## 5. 待辦 / 限制 / 已完成

### ✅ 已完成
- distractor 1000Q 全 4 dataset;PopQA baseline(17.9 / +C 30.3)
- 2Wiki bridge_comparison 9.6 EM debug → synthesize prompt fix
- synthesis value→entity guard(2026-06-11,零 regression)
- HotpotQA failure analysis(2026-06-23):direct-only 151 > decomp-only 98,代表性錯誤案例與改善方向已整理至 `results/hotpotqa_failure_analysis.md`
- conservative date-comparison synthesis guard(2026-06-23,offline replay):HotpotQA +0.3 EM/+0.5 F1,2Wiki +0.9 EM/+0.9 F1,MuSiQue/PopQA 0 regression
- original-question evidence union ablation(2026-06-23,formal 1000Q,current pipeline):正確隔離口徑為 decomp candidate EM,HotpotQA 25.0→30.5(+5.5),2Wiki 47.2→50.2(+3.0),MuSiQue 24.7→25.5(+0.8),PopQA 51.5→52.9(+1.4);full-system reference 為 HotpotQA 31.8/41.2→34.3/44.3,2Wiki 46.9/53.5→49.9/56.4
- type-aware verifier offline replay(2026-06-23):final candidate avg 40.5/47.6;HotpotQA 34.3/44.3→35.5/46.1,2Wiki 49.9/56.4→50.0/56.6
- honest wiki-21M round-1 sweep + round-2 router + verifier v1/v2/v3
- verifier v2 定為最終法
- **never/gate/always 三方對照(honest pool,4 dataset,§4.6)**:拆解後檢索 +9.3~+27.3 EM(①必要)、逐步閘門 ±0.3 EM 省 0.7~9.2% 檢索(②效率)、entropy 辨別力 AUROC 0.54(2Wiki)~0.81(MuSiQue)

### ⚠️ 已驗證的限制(寫進 paper limitation)
- **Entropy skip-decompose 全域 τ 行不通**:同 τ=2.0 套全部 → PopQA +12.4 但 MuSiQue −17.9、2Wiki −15.6(多跳 false-confident,skip rate 73–87%)。最終不開 option C。
- **單一全域 router τ 無法跨 dataset**(§3.3);需 per-query verifier 或更強信號(self-consistency variance、Top-1/Top-2 gap)。
- **evidence support 無全域 win**(§3.2)。
- **HotpotQA honest pool 仍輸**:original-question evidence union 先把 decomp candidate 25.0 EM 拉到 30.5 EM,full-system 再從 31.8/41.2 到 34.3/44.3,證明保留原題 disambiguating clues 有用;但 final 仍低於 DeepRAG 40.7/51.5。主瓶頸仍是 over-decomposition、open-domain retrieval query drift / BM25 full-coverage 與 both-wrong 題,非單純 gate。

### 🔭 未動 / 可選
- MuSiQue 4hop 強化(過採樣 4hop 訓練;自然分佈僅 6%)
- 將 conservative type-aware verifier replay 整合為 `run_seek_dual.py --typeaware-fallback` optional mode,再決定是否升 canonical
- original-question evidence union follow-up:補 MuSiQue/PopQA regression check;再試 rerank/limit union evidence,避免 context 變長造成 lost cases(HotpotQA −21,2Wiki −47)
- open-domain M3 recall A/B(需 ES;見 ES-heap-fix)
- 2Wiki/HotpotQA τ 掃描(目前只 τ=0.3)
- Bamboogle 125 test、LLM-as-Judge metric(PopQA 很多是語意對格式錯)
- PopQA 結構短板:SFT 加單跳「單步直答」decomposition,或 DeepRAG 同款 `[Retrieval]` token + DPO
