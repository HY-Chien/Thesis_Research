# 老師回饋逐條對照(6/25 簡報後,整理於 2026-07-05)

每一條:**老師的意見 → 現況(✅已有 / 🟡部分 / ❌未做)→ 數據與檔案位置 → 還需要補什麼**。
簡報頁碼以 `Thesis/6_25_ThesisPresentation.pptx (1).pdf` 的 PDF 頁碼為準
(p38=Retrieval Recall、p41=τ sensitivity、p42=Path-aware Title、p43=Evidence Union、
p45=2×2 correct distribution、p46=Oracle Gap、p47=Entropy as Confidence Signal)。

---

## 2. 與 DeepRAG 的「省資源/省時間」比較是否成立?需不需要量化?

**現況:🟡 部分已有 —— 檢索次數與訓練成本已量化,結論是「推論不省,省的是訓練 pipeline」;推論 wall-clock 與固定 timeout 情境未做。**

已有的量化(`RESULTS_MASTER.md` §4.8,2026-07-01):

- **每題檢索次數(1000Q, honest wiki-21M)**:SEEK 在 HotpotQA **+21%**、2Wiki **+61%** 反而比 DeepRAG 檢索更多,只有 MuSiQue **−19%** 較少。→ **不可以在簡報/論文寫「SEEK 推論時比 DeepRAG 省檢索/省時間」**,這句話會被數據直接打臉。
- **訓練成本(可防守的主張)**:SEEK 唯一要訓練的是 LoRA decomposer —— **單卡 37.9 分鐘、<1% 參數、2.40e17 FLOPs**(實測,`main_methodology/checkpoints/*/train_results.json`)。DeepRAG 需要 tree-search trajectory 資料建構(72B teacher + 8×7B vLLM 叢集)+ 全參數 SFT + 8-epoch 全參數 DPO,且換資料集要整條重跑。注意:**梯度 FLOPs 兩邊同量級(約 1.4×),不可宣稱「訓練快一個數量級」**;省的是 DPO 階段、tree-search 資料建構、serving 叢集、逐資料集重訓。
- DeepRAG 推論數據來源:`deeprag_compare/results/`(EM / avg_retrievals;`learned/*_summary.json` 內含 elapsed)。

**需要補的:**
1. **推論 wall-clock 對照表**:DeepRAG 側 elapsed 已存在 summary json;SEEK 側需從 `main_methodology/logs/` 彙整或重跑一次計時。做完才能誠實回答「時間」問題(預期結論:同量級、不占優,因為 SEEK 多跑一條 direct path)。
2. **固定 timeout 情境**(老師建議的應用情境):完全未做,屬新實驗。若要做,最便宜的版本是離線用既有 traces + 每題耗時,畫「時間預算 vs 答對率」曲線,不必重跑模型。
3. 簡報口徑改為 §4.8 最後一段的英文版敘述(training-free adaptation),不要再講省時省資源。

---

## 3. Methodology 前面要先用數據鋪陳設計動機

**現況:🟡 動機數據幾乎都已存在,缺的是「拉到 Methodology 前面」的敘事重組;可能要補「開發集版本」以免用測試結果當動機。**

每個設計對應的動機證據(都可以前移):

| 設計 | 動機主張 | 現有證據 | 檔案 |
|---|---|---|---|
| 為何要 decomposition | 拆解讓 retriever 找得到 | decomp hit@5 比原問題 +10~+31 pp | `results/answer_containing_recall_4ds_open_domain.md`(即 p38) |
| 為何拆解後要檢索 | 只拆不檢索 EM 崩 | never→gate +9.3~+27.3 EM | RESULTS_MASTER §4.6、p39 |
| 為何用 entropy 當 gate 訊號 | entropy 與錯誤率相關 | pooled AUROC 0.686、Spearman +0.326;8 訊號比較中 max_ent macro AUROC 0.689 最佳 | `experiment/maxent_correctness/RESULTS.md`;`results/gate_signal_replay_analysis.md`(§4.10) |
| 為何不能只用 entropy 路由、要 dual-path + verifier | confident-wrong 存在;全域 τ 失敗 | option C 全域 skip:MuSiQue −17.9 / 2Wiki −15.6;confidently-wrong 9.5~28.8% | RESULTS_MASTER §3.3、限制節;`results/quadrant_confidence_analysis.md` §3 |
| 為何保留原問題 context(union) | 拆解會丟 disambiguating 條件 | constraint-loss 在 HotpotQA 占 10% 錯誤;union +5.5 EM | `results/decomp_failure_taxonomy.md`;`results/union_ablation_pathlocal_summary.md` |

**需要補的:**
1. **敘事層**:把上表做成 Methodology 前的 1–2 頁「observation → metric → design」鋪陳(這是寫作/簡報工作,不是新實驗)。
2. **方法論潔癖問題**:上述分析多數跑在 dev 前 1000 題(同一批最後報 performance 的題目)。若口試被問「動機分析是不是看著考卷設計」,最穩的補法是挑 1–2 個關鍵動機(entropy–錯誤率相關、decomp recall 增益)在**另一段 dev 切片或 train 子集**重跑一次小規模版本。腳本都是現成的,成本低。

---

## 4. 訓練拆解模型的難度、用完我們的方法後還剩什麼問題、兩個方向

**現況:🟡 兩個方向的「證據」都已有,缺的是把它們組成「remaining problems → solution」的重點敘事。**

**方向 1:拆解會怎麼拆 / 會遇到什麼問題(用 MuSiQue 這類資料訓練完的行為)**

- **失敗分類已做**(`results/decomp_failure_taxonomy.md`,2026-07-03,腳本 `scripts/analyze_decomp_failure_taxonomy.py`):decompose 路徑錯誤中 `retrieval_gap_both` 占 **47%(HotpotQA)/ 64%(2Wiki)/ 68%(MuSiQue)** —— 主要死因是檢索不是 decomposer;**constraint loss(子問題丟掉限定條件)是 HotpotQA 特有(10% vs 其他 1–2%)**,正好解釋為什麼 MuSiQue 式 KG-triple 拆法在 HotpotQA 吃虧、以及 union 為何在 HotpotQA 增益最大;synthesis failure 穩定占 15–18%;另有 `subq1_restates_q`(第一子句復述原題)標記。
- **decomposer 內在品質**:RESULTS_MASTER §1.1–1.3(MuSiQue gold 4000 LoRA、+2Wiki 8000、GPT-OSS-20B 對照)。
- HotpotQA 個案:`results/hotpotqa_failure_analysis.md`。

**方向 2:用信心值會有什麼問題(高信心答錯、bias)**

- **高信心答錯的解剖**:`experiment/maxent_correctness/high_low_confidence_error_analysis.md`(2026-06-28)—— 最低 entropy 十分位的 169 個錯誤 **全部是 deterministic wrong path**(`n_uniq=1, sc_agree=1.0`),5 類失敗型態。
- **bias 檢查**:`experiment/maxent_correctness/confidence_popularity_bias.md`(2026-07-03)—— PopQA 上**沒有 raw popularity→confidence bias**(|rho|<0.05 n.s.);熱門實體的「自信錯誤」28% 是 alias 假錯,冷門實體的自信錯誤 100% 是真錯。結論:confident hallucination 是 deterministic-wrong-path 問題,不是 popularity bias。
- **量化的 confidently-wrong**:`results/quadrant_confidence_analysis.md` §3 —— both_wrong 中 9.5%(PopQA)~28.8%(2Wiki)entropy 低於 both_correct 中位數,任何閾值抓不到。
- 2Wiki entropy 近隨機(AUROC 0.540,§4.6b)是 dataset 級證據。

**需要補的:**
- 不是實驗,是**一頁「用完 SEEK 之後還剩什麼」**:both-wrong 是最大瓶頸(HotpotQA 584 / MuSiQue 714 題),需要更強 retrieval / rerank / final synthesis;confident-wrong 需要 self-consistency 或外部證據檢查,不是調 τ 能解的。素材在 p49 limitation + §4.12,把它升格成重點段落。
- 「train 拆解模型的難度」如果要正面講,可用 §1.2 的教訓(加 2Wiki 資料的 v2 與 v1 對比)+ DeepRAG 需要 72B teacher 的對照(§4.8)。

---

## 5. 用 MuSiQue 原本的 gold sub-questions 跑 p38 / 用原本模型拆

**現況:✅ 全部補做完成(2026-07-05):gold 拆解 recall + 未微調 base 模型拆解對照。**

- 檔案:`results/musique_gold_decomp_recall.md`;腳本 `scripts/recall_gold_decomp_musique.py`;逐題明細 `results/musique_gold_decomp_recall_detail.jsonl`。同 p38 口徑(wiki-21M、path-aware、answer-containing hit@k),N=1000(全為 2-hop)。
- **主結果(hit@5)**:原問題 **19.0%** < SEEK 拆解 **38.8%** < gold 拆解 **55.8%**;`gold_raw`(不代入中間答案、保留 `#1`)只有 **19.8% ≈ 原問題**。
- **關鍵發現**:SEEK 中間答案 EM 僅 **~42%**;條件化 hit@5(gold 用**同子集**算,2026-07-06 修正選擇偏誤):中間答案答對時 seek **56.4** ≈ gold 同子集 **58.5**;答錯時 seek 崩到 **25.9** 而 gold 幾乎不掉(**53.8**)—— 答錯的題目本身沒有變難(gold 證明檢索得到),**seek 與 gold 的 17pp 差距完全由中間答案錯誤解釋,decomposer 的拆解文字不是瓶頸**(拆得跟 gold 同構),瓶頸是 bridge entity 答錯後 step-2 查詢跟著歪。這同時回應 #4 方向 1。
- **未微調 base 拆解對照(同 prompt、無 LoRA,`scripts/decompose_base_musique.py`)**:2-hop 題目 base 只有 149/1000 拆成 2 步(SEEK 963/1000),15.3% 格式解析失敗;oracle 中間答案下文字檢索力 52.9%@5(還吃 1.5 倍 union 預算紅利)vs SEEK 文字 54.1 / gold 55.8 —— **訓練買到的是步數與格式的結構可靠性,不是查詢文字**。完整階梯(@5):不拆 19.0 → 可部署 SEEK 38.8 → oracle 中間答案下 base 52.9 ≤ seek 文字 54.1 ≤ gold 55.8。
- 注意:與 6/24 舊報告有 ~1–2pp 實作差(舊腳本未留存);各 arm 出自同一實作、內部一致,引用以新表為準。
- **投影片稿已完成(2026-07-06)**:`slides/advisor5_6_gold_decomp_direct_gate_slides.md` 頁 1–3(oracle 天花板 / 同子集條件化歸因 / base 對照),插 p38 後;另含 HotpotQA 例外 backup 頁(機制 + 四證據 + union 修復弧線 + oracle 39.8 ≈ DeepRAG 40.7)。

**仍可選(未做):**
1. **gold sub-questions 端到端**(`run_seek.py`):量化 decomposer 品質對 final EM 的天花板(需 GPU 推論)。

---

## 6. 「不拆解、只用信心度決定要不要 RAG」的 baseline(凸顯 p41 拆解的作用)

**現況:✅ 已做完(2026-07-04),就是老師記得的那個實驗。**

- 檔案:`results/direct_gate_ablation.md` + RESULTS_MASTER §4.11;腳本 `scripts/run_direct_norag_probe.py`(parametric probe)+ `scripts/replay_direct_gate.py`(離線 τ 全曲線,τ=0 精確重現 always-RAG ✓);traces `results/{ds}_1000_direct_norag.jsonl`。
- 關鍵數據(EM):即使給 direct-gate **事後最佳 τ**,2Wiki 18.6 vs decompose+gate **47.2**、MuSiQue 10.8 vs **24.7**、PopQA 39.2 vs **51.5**(差 **+12.3~+28.6**);HotpotQA 是唯一例外(30.7 vs 25.0,但 full SEEK 31.8 仍 ≥ 30.7)。
- 論點:訊號不差(AUROC 0.72–0.81)但 gate 只能在兩個爛候選間二選一 —— **把候選變好的是拆解,不是 gate**。
- **投影片稿已完成(2026-07-06)**:`slides/advisor5_6_gold_decomp_direct_gate_slides.md` 頁 4(direct-gate 主表)+ 頁 5(AUROC 定義、query-level vs step-level 分化、「gate = 效率非準確率」鏡像結構),插 p41 後。

---

## 7. p42 要講詳細/直覺:「比對標題」是比對 document title

**現況:✅ 數據齊全,純簡報表達問題,不用補實驗。**

- 支撐數據:RESULTS_MASTER §4.5 title-handling ablation(2026-06-20)+ p42 現有表格;相關 traces `results/*_verify_v2_titleoff*.jsonl`、`*_titlefix.jsonl`。
- 講法(7/4 已為 p43 整理過同款素材,含真實 BM25 例子,見 `slides/` 修訂稿):**title 是 BM25 索引裡 Wikipedia 文章的 title 欄位**。短子問題通常指向單一實體 → title 欄位命中就是對的文章;長原問題詞多 → title 欄位會把每個提到的詞的文章都拉進來 = **noise**(HotpotQA title-on 30.5 vs title-off 31.9 就是這個 noise 的代價)。
- 建議:p42 加一個一行示意(同一 query 在 title-on/off 下撈到的 top 文章對比),素材可從 `*_m3_wiki_titlefield_detail.jsonl` 抓現成例子。

---

## 8. p43 比較對象錯誤:只該比 decompose 路徑有無 union + 補路徑百分比

**現況:✅ 正確口徑的數據已算好(2026-07-04);❌ 投影片本身還沒改。**

- 檔案:`results/union_ablation_pathlocal_summary.md`(腳本 `scripts/analyze_union_ablation_pathlocal.py`)。
- **正確的比較表(§1,decomp candidate only,不混 direct)**:HotpotQA 25.0→30.5(+5.5)、2Wiki 47.2→50.2(+3.0)、MuSiQue 24.7→25.5(+0.8)、PopQA 51.5→52.9(+1.4)。p43 現在那張表(31.8→34.3 等)混入了 direct 路徑與 verifier,必須換掉。
- **老師要的路徑百分比(§2)**:final 選 direct/decompose = HotpotQA **27.4/72.6**、2Wiki **6.5/93.5**、MuSiQue **18.2/81.8**、PopQA **3.7/96.3**(pooled 14.0/86.1)。
- **「都參考上路 evidence 是否就沒有兩條 path」的回應口徑**(§4.9 已定):union 共享的是 **evidence(檢索到的文件)**,不共享 direct path 的**答案**;direct path 自己檢索 0 次,兩條 path 的「答案生成」仍獨立,verifier 才在答案層面選路。另可用 §4(只看 decomp-routed 子集,union +0.6~+7.4 EM)佐證增益發生在 decompose 路徑內部。

**需要補的:** 只剩改投影片 —— 換 §1 的表、加 §2 的路徑比例、加上面那段口徑說明。

---

## 9. p45:both correct / both wrong 的信心度差別?是猜對嗎?這張表想講什麼?

**現況:✅ 已做完(2026-07-04)。**

- 檔案:`results/quadrant_confidence_analysis.md` §1–§2 + RESULTS_MASTER §4.12;圖 `Thesis/figures/fig_quadrant_confidence.png`;腳本 `scripts/analyze_quadrant_confidence.py`(象限 n 與 p45 精確重現 ✓)。
- **不是猜對**:both_correct 的 direct entropy pooled **0.49** vs both_wrong **1.40**(AUROC **0.782**、Cohen's d +0.89;PopQA 0.819),且四象限大多呈 both_correct → direct_only → decomp_only → both_wrong 單調上升 —— 答對的題目模型是真的有信心。
- **p45 那張表想講的**(§4.9 slide-45 敘事):兩路**互補** —— 2Wiki/MuSiQue/PopQA 是 decomposition-dominant(decOnly 361/184/179 vs dirOnly 39/31/28),HotpotQA 兩路打平(111 vs 113)所以才需要 routing;both-wrong(584/714…)是剩餘瓶頸,說明 routing 之外還需要更好的 retrieval/synthesis。這正是後面 p46 oracle gap 與 limitation 的鋪陳。

---

## 10. p47:算平均、看選錯的信心程度等細部

**現況:✅ 已做完,同一份分析涵蓋;只有一個小變體未列。**

- 檔案同上:`results/quadrant_confidence_analysis.md`。
  - **§1 平均/中位數**:四象限 × 四資料集的 direct_ent 與 decomp_max_ent mean/median 全表。
  - **§3 答錯時的信心**:confidently-wrong 比率 = both_wrong 中 entropy ≤ both_correct 中位數者,**PopQA 9.5% ~ 2Wiki 28.8%**(decomp 訊號更糟 28–48%)—— 這就是 p47「low entropy does not always mean correctness」的量化數字,建議直接放上投影片。
  - **§4 選錯的分析**:分歧象限中 verifier 選對路徑比率 vs「選 entropy 較低那條」的 naive 規則;z-score 後 naive 規則只剩 45–69%(≈丟銅板),verifier 的優勢來自 decomp prior + 規則,不是信心訊號。
- 可放上 p47 的兩個數字:**pooled AUROC 0.78**(entropy 確實與錯誤率相關)與 **confidently-wrong 9.5–28.8%**(低 entropy 不保證對)。

**唯一沒列的變體**:「verifier 選了 direct 但 direct 是錯的」那群題目的 entropy 分布(§4 只有比率沒有信心值分布)。若老師要的是這個,小改 `analyze_quadrant_confidence.py` 即可離線補,不用重跑模型。

---

## 待辦優先序總結

| 優先 | 事項 | 類型 | 成本 |
|---|---|---|---|
| 1 | p43 投影片改表 + 路徑比例(#8) | 改簡報 | 低(數據已有) |
| ~~2~~ | ~~MuSiQue gold sub-question recall(#5)~~ **✅ 已補(2026-07-05,`results/musique_gold_decomp_recall.md`)** | 新實驗 | 完成 |
| 3 | Methodology 前的 motivation 頁 + 開發集重跑 1–2 個動機分析(#3) | 敘事+小實驗 | 中 |
| 4 | SEEK vs DeepRAG 推論 wall-clock 表;(可選)timeout 曲線(#2) | 彙整+可選新實驗 | 中 |
| 5 | 「用完 SEEK 剩什麼問題」重點頁(#4) | 敘事 | 低 |
| 6 | 選錯路徑那群的信心分布(#10 變體,若老師需要) | 離線分析 | 低 |
| — | #6、#7、#9、#10 主體 | 已完成 | — |
