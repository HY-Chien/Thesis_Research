---
marp: true
theme: default
paginate: true
size: 16:9
---

<!--
老師回饋 #5(gold 拆解 / base 模型拆解)+ #6(direct-gate baseline)投影片稿:5 頁主線 + 3 頁 HotpotQA backup。
建議插入位置:頁 1–3 接在 p38(拆解 recall)之後;頁 4–5 接在 p41(gating)之後;backup 三頁放附錄,
主線講到 HotpotQA 例外時一句帶過、被追問再翻(B1 機制+檢索層 → B2 端到端證據 → B3 修復+剩餘差距)。
數字來源:results/musique_gold_decomp_recall.md(含 2026-07-06 同子集條件化修正)、
results/direct_gate_ablation.md、RESULTS_MASTER §4.1(oracle 上界)/ §4.3–4.4(M3)/
§4.6(step-level AUROC、never-gate-always)/ §4.11、results/direct_decompose_2x2_current_final.md、
results/hotpotqa_failure_analysis.md。
壓頁方法:頁 2 + 頁 3 合併(留條件化 hit@5 一行 + base 步數分布一行);頁 5 可併入頁 4 只留鏡像結論一句。
-->

## 拆解檢索的天花板在哪?— 用 MuSiQue gold 拆解當 oracle

**問題:SEEK 的拆解離「完美拆解」多遠?**(MuSiQue 官方附 gold sub-questions,可直接對照)

同 p38 口徑:honest wiki-21M、path-aware、answer-containing hit@k;N=1000(**全 2-hop**)。

<small>註:本 oracle 實驗只能做在 MuSiQue(四個資料集中唯一附 gold sub-questions);而 canonical protocol「dev 前 1000、不 shuffle」(對齊 DeepRAG)在按題型排序的 MuSiQue dev 上恰好全為 2-hop。>2-hop 的覆蓋由主表的 2Wiki 承擔(bridge_comparison 4-hop,228/1000)。2-hop 單一 bridge 使下一頁的條件化分析因果最乾淨;hop 更多時 bridge 錯誤逐跳複利,差距預期更大,但 +17pp headroom 僅主張於 2-hop。</small>

| hit@5 (%) | 說明 |
|---|---|
| 原問題直接查 **19.0** | 不拆解的地基 |
| SEEK 拆解 **38.8** | 可部署(代入**預測的**中間答案) |
| **gold 拆解 55.8** | oracle 拆解 + oracle 中間答案 |
| gold 拆解**不代入**中間答案 **19.8** | 保留 `#1` 佔位符 → **≈ 原問題** |

- **p38 主張被 oracle 強化**:拆解 38.8 ≫ 不拆 19.0;完美拆解鏈還有 **+17pp headroom**。
- **關鍵對照 `gold_raw ≈ orig`**:2-hop 的最終答案幾乎只能靠 step-2 查詢(`<bridge> >> relation`)撈到——**拆解的增益來自「拆解+逐步作答」這條鏈,不是拆解文字本身**。

---

## 差距 17pp 在哪?— 中間答案,不是拆解文字

兩個隔離實驗,把「拆解文字」和「中間答案」分開:

| arm(hit@5) | 拆解文字 | 中間答案 | 結果 |
|---|---|---|---:|
| seek_goldres | **SEEK** | gold | **54.1 ≈ gold 55.8** |
| seek(可部署) | SEEK | 預測 | 38.8 |

SEEK step-1 中間答案 EM 只有 **42.4%**(vs gold)。條件化(**gold 用同子集算,避免選擇偏誤**):

| SEEK step-1 中間答案 | n | seek hit@5 | gold hit@5(同子集) |
|---|---:|---:|---:|
| ✅ 答對 | 424 | 56.4 | 58.5(**seek ≈ gold**) |
| ❌ 答錯 | 576 | **25.9** | **53.8**(gold 幾乎不掉) |

> **seek 與 gold 的差距幾乎完全由中間答案錯誤解釋**:答錯子集上題目並沒有變難
> (gold 仍檢索得到 53.8),是 bridge entity 答錯 → step-2 查詢跟著歪(**retrieval drift**),
> seek 崩到 25.9。拆解文字(KG-triple 風格)與 gold 同構、不是瓶頸;
> 與 failure taxonomy(retrieval_gap_both 68%)互相印證;瓶頸 = 中間答案正確率,這也是最自然的 future work。

---

## 為什麼要 train decomposer?— 未微調 base 對照

同 base(Llama-3-8B-Instruct)、同 prompt、greedy,只差**沒掛 LoRA**(題目全是 2-hop):

| | SEEK decomposer(LoRA) | 未微調 base |
|---|---|---|
| 拆成正確的 2 步 | **963 / 1000** | **149 / 1000**(多數拆 3–11 步) |
| 格式解析失敗 | ~0 | **153 / 1000(15.3%)**→ decompose path 直接失效 |
| 查詢風格 | KG-triple 短查詢,BM25-friendly | 長自然語言、過度拆解、偶帶(可能錯的)parametric 實體 |
| oracle 中間答案下 hit@5 | 54.1(seek 文字) | 52.9 †(還多吃 **1.5×** 檢索預算:平均 2.98 步) |

- **訓練買到的不是「查詢文字更會檢索」**(oracle 下 base 52.9 ≤ seek 54.1 ≤ gold 55.8,差距小),
  **而是步數/格式的結構可靠性**:步數對 → 要答對的中間答案少、drift 不逐跳複利;且推論成本低。
- 完整階梯(hit@5):不拆 **19.0** → 可部署 SEEK **38.8** → base+oracle **52.9** ≤ seek 文字 **54.1** ≤ gold **55.8**。

---

## 拆解可以被 confidence gate 取代嗎?— direct-gate baseline

**Baseline:不拆解**,同一條 gate 規則直接套在原問題:`max_ent < τ` → 用 parametric 答案,否則檢索原問題後重答。設定與主表完全一致;離線 replay,τ=0 精確重現 direct-always ✓。**對 baseline 最寬容:讓它事後挑最佳 τ。**

| EM(1000Q) | Dir-never | Dir-gate **best-τ** | **Dec-gate(SEEK)** | Δ |
|---|---:|---:|---:|---:|
| 2Wiki | 12.6 | 18.6 | **47.2** | **+28.6** |
| MuSiQue | 5.9 | 10.8 | **24.7** | **+13.9** |
| PopQA | 22.4 | 39.2 | **51.5** | **+12.3** |
| HotpotQA * | 18.2 | 30.7 | 25.0 | −5.7 |

- **不是訊號不好,是天花板太低**:query-level AUROC 0.72–0.81(訊號可用),但 direct-gate 對 direct-always 只擠出 ≤ +2.1 EM——不拆解時 gate 只是在「parametric」和「單 query 檢索」**兩個常常都錯的候選**間二選一。**把候選變好的是拆解,不是 gate。**
- 連 parametric-only 都是拆解贏(Dec-never 2Wiki 29.4 vs 12.6):逐步答小問題更能引出 parametric 知識。
- \* HotpotQA 例外與既有敘事一致(第二跳共享 surface entity、單 query 已 cover);**full SEEK 31.8 ≥ 30.7**,verifier 路由把 direct 優勢吃回來。

---

## 那訊號本身呢?— query-level 可用、step-level 分化,系統為何照樣贏

**AUROC 在算什麼**:每題一個「分數 vs 標籤」對 —— 分數 = 熵(不確定性)、標籤 = 該題 parametric 答案**錯**(EM=0)。AUROC = 隨機抽一錯一對兩題,熵把錯的排前面的機率;**0.5 = 亂猜、1.0 = 完美**。它衡量「訊號能不能把該檢索的題挑出來」,與 τ 選在哪無關。

| 層級 | 分數(每題) | 標籤(每題) |
|---|---|---|
| **query-level** | 原問題**直答**的 `max_ent` | 該直答錯 |
| **step-level** | 拆解後各 step sub-answer `max_ent` 取 **max**(鏈上最沒把握的一步) | 該題 never-RAG 最終答案錯 |

| Dataset | **query-level**(原問題直答) | **step-level**(拆解後 sub-answer) |
|---|---:|---:|
| HotpotQA | 0.813 | 0.696 |
| 2Wiki | 0.781 | **0.540(近隨機)** |
| MuSiQue | 0.716 | **0.809** |
| PopQA | 0.748 | 0.749 |

- 拆解後訊號**不是一律變差,是 dataset-dependent**:MuSiQue 反而升;**2Wiki 崩到近隨機** —— sub-question 的 parametric 錯答是 **confident-but-wrong**(低熵卻錯),與方向 1 的 confident hallucination 是同一件事。
- **訊號變弱為何不傷系統?因為拆解後 gate 的角色變了**:拆解後檢索幾乎必要(never→always **+9.3~+27.3 EM**,§4.6),正確策略就是「預設檢索」——gate τ=0.3 檢索率 **90.8–99.3%**、EM 與 always 差 **±0.3**。**gate = 效率論證(省 0.7–9.2% 檢索),不是準確率論證**;訊號弱只少省一點成本。
- **鏡像結構(一句話總結兩頁)**:direct path **訊號好但候選爛**(天花板低,gate 無用武之地);decompose path **候選好但訊號弱**——好在這裡的 gate 不需要準。訊號真正需要準的地方是「整題走哪條路」的 **query-level 路由(0.72–0.81,可用)= verifier 的工作**。

---

## (Backup 1/3)HotpotQA 例外:機制 + 檢索層證據(M3)

**機制假說**:HotpotQA 由標註者**看著兩篇相連段落寫問題** → 兩篇文章的表面詞彙都嵌在題目裡,原問題本身就是很強的 BM25 query;KG-triple 子查詢反而**丟失原題的限定條件**(failure taxonomy:`constraint_loss` 佔 HotpotQA decompose 錯誤 **10%,幾乎專屬** —— 2Wiki 1% / MuSiQue 2%)。對照組 MuSiQue 刻意反捷徑建構(過濾可單 query 走捷徑的題),第二跳文章與原問題零字面共享。

**M3 實驗怎麼做**:同一題、同 k 預算,比兩種 query 撈 gold 段落(`is_supporting`)的能力 —— A = 原問題單 query top-k;B = 每個 sub-question best-match(同預算,公平)。distractor pool(per-question 20 候選,BM25)。

| Δ(B−A)recall@5 | HotpotQA | 2Wiki | MuSiQue |
|---|---:|---:|---:|
| 拆解帶來的增益 | **+5.7(最小)** | +19.7 | **+21.9** |
| A(原問題)本身 | **81.4(已很高)** | 75.7 | 56.1 |

→ 增益排序 = 「第二跳與原問題字面距離」排序。旁證:HotpotQA k=8 飽和(k8≈k10)、entity-only query rewriting ~0 —— 瓶頸是 query 文字,不是撈的深度。

---

## (Backup 2/3)HotpotQA 例外:端到端證據(三個獨立實驗怎麼做)

**① 強制單步消融**(`--skip-decompose`,同 pipeline、把原問題當唯一 step,只關拆解這一個變因):

| ΔEM(single − decompose) | HotpotQA | 2Wiki | MuSiQue |
|---|---:|---:|---:|
| | **+4.6(唯一贏)** | **−27.6(崩)** | −12.5 |

**② forced-path 2×2 + oracle**(跑兩條 forced run:全 direct / 全 decompose,逐題以 gold EM 分四象限;oracle = 每題挑對的那條 = 路由上界):

| HotpotQA(1000Q) | bothC | **dirOnly** | decOnly | bothWrong | oracle |
|---|---:|---:|---:|---:|---:|
| | 152 | **141–151** | 95–98 | **~600** | **39.8–40.1** |

其他三資料集 decOnly ≫ dirOnly(如 2Wiki 326 vs 50)—— 唯 HotpotQA 反向。

**③ direct-gate baseline**(前頁):30.7 > dec-gate 25.0,唯一例外。

**錯誤分解**(failure analysis,final wrong 的組成):both-wrong **599 = 87.8%**、router 漏掉 direct-only 74 = 10.9%、漏掉 decomp-only 9 = 1.3% → **路由只能再賺 ~7 EM,大頭在 both-wrong**。另:verifier 結構性偏 decompose(decomp-only 抓 91–95%、direct-only 只 24–51%)—— 對 2Wiki/MuSiQue 是正確取捨,對 HotpotQA 是反的。

---

## (Backup 3/3)HotpotQA 例外:機制理解 → 對症修復 → 剩餘差距

**修復弧線(因果順序:先診斷、後設計)**:

1. **verifier 路由**:HotpotQA 正是 query-level entropy routing 真正有效的資料集(router 33.0 > all-single 30.3 > all-decompose 25.7;對照 2Wiki/MuSiQue 任何 skip 都傷)。verifier 31.8,同時贏過兩條純路徑。
2. **original-question evidence union**:每個 sub-question 的 evidence = sub-q top-k **∪ 原問題 top-k**(去重),把原題的 disambiguating 線索還給 decompose path —— 正對 `constraint_loss` 機制。效果:HotpotQA decomp **25.0 → 30.5**(+5.5,四資料集中最大,機制預測吻合;2Wiki +3.0),2×2 變 **111 ≈ 113**,6/25 主系統 **35.5**。

**剩餘差距為什麼不是方法問題**:oracle 路由上界 **39.8 ≈ DeepRAG 40.7** —— 就算 verifier 全知也只是追平。剩下的 both-wrong ~600 題已釘死在 **BM25 第二跳 lexical 上限**(§4.4:單篇命中天花板 87%、full-coverage 遠低;entity rewriting ~0、title-field 已修、k 已飽和 —— 能排除的都排除了)→ 需要 dense retrieval 等檢索升級,列 future work。

**定位句**:不主張 HotpotQA 贏;主張的是知道**為什麼**不贏 —— 且這個理解直接產出了 +10.5 EM 的修復(25.0 → 35.5)。
