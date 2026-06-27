---
marp: true
theme: default
paginate: true
size: 16:9
---

<!--
進度簡報(前半段):實驗 → 對應教授方向 1 & 2。
預覽/匯出:VS Code 裝 Marp for VS Code 外掛,右上角可匯出 PDF / PPTX。
數字全部對齊 results/ 下實際檔案(2026-06 重跑;2026-06-11 套用合成 value→entity guard 後重評分)。
2026-06-23 最新增量見 `RESULTS_MASTER.md`:HotpotQA failure analysis + conservative date-comparison synthesis guard + direct-context union ablation(HotpotQA +2.5 EM,2Wiki +3.0 EM)。D3/D4 本簡報不談。
-->

# SEEK:以不確定性驅動的自適應檢索與分解

### 進度報告(前半段)— 實驗與發現

聚焦教授指導 **方向 1(校準 / confident hallucination)** 與 **方向 2(動態檢索 / 分解的正當性)**

2026-06 · 主結果與分析皆在 `main_methodology/`

---

## 教授四方向 → 本次聚焦

| 方向 | 內容 | 本次 |
|---|---|---|
| **1 校準** | 高信心 ≠ 答對;要抓出 confident hallucination、找單一 SC 以外的訊號 | ✅ 主力 |
| **2 動態檢索正當性** | 實證「分解 + 動態檢索」比 always-retrieve 更精準(非速度論) | ✅ 主力 |
| 3 推理鏈脆弱性 | 生成中插檢索打斷推理鏈 | ⏸ 本簡報略 |
| 4 Agent Tool Use | 把校準訊號推廣到工具呼叫決策 | ⏸ 本簡報略 |

> 主軸:用**生成時的不確定性(entropy)**決定「要不要拆解、要不要檢索」。本簡報用實驗證明它**何時有用、何時失效**。

---

## 背景:SEEK 的兩個決策

對每個問題,SEEK 依**原問題直答的 token 熵**做兩層決策:

```
        ┌─ 熵低(有信心) ─→  直接單步作答          ← 決策 A:該不該「拆解」
原問題 ─┤
        └─ 熵高(沒把握) ─→  拆解成 sub-questions
                                  └─ 每個 sub-q 再依熵決定要不要「檢索」 ← 決策 B
```

- 訊號:`max_ent`(輸出序列中最猶豫那個 token 的熵)。
- 本簡報順序:先證 **拆解對檢索有益(方向2)**,再證 **熵訊號的能力與極限(方向1)**。

---

## 實驗設定(canonical SEEK 配置)

| 元件 | 設定 |
|---|---|
| Base model | Llama-3-8B-Instruct |
| Decomposer | LoRA-SFT,**12K 三來源** gold(MuSiQue + 2Wiki + PopQA);rank 16 / all-linear / lr 1e-4 / cosine / 3 ep |
| 推論模式 | **Mode C**(LoRA 全程開;最終 synthesis 步 LoRA-**OFF** = synthOFF 修正) |
| 不確定性訊號 | `max_ent`(輸出序列中最猶豫 token 的熵) |
| Gate | 拆解 gate **τ=0.3**;skip-decompose router τ ∈ {0.5, 0.8, 1.2} |
| Retrieval(honest) | **wiki-21M**(psgs_w100)BM25,top_k=**8**(PopQA 單跳用 top-3) |
| Retrieval(distractor) | per-question 20 候選段落 BM25(IRCoT / Adaptive-RAG protocol) |
| 合成後處理 | **value→entity guard**(comparison 題合成回「值」時映射回實體;2026-06-11)+ **date-comparison guard**(older/younger/earlier/later 日期比較;2026-06-23 offline replay) |
| 評估 protocol | dev 前 1000、**不 shuffle**(對齊 DeepRAG);maxent 校準另跑 4×1000 |

> 對照 **DeepRAG-8B**:full DPO + SC-trained gate;SEEK = LoRA-SFT decomposer + **訓練-free 熵 gate**(約 ¼ 訓練量)。

---

## 主結果:先把「我們在哪」講誠實

兩種 retrieval pool,結論天差地別:

EM / F1(honest 欄 = 最佳統一配置 k=8;PopQA 單跳用 top-3):

| Dataset | distractor pool（必含 gold） | **honest wiki-21M**（= DeepRAG 同 pool） | DeepRAG | Δ(honest−DeepRAG) |
|---|---|---|---|---|
| **PopQA** | 48.7 / 51.2 | **50.7 / 52.9** | 42.5 / 47.8 | **+8.2 / +5.1** ✅ 贏 |
| **MuSiQue** | 44.3 / 53.7 | **22.7 / 30.5** | 22.5 / 30.4 | **+0.2 / +0.1** ≈ 平 |
| **2Wiki** | 65.2 / 73.1 | **45.6 / 51.5** | 48.1 / 53.3 | **−2.5 / −1.8** 接近 |
| HotpotQA | 41.9 / 53.5 | **25.7 / 33.9** | 40.7 / 51.5 | −15.0 / −17.6 ❌ |

- distractor pool 把多跳灌水(2Wiki +20、MuSiQue +22、Hotpot +16 EM)。
- **同 pool 下:PopQA 贏、MuSiQue 平,2Wiki 已逼近(−2.5)、HotpotQA 大輸**——不是全輸,但也非排行榜全勝。
- 2026-06-11:合成 **value→entity guard** 修掉「比較題回值不回實體」(2Wiki +2.7 EM,零退步)。
- → 貢獻不是排行榜,而是「**何時 / 為何**」的機制理解。

---

## 方向2 · 拆解讓檢索更準(M1 + M3)

**M1**(強制單步直答)= 不拆解地基:2Wiki 21.1 / Hotpot 46.8 / MuSiQue 22.6 / PopQA 31.1　**M3**:同預算下比「原問題」vs「拆解 sub-q」撈 gold 段落的能力 ↓

![w:840](figures/fig_m3_recall.png)

> 📎 `examples/ex3`:*"child of Caroline LeRoy's spouse?"* — 原問題沒有 "Daniel Webster",單一 query 撈不到他兒子那篇;**拆解揭露 bridge entity 後** sub-q 才命中 gold。

---

## 方向2 · 但拆解不是免費的(M4)

2×2 交叉表:有一格是 **dirOnly**(直答對、一拆反而錯)。

| dirOnly 題數 | 2Wiki | HotpotQA | MuSiQue | PopQA |
|---|---|---|---|---|
| 直答對→拆解錯 | 83 → **39** | 197 → **180** | 65 | 39 |

> 📎 範例 `examples/ex2_dironly_decompose_breaks.md`:
> *"Which film came out first, Blind Shaft or The Mask of Fu Manchu?"*
> 直答 ✓;拆解卻在**合成步**回了被比較的值 `1932` ✗。
> **這一類已被修掉**:value→entity guard 確定性映射回實體(2Wiki 83→39,修掉的 44 題全是此型)。

**含意:** 可確定性修的已修;**剩下的 dirOnly 才是真 trade-off** → 需要一個 router 來決定。

---

## 方向2 · 強制單步消融:拆/不拆是 dataset-dependent(M2 組A)

honest pool,k=8,把所有題**強制單步(不拆)**:

| Dataset | 拆解 | 強制單步 | Δ |
|---|---|---|---|
| HotpotQA | 25.7 | **30.3** | **+4.6**(單步反而贏) |
| 2Wiki | 45.6 | 18.0 | **−27.6**(崩盤) |
| MuSiQue | 22.7 | 10.2 | **−12.5**(崩盤) |

- 真多跳(2Wiki/MuSiQue):**不拆 = 災難**。
- HotpotQA:很多題其實**單步可答**(第二跳實體常already在問題字面)→ 拆解反而有害。
- → 沒有單一策略全贏,**正是 router 該登場的地方**。

---

## 方向2 · 自適應 router 與它的極限(M2 組B)

依原問題熵 `< τ` 就 skip 拆解(honest wiki-21M）:

![w:730](figures/fig_router_tau.png)

- ✅ **HotpotQA:router 33.0 > 全單步 30.3 > 全分解 25.7** — 混合勝任一純策略,query 級熵帶真信號。
- ❌ 其他三個**單調**:一 skip 就降,最佳是完全不 skip。
- **殘酷核心:單一全域 τ 救不了**(Hotpot 要高 skip、其他要零)→ router 是「為何樸素路由不夠」的**分析貢獻**,非排行榜勝利。

---

## 方向1 · 熵能不能預測對錯?(maxent 校準)

訊號 `mean_max_ent`,4×1000 題,對錯 = SC majority vs gold。

![w:560](../../experiment/maxent_correctness/figures/fig1_main_maxent_vs_accuracy.png)

| | 答對均熵 | 答錯均熵 | AUROC |
|---|---|---|---|
| Pooled | **0.268** | **0.602** | 0.686 |
| PopQA(最強) | 0.228 | 0.663 | **0.741** |

分桶準確率 **~58% → 13% 單調下降**,四資料集方向一致 → **熵在整體層級是堪用的信心訊號**。

---

## 方向1 · 合題:正(maxent)× 負(router)= confident hallucination

兩個結果不是打架,而是同一刀的兩面:

- **maxent(正面)**:平均而言,低熵 → 較可能答對(AUROC 0.69~0.74)。
- **router(負面)**:但在**真多跳的低熵尾端**,模型是**自信地錯** → 一 skip 就壞。

> 📎 範例 `examples/ex1_confident_hallucination.md`:
> *"...paternal grandfather of Alexandru Coconul?"* 熵 **0.002**(幾乎零猶豫)→ 答 `Radu Mihnea`,實際是 `Mihnea Turcitul`。**自信、流暢、錯。**

**這正是教授方向 1 的實證:** 高信心 ≠ 正確;單一熵/SC 分數會在這個尾端失效 → 需要更強的 confident-hallucination 偵測(接 B1/B2/B3)。

---

## 前半段小結 → 後半段:放大那個「失效尾端」

**前半段(以方向 2 為主)已證:**
- 拆解對真多跳**非做不可**(組A);拆解 → **檢索更準**(M3);但 routing 有**範圍極限**(M2 組B)
- 熵在**整體層級**可預測對錯(maxent),**但在低熵尾端失效** → router 一 skip 就壞

**後半段(方向 1):放大那個低熵失效尾端 ——**

> 那些「**低熵卻錯**」的題到底是什麼?能不能用單一 SC / 熵**以外**的訊號抓出來?
> → confident hallucination 的 **B1/B2/B3 解剖 + 偵測器**(教授最在意的「測謊機」)。

---

## 方向1 · 問題:熵在「最該決定的地方」用盡了

教授原話:**高 SC / 高信心 ≠ 答對。** 用數據說話:

| 任務 | `max_ent` 的 AUROC |
|---|---|
| 全體題目:預測「對 / 錯」 | **0.735**(堪用) |
| **已自信子集**:分辨「自信對 / 自信錯」 | **0.488 ≈ 亂猜** |

- 在**整體**上,低熵 → 較可能對;但一旦進入**低熵子集**,熵就**沒有鑑別力了**。
- → 這正是 **router 低熵尾端必敗的機制**:gate 想在那裡決定「可不可以 skip」,但訊號在那裡剛好用完。
- 接下來解剖:那個尾端的「自信錯誤」到底是什麼。

---

## 方向1 · 不是單一現象:B1 / B2 / B3

confident hallucination 是**三種**解析失敗的混合,各有偵測器、各有不同「RAG 能不能救」:

| 子型 | 機制 | RAG 能救嗎 | 偵測器 |
|---|---|---|---|
| **B1 知識錯** | schema 解析對,但事實信念錯(如生年錯) | ✅ 補知識可救 | 信心訊號**看不出**(自信且一致)→ 需外部知識 |
| **B2 鏈斷/隨機填** | 第一跳實體不穩、推理鏈斷 | ⚠️ 檢索對才救得到 | multi-sample `n_uniq` 最佳 |
| **B3 角色混淆** | 把問題的**約束實體**當成**答案實體** | ❌ 救不了,要改 prompt | role-overlap × schema(句法角色) |

> 註:舊的「A1 = parametric override / 無視輸入」假說已被 2026-05-07 擾動實驗**證偽**——看似 A1 的其實是 B1。

---

## 方向1 · 子型分佈(GPT-OSS judge, 998 題)

![w:880](figures/fig_subtype_dist.png)

- **B2(鏈斷/隨機填)是大宗**(443);B1(知識錯)178;**B3 罕見(6)但卻是 token/SC 完全看不到的那種**。
- 含意:沒有「單一修法」能一次解決——不同子型要不同對策。

---

## 方向1 · 範例:B3 角色混淆(最隱形)

```
Q   : What director worked with Vikram Bhatt on a film starring
      Rajneesh Duggal and Adah Sharma?
GOLD: Tinu Suresh Desai
PRED: Vikram Bhatt   ← 把問題裡的「約束實體」當成答案
SC = 13.16(很高) · 20 取樣 EM = 0/20 · 每個 token 都自信
```

> 📎 `examples/ex4_b3_role_confusion.md`

- SC 高、token 全自信 → **任何信心/熵訊號都失靈**。
- 只有 **role-overlap × schema**(spaCy 判 CONSTRAINT/SUBJECT/DESCRIPTOR)抓得到 → **這就是教授要的 QA-native 指標**。

---

## 方向1 · 偵測器:沒有單一訊號夠用

![w:760](figures/fig_detector_auroc.png)

- SEEK gate 用的 **`max_ent` 在自信子集 ≈ 0.49(亂猜)**;熵/margin 系都 ~0.58。
- first-entity log-prob 0.74–0.76、**multi-sample `n_uniq` 0.81** 才較有力(主抓 B2)。
- **結論:需要三件套**(`n_uniq` + first-entity min-lp + role-overlap×schema),單一分數不可能。
- 誠實:pilot 的 `min_margin` AUROC 0.96 是 **n=5 小樣本雜訊**,100Q 掉到 0.58 → **不引用**。

---

## 方向1 ↔ 方向2 · 機制閉環:為何 router 必敗

- router 的單一熵門檻,天花板就是上頁 `max_ent` 0.49 的**同一道牆**。
- 低熵尾端混著 **B1(自信、一致、知識錯)** → gate 判「可 skip」→ 直接給出自信的錯答案。
- → M2 組B「一 skip 就壞」**不是 gate 沒調好,而是這個尾端原則上無法用熵分**。
- **推論(指向後續方向):** 要救多跳,得靠 **RAG 補 B1 知識** + **query 品質補 B2 檢索**,**不是調 gate**。

---

## 方向2 · 怎麼判斷 verifier「要不要拆解」這個決策是對是錯?(反事實 oracle)

範圍先講清楚:這頁評的是 **verifier 的「要不要拆解」選擇**——不是「要不要 RAG」。
direct 與 decompose **兩條都檢索**(direct = 原問題單跳 RAG;decompose = 拆解後多跳 RAG),差別只在拆不拆。

**設計上握有反事實:每題都跑完 direct(單跳RAG)＋ decompose(多跳RAG)兩條,verifier 才事後二選一**,所以在「**兩條答案不一致**」的題上,可把 verifier 的選擇標成對/錯:

| 1000 題 | 兩條都對 | **dirOnly**(單跳對·拆解錯) | **decOnly**(該拆解) | 兩條都錯 | **verifier EM** | **oracle EM** | gap |
|---|---|---|---|---|---|---|---|
| HotpotQA | 152 | 151 | 98 | 599 | 31.8% | 40.1% | **+8.3** |
| 2Wiki | 125 | 55 | 347 | 473 | 46.9% | 52.7% | **+5.8** |
| MuSiQue | 65 | 37 | 182 | 716 | 24.6% | 28.4% | **+3.8** |
| PopQA | 332 | 39 | 183 | 446 | 51.8% | 55.4% | **+3.6** |

> verifier = 跑完兩條再事後選(**不是**事前用熵直接決定拆不拆——那個 a-priori router 我們證明會輸,見 M2 組B)。
> oracle = 每題都選對那條的上界;**gap = verifier 拆解選錯而損失的 EM**。

---

## 方向2 · 選錯都選在同一邊:verifier 不敢「不拆解」

把選擇拆成兩種命中率(在不一致的題上選對的比例):

| | 抓到「該拆解」decOnly | 抓到「該走單跳」dirOnly |
|---|---|---|
| HotpotQA | **91%**(89/98) | **51%**(77/151) |
| 2Wiki | **95%**(331/347) | **24%**(13/55) |
| MuSiQue | **94%**(171/182) | **27%**(10/37) |
| PopQA | **97%**(178/183) | **21%**(8/39) |

- **verifier 極會認「該拆解」(91–97%),卻幾乎認不出「單跳就夠、拆了反而錯」(21–51%)** → 選擇**系統性偏向拆解**。
- 殘餘 oracle gap(+3.6~+8.3)**幾乎全卡在 dirOnly 那格** —— 正是**低熵、自信卻錯**的尾端。
- → 三條線在此接起來:**verifier 選錯 = dirOnly 格 = confident-wrong 尾端 = `max_ent` 在自信子集 AUROC 0.49 那道牆**。

> **界定清楚(回老師):** 這頁回答的是「verifier 的拆解選擇對不對」,**不是**「模型答題需不需要 RAG 是對是錯」(那是 retrieve-or-not,由三方 never/gate/always 回答)。結論:verifier 對「該拆解」判得準、對「不該拆解」判不準,而後者那格原理上就是熵分不開,**不是調 threshold 能救**。

---

## 限制與誠實定位

- **Pool:** honest wiki-21M 上 PopQA 贏、MuSiQue 平,2Wiki 接近、HotpotQA 仍是主缺口;6/23 date-comparison guard 只小幅修 synthesis(HotpotQA offline +0.3 EM),大缺口仍是 over-decomposition / retrieval drift。
- **B3 樣本少(n=6):** 罕見,結論需更大樣本佐證。
- **子型標註是 GPT-OSS judge(998 題)**,非人工 → 需抽樣人工複核。
- **maxent 正相關屬 SC / 整體層級**;低熵尾端的反例(confident-but-wrong)**已主動揭露,不掩蓋**。

---

## 貢獻總結(對應教授方向 1 + 2)

- **方向2(動態檢索/分解的正當性):**
  拆解對真多跳**非做不可**(組A)、拆解 → **檢索更準**(M3)、routing 的**範圍與極限**(M2)、**pool 誠實校正** vs DeepRAG。
- **方向1(校準 / 測謊機):**
  confident hallucination 解剖成 **B1/B2/B3**,各有偵測器;**B3 role-overlap×schema 是 QA-native 新指標**(教授指定方向),並證明**單一 SC/熵不可能**。
- **一句話:** 把「不確定性訊號**何時有用、何時失效、失效時怎麼辦**」畫成一張完整地圖。

---

## 待辦 & 時程(7 月口試)

**收尾(多為半離線,不吃大 GPU):**
1. 把 **B1/B2/B3 偵測器直接套到 SEEK skip 子集** → 閉合「router 負面 ↔ confident hallucination」最後一哩
2. ✅ **Oracle 上界(已做)**:verifier 拆解選擇的反事實 2×2 + oracle gap(+3.6~+8.3 EM)已算出;殘餘增益全卡在 dirOnly(該走單跳卻拆解)格,證明「gap 來自熵分不開的尾端,非調 threshold 能救」→ 待補:逐題質化讀
3. ✅ **HotpotQA 質化讀 / failure analysis(已做 6/23)**:direct-only、decomposition/retrieval/synthesis 三類與 3 個代表案例已整理;date-comparison synthesis guard 已實作
4. **honest pool 的 query 改寫 / direct-context union**:下一個真正可能推動 HotpotQA 的槓桿

> D3(推理鏈脆弱)/ D4(Agent Tool Use)本階段暫不納入,先把 D1 + D2 做紮實。
> 範例:`slides/examples/ex1~ex4`;圖:`slides/figures/`;數字:`results/` 與 `deeprag_compare/check/`。
