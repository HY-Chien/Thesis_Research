# 信心指標(Entropy)的問題總整理:高信心答錯與低信心答對

Date: 2026-07-07
目的:回應老師關於「信心指標」的整組問題(p45 / p47 / 用信心值會有什麼問題),把散在
三份分析裡的證據整合成單一敘事,含案例與分析。

來源檔案:
- `experiment/maxent_correctness/high_low_confidence_error_analysis.md`(2026-06-28,案例解剖)
- `experiment/maxent_correctness/confidence_popularity_bias.md`(2026-07-03,bias 檢查)
- `results/quadrant_confidence_analysis.md`(2026-07-04,四象限量化;圖 `Thesis/figures/fig_quadrant_confidence.png`)
- RESULTS_MASTER §4.6b(2Wiki AUROC 0.540)、§4.12

## 重要:本文混用兩套「不同實驗」的信心測量,先分清楚

| | 診斷實驗(msample) | 部署系統(dual + verifier,主方法) |
|---|---|---|
| 檔案 | `{ds}_1000_msample.jsonl` | `{ds}_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl` |
| 生成方式 | **另外跑的實驗**:每題 no-RAG 取樣 **15 次**(self-consistency),非主方法 | 主方法:每題每路徑**單次**生成 |
| 訊號 | `mean_max_ent` = 15 個 sample 各自 max token entropy 的平均;另有 `n_uniq`、`sc_agree`(答案投票分散度) | token-level entropy(單次生成內逐 token 的分布 entropy) |
| 用在本文哪裡 | §2.1 尾端統計(169/400)、五類案例、§3 低信心答對、popularity bias | §1 象限分析、§2.1 部署量化表、§4 路徑選擇 |
| 特性 | 有大量 `mean_max_ent = 0` 的離散平台(15 次全同答案) | 連續值,**沒有** entropy=0 平台(已驗證 4×1000 題 min > 0) |

訊號定義(全部「值越高 = 越不確定」):
- `mean_max_ent`:診斷實驗訊號(上表左欄)
- `direct_ent`:部署系統 direct 路徑單次生成答案的 max token entropy。
  **注意**:trace 欄位名是 `skip_probe_ent`,這是早期 skip-decompose 機制留下的舊名;
  最終方法(dual)是用 `force_direct_tau=1e9` 強制 direct 路徑必走、兩路都跑完再由
  verifier 選,**沒有 skip 機制**。論文/投影片請稱它 "direct-path answer entropy",
  不要寫 skip probe。
- `decomp_max_ent`:decompose 路徑各步 max_ent 的最大值(canonical gate 訊號)

---

## 0. 一句話結論

**Entropy 對「整體」校準有效(pooled AUROC 0.78),但它量的是「答案分布的分散度」,
不是「推理是否正確」。** 因此存在兩個系統性反例:
1. **高信心答錯** = 診斷實驗中模型 15 次取樣穩定收斂在同一個錯誤答案(deterministic
   wrong path,`mean_max_ent=0`),任何閾值都抓不到;在部署系統中對應 both_wrong 的
   9.5%(PopQA)~28.8%(2Wiki)entropy 低於 both_correct 中位數。
2. **低信心答對** = 答案分布很散但正確答案仍勝出 majority vote,或只是 span 寫法搖擺;
   其中近半(37/76)只是靠 token-F1≥0.5 過關的 metric artifact。

---

## 1. Entropy 整體上「有用」的證據(先立再破)

| 證據 | 數字 | 出處 |
|---|---|---|
| both_correct vs both_wrong 的 direct entropy | pooled 0.49 vs 1.40,AUROC **0.782**,Cohen's d +0.89 | quadrant §2 |
| 單資料集最好(PopQA) | AUROC 0.819,d +1.09 | quadrant §2 |
| 四象限單調性 | both_correct → direct_only → decomp_only → both_wrong entropy 遞增(2Wiki/HotpotQA/MuSiQue) | quadrant §1 |
| PopQA 分 popularity 層仍穩定 | 每層 AUROC(ent→wrong) 0.72–0.75,訊號不因冷門實體崩壞 | bias §5 |

→ 答對的題目模型是**真的有信心**,不是猜對(回應 p45 的問題)。

### 1.1 深入分析:0.49 vs 1.40 / AUROC 0.782 到底代表什麼(老師追問用)

以下全部在部署 dual trace 上離線重算(腳本 `scripts/deep_quadrant_entropy.py`,
2026-07-07,不需重跑模型)。

**(a) 分布百分位:中位數分得開,但尾端大量重疊**

| Dataset | both_correct p25 / p50 / p75 | both_wrong p25 / p50 / p75 |
|---|---|---|
| 2Wiki | 0.35 / 0.70 / 1.13 | 0.62 / 1.23 / 2.08 |
| HotpotQA | 0.02 / 0.17 / 0.64 | 0.35 / 0.82 / 1.73 |
| MuSiQue | 0.15 / 0.53 / 1.05 | 0.77 / 1.35 / 2.45 |
| PopQA | 0.01 / 0.15 / 0.56 | 0.50 / 0.93 / 1.70 |

pooled 中位數 0.27 vs 1.12 — 分得開。但 both_wrong 的 **p10 都低於 both_correct 的
p50**(如 HotpotQA 0.056 vs 0.171),即答錯組的低尾完全鑽進答對組的核心區。
AUROC 0.782 的正確解讀是「隨機抽一對(對,錯),錯的 entropy 較高的機率 78%」——
是**群體層次**的可分性,不是題級的判別器。

**(b) 校準曲線:entropy 五分位 → direct 路徑準確率,四資料集全部嚴格單調**

| Dataset | Q1(最低 ent) | Q2 | Q3 | Q4 | Q5(最高) |
|---|---:|---:|---:|---:|---:|
| 2Wiki | 30.5 | 25.5 | 14.5 | 14.5 | 5.0 |
| HotpotQA | 60.5 | 34.5 | 29.5 | 19.0 | 8.0 |
| MuSiQue | 22.0 | 13.5 | 7.0 | 5.5 | 3.0 |
| PopQA | 81.5 | 43.0 | 27.5 | 26.0 | 11.0 |

單調性成立 → 「答對是猜的」不成立:若是猜對,低 entropy 段不會系統性更準。
但注意**絕對值**:MuSiQue 最有信心的一段也只有 22% 對、2Wiki 30.5% —— 多跳資料集上
「最有信心」離「可信」還很遠;PopQA(單跳)才有 81.5%。

**(c) 操作點:當成錯誤偵測器用會怎樣**(規則:ent > τ 就標記為可能錯)

| Dataset | τ = bc p90(誤標對題 10%) | 能抓到的錯題 recall | τ = bc p75(誤標 25%) | recall |
|---|---:|---:|---:|---:|
| 2Wiki | 1.62 | 38.6% | 1.13 | 53.8% |
| HotpotQA | 1.00 | 44.7% | 0.64 | 61.5% |
| MuSiQue | 1.88 | 36.4% | 1.05 | 64.0% |
| PopQA | 0.95 | 49.0% | 0.56 | 71.6% |

→ 只允許誤標 10% 的對題時,**只能抓到 36–49% 的錯題**;剩下一半以上的錯答案
entropy 跟對答案一樣低。這就是 §2.1「confidently-wrong 9.5–28.8%」的另一面。

**(d) 綜合回答老師的問題**:both_correct vs both_wrong 的信心差是真實、單調、
跨四資料集一致的(不是猜對);但它是統計趨勢而非判別器 —— 在任何合理閾值下,
一半左右的錯誤逃得掉。所以 entropy 適合做 accept/abstain 的**風險排序**
(例如 selective answering、決定要不要多花檢索),不適合做 per-question 對錯判定。

但兩個重要的破口:
- **decompose 路徑的 entropy 幾乎沒有題級鑑別力**:decomp_max_ent 在每個象限都 ~3 nats
  (pooled AUROC 0.588;HotpotQA 0.553、PopQA 0.538),decomp_mean_ent 完全無訊號(0.503)。
  它只適合當**逐步檢索閘門**訊號,不適合當**整題對錯**訊號。
- **資料集級失效**:2Wiki 上 entropy→答錯的 AUROC 只有 **0.540(近隨機)**,且分位數
  rescue 率**遞減**(41.8%→29.5%,與閘門假設相反)——2Wiki 的 parametric 錯答本質上就是
  confident-but-wrong(RESULTS_MASTER §4.6b)。

---

## 2. 高信心答錯(confidently-wrong)

### 2.1 有多嚴重

**尾端統計**(來自**診斷實驗 msample**:每題另外取樣 15 次、no-RAG,非主方法單次生成;
每個資料集各取 entropy 最低的 100 題):

| Dataset | 最低 entropy 100 題中答錯 | 該尾端 Acc |
|---|---:|---:|
| 2WikiMultiHopQA | 57 / 100 | 43.0 |
| HotpotQA | 27 / 100 | 73.0 |
| MuSiQue | 48 / 100 | 52.0 |
| PopQA | 37 / 100 | 63.0 |
| ALL | **169 / 400** | 57.8 |

**關鍵解剖:這 169 個錯誤全部是 `mean_max_ent = 0, n_uniq = 1, sc_agree = 1.0`** ——
15 次 self-consistency 取樣產生**一模一樣**的錯誤答案。這不是模型猶豫,而是模型
**很穩定地停在錯的推理路徑上**。

zero-entropy 平台整體規模:1538/4000 題 entropy 完全為 0,其中 620 題(40.3%)答錯。

**在部署系統中的量化**(quadrant §3):both_wrong 裡 direct entropy 低於 both_correct
*中位數*的比率——即「錯得跟自信的對答案一模一樣、任何 τ 都抓不到」的題:

| Dataset | both_wrong n | direct 訊號抓不到 % | decomp 訊號抓不到 % |
|---|---:|---:|---:|
| PopQA | 443 | 9.5 | 47.9 |
| MuSiQue | 714 | 16.5 | 31.5 |
| HotpotQA | 584 | 18.3 | 42.8 |
| 2Wiki | 459 | 28.8 | 27.9 |

### 2.2 五類失敗型態與案例

**類型 1:停在中間 bridge entity,沒有走完最後一跳**(2Wiki / MuSiQue 最典型)

| Dataset | Question | Gold | 模型答案(15/15 一致) |
|---|---|---|---|
| 2Wiki | Where was the director of film Lettre Ouverte born? | Paris | Alex Joffe(導演名,不是出生地) |
| 2Wiki | Where was the husband of Caterina Visconti born? | Pavia | Gian Galeazzo Visconti(丈夫名) |
| MuSiQue | Who founded the company that distributed the film UHF? | Mike Medavoy | Orion Pictures(公司名,不是創辦人) |

分析:第一跳的 entity 很好找、很 salient,所以 entropy 極低;模型對一條**不完整的
推理路徑**很有信心。這是「答案路徑不完整」問題,不是「知識缺乏」問題。

**類型 2:答案型別錯誤**(comparison / yes-no 題輸出屬性而非選項)

| Dataset | Question | Gold | 模型答案 |
|---|---|---|---|
| 2Wiki | Which film came out earlier, Emmy Of Stork's Nest or Indira Vizha? | Emmy Of Stork's Nest | 1915(年份) |
| 2Wiki | Are both Kaufland and Otrag located in the same country? | yes | Germany |
| HotpotQA | Were Scott Derrickson and Ed Wood of the same nationality? | yes | American |

分析:證據可能都對,但 evidence→answer type 的映射失敗。這是 output schema failure,
支持 type-aware verifier 規則的必要性。

**類型 3:鄰近錯誤實體 lock-in(bridge drift)**

| Dataset | Question | Gold | 模型答案 |
|---|---|---|---|
| 2Wiki | Which country the director of film Daadagiri is from? | Pakistan | India |
| MuSiQue | What is the name of the city where Tess Gallagher's spouse was born in? | Clatskanie | Boston |
| PopQA | In what city was Yuriy Vernydub born? | Zhytomyr | Zaporizhya |

分析:entity prior 把模型推向一個「合理的鄰居答案」;因為該答案片語本身很熟悉,
token entropy 保持極低。這是最典型的 confident hallucination。

**類型 4:評分/alias 假錯**(official wrong 但語義不一定錯)

例:United States vs American、actor vs actress。在 169 題中 alias-aware F1 只能救回
7 題(全在 PopQA)——不主導現象,但報告時要註明。

**類型 5:span 粒度不符**(Yellowcraig vs Yellowcraigs、Roland R-8 vs The R-8 Human
Rhythm Composer)——寫 caveat 用,屬於 answer-span mismatch 而非推理失敗。

### 2.3 排除一個假設:高信心答錯是不是 popularity bias?

PopQA + Wikipedia 月流量檢查(bias 分析,n=1000):
- **沒有 raw popularity→confidence bias**:mean_max_ent vs log popularity 的
  Spearman |rho| < 0.05(n.s.),在「只看答錯題」的子集內同樣不顯著。
- zero-entropy 答錯組與 zero-entropy 答對組的 popularity 分布無差異(Mann-Whitney p=0.58)。
- 但有一個**語義層的偏斜**:熱門實體的自信錯誤 28%(13/46)是 alias 假錯
  (例:Amitabh Bachchan occupation gold=film producer、答 actor),而**冷門實體的
  自信錯誤 100%(50/50)是真錯**。
- 結論:confident hallucination 是 **deterministic-wrong-path 問題,不是 popularity
  bias 問題**;entropy 訊號在各 popularity 層 AUROC 都維持 0.72–0.75。

---

## 3. 低信心答對(low-confidence-correct)

尾端統計:entropy 最高的 100 題/資料集中,官方判對 76/400(19.0%),其中 HotpotQA
最多(44/100)。分散度特徵與高信心錯誤組完全相反:36/76 sc_agree < 0.5,只有 1 題
sc_agree = 1.0。

### 3.1 三類真實型態

**類型 1:真答案在雜訊中險勝 majority vote**

| Question | Gold | Sample 分布 |
|---|---|---|
| Where was the place of burial of the director of The Secret Of Dr. Kildare? | Forest Lawn Memorial Park | 正確 4/15,其餘是各種墓園 distractor |
| What is the latest album of Tim Hecker? | Love Streams | 正確 4/15,其餘是幻覺專輯名 |
| In what city was Antonio Calcagni born? (PopQA) | Recanati | 正確 5/15,其餘義大利城市 |

分析:entropy **正確地**反映了不確定性,只是 majority 剛好還是對的。這解釋為什麼
「高 entropy 就丟去重跑/檢索」會浪費計算——答案不一定錯。

**類型 2:yes/no 題的中間證據字串污染取樣分布**

例:「Are the directors of Back To Back and The White Hen from the same country?」
gold=yes、majority=yes,但 samples 混入 United Kingdom、USA 等中間屬性。最終 boolean
仍勝出,但 entropy 被推高。→ 高 entropy 有時反映的是**輸出格式不穩**,不是知識不確定。

**類型 3:span 粒度/表面形式搖擺**(New York vs New York City、officer vs military
officer)——token overlap 大,F1 判對,entropy 高只因寫法不穩定。

### 3.2 最大的 caveat:近半是 F1-only 假對

76 題官方判對中,只有 **39 題 exact match,37 題僅靠 token-F1 ≥ 0.5 過關**。
MuSiQue 最嚴重:10 題低信心「對」裡 **9 題是 F1-only**,且多為語義錯:

| Gold | Majority | 問題 |
|---|---|---|
| 323-272 BC | 5043 BC | 只共用 `BC`,年代全錯 |
| Lena Olin | Stig Olin | 同姓不同人 |
| Slade School of Art | Camden School of Art | 同類型 token,錯的學校 |
| orange juice | Orange Julius(HotpotQA) | 相關實體,不是題目要的成分 |

→ 論文措辭必須區分 **officially correct** 與 **semantically correct**;低信心答對組
只能宣稱前者。

---

## 4. 延伸問題:信心值能不能拿來「選路徑」?(p47 細部)

不能。在 direct_only / decomp_only 兩個分歧象限(quadrant §4):
- raw「選 entropy 較低那條」規則是退化的:direct entropy 天生尺度小,所以幾乎永遠選
  direct(direct_only 命中 90–97%,decomp_only 只有 6–25%)。
- 每資料集 z-score 校正後,命中率掉到 **45–69% ≈ 丟銅板**。
- 現行 verifier 在 decomp_only 表現好(78–96%)但在 direct_only 差(7–68%),其優勢
  來自 **decomp prior + 規則檢查**,不是信心訊號本身。

**routing implication:信心值能告訴你「系統大概對不對」(accept/abstain),
不能告訴你「哪條路徑對」。** p46 的 oracle routing gap(+3.8~+6.1 EM)無法靠比較
兩路 entropy 關閉。

---

## 5. 對 SEEK 的意涵(投影片 / 論文可直接用)

1. **MAX_ENTROPY 是有用的全域訊號,但抓不到 deterministic 推理錯誤**——entropy=0 的
   平台上仍有 40% 錯誤率,這些題任何 τ 都無效。
2. **最大的高信心失敗不是知識缺乏,而是答案路徑不完整**(停在 bridge entity / 中間
   屬性 / 型別錯誤)→ 支持 type-aware verifier 規則與 final-answer guard 的設計動機。
3. **低信心 ≠ 錯**——正解常在雜訊中勝出,或只是 span 搖擺 → aggressive routing 會
   浪費計算。
4. **信心比較不能做 direct-vs-decompose routing**(§4)→ verifier 的價值在規則,
   不在 entropy 比大小。
5. 建議 p47 直接放兩個數字:**pooled AUROC 0.78**(entropy 確實與錯誤率相關)+
   **confidently-wrong 9.5–28.8%**(低 entropy 不保證對)。
6. 論文措辭範本(已寫好,見 high_low_confidence_error_analysis.md 末段):entropy
   captures answer dispersion, not all forms of reasoning failure;應與 answer-type
   與 path-aware verification 併用,而非單獨當 correctness predictor。

## 附:唯一尚未補的小變體

「verifier 選了 direct 但 direct 是錯的」那群題的 entropy **分布**(quadrant §4 只有
比率)。若老師要,小改 `scripts/analyze_quadrant_confidence.py` 離線重放即可,
不需重跑模型。
