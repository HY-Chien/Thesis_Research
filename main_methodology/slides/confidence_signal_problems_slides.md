---
marp: true
theme: default
paginate: true
size: 16:9
---

<!--
信心指標問題總整理投影片。
來源:results/confidence_signal_problems.md(2026-07-07)。
數字出處:results/quadrant_confidence_analysis.md、
experiment/maxent_correctness/{high_low_confidence_error_analysis,confidence_popularity_bias}.md、
scripts/deep_quadrant_entropy.py。
預覽/匯出:VS Code Marp 外掛 → PDF / PPTX。
-->

<style>
section { font-size: 25px; }
table { font-size: 0.8em; line-height: 1.25; }
blockquote { font-size: 0.85em; }
</style>

# 信心指標(Entropy)的能力與極限

### 高信心答錯、低信心答對:量化 + 案例分析

回應老師問題:p45(both correct/wrong 信心差別、是猜對嗎?)
p47(低 entropy 不保證對)、「用信心值會有什麼問題」

2026-07 · `results/confidence_signal_problems.md`

---

## 先分清楚:兩套「不同實驗」的信心測量

| | 診斷實驗(msample) | 部署系統(dual + verifier,主方法) |
|---|---|---|
| 生成方式 | 每題 no-RAG **取樣 15 次**(self-consistency)| 每題每路徑**單次**生成 |
| 訊號 | `mean_max_ent`(15 個 sample 的 max token entropy 平均)+ 答案投票分散度(`n_uniq`, `sc_agree`) | token-level entropy(單次生成) |
| 特性 | 有大量 **entropy = 0** 的離散平台(15 次全同答案) | 連續值,**沒有** 0 平台 |
| 用途 | 解剖失敗案例(為什麼錯) | 象限分析、routing(部署行為) |

> ⚠️ direct 路徑訊號的 trace 欄位名 `skip_probe_ent` 是早期 skip 機制的**舊名**;
> 最終方法**沒有 skip**,兩路都跑、verifier 選。正名:**direct-path answer entropy**。

---

## 一句話結論

**Entropy 量的是「答案分布的分散度」,不是「推理是否正確」**

整體校準有效(pooled AUROC 0.78),但存在兩個系統性反例:

<br>

| 反例 | 機制 | 規模 |
|---|---|---|
| **高信心答錯** | 模型**穩定收斂在同一個錯誤答案**(deterministic wrong path),15 次取樣全同、entropy=0 → 任何閾值都抓不到 | both_wrong 中 **9.5%(PopQA)~28.8%(2Wiki)** |
| **低信心答對** | 答案分布散但正解仍勝出投票;或只是 span 寫法搖擺 | 最高 entropy 尾端仍 19% 判對;其中 **37/76 只是 F1 假對** |

---

## Entropy 有效的證據:答對不是猜的(p45)

**direct-path answer entropy(部署 trace;四象限 mean,越高越不確定)**

| Dataset | both_correct | direct_only | decomp_only | both_wrong | AUROC(bc vs bw) |
|---|---:|---:|---:|---:|---:|
| 2Wiki | 0.80 | 0.83 | 1.23 | 1.50 | 0.688 |
| HotpotQA | 0.41 | 0.48 | 1.06 | 1.15 | 0.750 |
| MuSiQue | 0.74 | 1.12 | 1.51 | 1.69 | 0.752 |
| PopQA | 0.35 | 1.19 | 0.80 | 1.19 | **0.819** |
| **ALL** | **0.49** | **0.74** | **1.18** | **1.40** | **0.782** |

- 整體趨勢:both_correct → direct_only → decomp_only → both_wrong entropy 遞增
  (2Wiki/HotpotQA/MuSiQue 明顯;PopQA 的 direct_only n=28,不完全單調)
- 但 decompose 路徑訊號**幾乎無鑑別力**:decomp_max_ent 每象限都 ~3 nats(AUROC 0.588)

---

## 深入(a):是統計趨勢,不是題級判別器

**分布百分位:中位數分得開,尾端大量重疊**

| Dataset | 對:p25 / p50 / p75 | 錯:p25 / p50 / p75 |
|---|---|---|
| HotpotQA | 0.02 / 0.17 / 0.64 | 0.35 / 0.82 / 1.73 |
| PopQA | 0.01 / 0.15 / 0.56 | 0.50 / 0.93 / 1.70 |
| 2Wiki | 0.35 / 0.70 / 1.13 | 0.62 / 1.23 / 2.08 |
| MuSiQue | 0.15 / 0.53 / 1.05 | 0.77 / 1.35 / 2.45 |

- 答錯組的 **p10 低於答對組的 p50**(HotpotQA 0.056 vs 0.171)→ 錯答案低尾鑽進對答案核心區
- **AUROC 0.782 的正確解讀**:隨機抽一對(對, 錯),錯的 entropy 較高的機率 78% —— **群體可分性**,不是逐題判別器

---

## 深入(b):校準曲線 —— 四資料集全部嚴格單調

**entropy 五分位 → direct 路徑準確率(每資料集 1000 題)**

| Dataset | Q1(最有信心) | Q2 | Q3 | Q4 | Q5(最沒信心) |
|---|---:|---:|---:|---:|---:|
| PopQA | **81.5** | 43.0 | 27.5 | 26.0 | 11.0 |
| HotpotQA | **60.5** | 34.5 | 29.5 | 19.0 | 8.0 |
| 2Wiki | **30.5** | 25.5 | 14.5 | 14.5 | 5.0 |
| MuSiQue | **22.0** | 13.5 | 7.0 | 5.5 | 3.0 |

- 單調性成立 → **「答對是猜的」不成立**(猜對的話低 entropy 段不會系統性更準)
- 但看**絕對值**:多跳資料集「最有信心」的一段也只有 22~30% 對,離「可信」很遠;只有單跳 PopQA 到 81.5%

---

## 深入(c):當錯誤偵測器用,一半的錯逃得掉

規則:`entropy > τ` 就標記「可能答錯」(τ 取 both_correct 的 p90 / p75)

| Dataset | τ=p90(誤標對題 10%)抓到錯題 | τ=p75(誤標 25%)抓到錯題 |
|---|---:|---:|
| 2Wiki | 38.6% | 53.8% |
| HotpotQA | 44.7% | 61.5% |
| MuSiQue | 36.4% | 64.0% |
| PopQA | 49.0% | 71.6% |

> 只允許誤傷 10% 對題時,**只能抓到 36~49% 的錯題** ——
> 剩下一半以上錯答案的 entropy 跟對答案一樣低(= confidently-wrong 9.5~28.8% 的另一面)

**結論:entropy 適合做 accept/abstain 的風險排序,不適合逐題判對錯**

---

## 高信心答錯:規模與本質(診斷實驗)

各資料集取 entropy **最低** 100 題:錯誤 2Wiki 57 / MuSiQue 48 / PopQA 37 / HotpotQA 27,合計 **169/400**

**關鍵解剖:169 個錯誤全部是**

```text
mean_max_ent = 0   n_uniq = 1   sc_agree = 1.0
```

15 次取樣產生**一模一樣**的錯答案 —— 不是猶豫,是**穩定停在錯的推理路徑**

- zero-entropy 平台全貌:1538/4000 題 entropy=0,其中 **620 題(40.3%)答錯**
- 資料集級失效:2Wiki 上 AUROC(ent→錯)只有 **0.540(近隨機)**,rescue 率隨 entropy 遞減 —— parametric 錯答本質就是 confident-but-wrong

---

## 高信心答錯:三大失敗型態(案例)

**① 停在 bridge entity,沒走完最後一跳**(MuSiQue 乾淨案例;2Wiki 也常見)

| Case | Question | Hop 1:找 bridge entity | Hop 2:回答 final relation | 15/15 錯答 |
|---|---|---|---|---|
| UHF | Who founded the company that distributed *UHF*? | *UHF* → distributor = **Orion Pictures** | **Orion Pictures** → founder = **Mike Medavoy** | **Orion Pictures** |
| Fed Cup | The 1967 Federation Cup is an instance of this tournament which is named after what organization? | 1967 Federation Cup → instance of = **Fed Cup** | **Fed Cup** → named after = **International Tennis Federation** | **Fed Cup** |

> 註:這裡的 15-sample 診斷是 **no-RAG**,沒有真正 retrieved docs;
> 下方列的是 raw paragraph/supporting facts,用來看為什麼第一跳很 salient。
> - *UHF (film)* 文件給 **Orion Pictures**;*Mike Medavoy* 文件才給 **co-founder of Orion Pictures**。
> - *1967 Federation Cup* 文件給 **Fed Cup**;*Fed Cup* 文件才給 **International Tennis Federation**。

> 口徑:169 個 lowest-entropy errors 目前是案例解剖,未逐題標註三類比例;
> 可嚴謹宣稱「answer-path incompleteness 是主要族群」,但不要寫 type-1 精確佔比。
> 另一個 GPT-OSS subtype judge(n=998) 中,B2 鏈斷/中間答案類最大:443/998(44.4%;HI_wrong 中 60.4%)。

**② 答案型別錯**:*Are Kaufland and Otrag in the same country?* → gold **yes**、答 **Germany**

**③ 鄰近實體 lock-in**:*director of Daadagiri is from?* → gold **Pakistan**、答 **India**;Vernydub 出生地 gold **Zhytomyr**、答 **Zaporizhya**

> 共通點:第一跳/鄰居答案很 salient → entropy 極低。
> **失敗不是知識缺乏,是答案路徑不完整** → 支持 type-aware verifier + final-answer guard
> (另有 ④ alias 假錯僅 7/169、⑤ span 粒度不符 —— 屬 metric caveat)

---

## 完整案例:UHF 為什麼會停在 bridge entity

**Original question:** Who founded the company that distributed the film *UHF*?  
**Gold:** **Mike Medavoy**  
**診斷結果(no-RAG 15 samples):** `Orion Pictures` ×15,`mean_max_ent=0`,`n_uniq=1`,`sc_agree=1.0`

> 註:這個 15-sample 診斷本身**沒有檢索**;下表列的是若走 decomposition 時應命中的 raw support documents。

| Hop | 子問題 | 應得到 | 關鍵文件(raw support) | 文件線索 |
|---|---|---|---|---|
| 1 | `UHF >> distributed by` | **Orion Pictures** | idx 10: *UHF (film)* | 電影取得 Orion Pictures 支持/發行 |
| 2 | `Orion Pictures >> founded by` | **Mike Medavoy** | idx 6: *Mike Medavoy* | Mike Medavoy 是 Orion Pictures co-founder |

**為什麼 direct answer 會停在 Hop 1?**

- 原題表面詞集中在 `UHF` / `distributed company`,最容易被喚出的答案是公司名 **Orion Pictures**。
- 但題目真正問的是「那家公司由誰 founded」,需要把 **Orion Pictures** 再當 bridge query 走 Hop 2。
- entropy / SC 只看到答案分布很集中,看不到「這個答案其實只是中間節點」。
- 候選池也有 UHF 干擾文件(SICRAL 1B、Videodrome、WCCO-TV、CFVS-DT),所以 broad retrieval 可能 drift;正確鏈路是先定位 *UHF (film)*,再查 *Mike Medavoy*。

**對照 SEEK 拆解 trace:** `UHF >> distributed by → Orion Pictures`;`Orion Pictures >> founded by → Mike Medavoy`。

---

## 排除一個假設:是 popularity bias 嗎?→ 不是

PopQA × Wikipedia 月流量(n=1000):

- confidence 與實體熱門度**無關**:Spearman |ρ| < 0.05(n.s.),答錯子集內亦然
- zero-entropy 答錯組 vs 答對組的 popularity 分布無差異(Mann-Whitney p=0.58)
- entropy 訊號不因冷門實體崩壞:各 popularity 層 AUROC 穩定 **0.72~0.75**

唯一的偏斜在**語義層**:

| 實體熱門度 | 自信錯誤中 alias 假錯 | 真錯 |
|---|---:|---:|
| 熱門 | 28%(13/46) | 72% |
| 冷門 | **0%(0/50)** | **100%** |

> **結論:confident hallucination 是 deterministic-wrong-path 問題,不是 popularity bias**

---

## 低信心答對:entropy 高 ≠ 錯(p47 另一面)

最高 entropy 尾端 100 題/資料集,官方判對 **76/400(19%)**;36/76 的 sc_agree < 0.5

**三類型態:**
1. **正解在雜訊中險勝**:*Tim Hecker 最新專輯?* → Love Streams 只拿 4/15 票仍贏(其餘為幻覺專輯名)
2. **yes/no 題被中間證據污染**:samples 混入 UK / USA 等屬性,最終 boolean 仍勝出
3. **span 寫法搖擺**:New York vs New York City → F1 判對,entropy 高只因寫法不穩

**最大 caveat:76 題中只有 39 題 exact match,37 題僅靠 token-F1≥0.5**
(例:gold *323-272 BC* → 答 *5043 BC* 只共用 "BC";*Lena Olin* → *Stig Olin* 同姓不同人)

> MuSiQue 最嚴重:10 題「低信心答對」裡 9 題是 F1-only
> → 論文措辭須區分 **officially correct / semantically correct**

---

## 延伸:信心值能拿來「選路徑」嗎?→ 不能

在 direct_only / decomp_only 兩個分歧象限,「選 entropy 較低那條」:

| 規則 | 結果 |
|---|---|
| raw 比較 | 退化:direct entropy 尺度天生小 → 幾乎永遠選 direct(decomp_only 只命中 6~25%) |
| z-score 校正後 | 命中率 **45~69% ≈ 丟銅板** |
| 現行 verifier | decomp_only 78~96% / direct_only 7~68% → 優勢來自 **decomp prior + 規則**,不是信心訊號 |

<br>

> **信心值能告訴你「大概對不對」(accept/abstain),不能告訴你「哪條路徑對」**
> p46 oracle routing gap(+3.8~+6.1 EM)無法靠比較兩路 entropy 關閉

---

## 總結:對 SEEK 的意涵

1. **MAX_ENTROPY 是有用的全域訊號,但抓不到 deterministic 推理錯誤** —— entropy=0 平台仍有 40% 錯誤率,任何 τ 無效
2. **最大的高信心失敗不是知識缺乏,是答案路徑不完整**(bridge entity / 型別錯)→ type-aware verifier 與 final-answer guard 的設計動機
3. **低信心 ≠ 錯** —— 正解常在雜訊中勝出 → aggressive routing 浪費計算
4. **信心比較不能做 direct-vs-decompose routing** → verifier 的價值在規則,不在 entropy 比大小

<br>

**p47 建議直接放兩個數字:**

> 📈 **pooled AUROC 0.78** — entropy 確實與錯誤率相關(且五分位校準嚴格單調)
> ⚠️ **confidently-wrong 9.5~28.8%** — 低 entropy 不保證正確
