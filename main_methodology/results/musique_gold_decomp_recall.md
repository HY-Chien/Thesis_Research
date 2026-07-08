# MuSiQue Gold-Decomposition Retrieval Recall(p38 補做,老師回饋 #5)

Date: 2026-07-05(同日補:未微調 base 模型拆解 arm)
Scripts: `scripts/recall_gold_decomp_musique.py`、`scripts/decompose_base_musique.py`
Detail: `results/musique_gold_decomp_recall_detail.jsonl`(每題 per-k per-arm hit)、
`results/musique_1000_base_decomp.jsonl`(base 模型拆解原文)

## 設定

與 `answer_containing_recall_4ds_open_domain.md`(p38)同口徑:honest wiki-21M ES
index、path-aware retrieval(原問題 title-off、子問題 title-on)、
answer-containing hit@k = top-k 檢索段落含 gold answer 或 alias
(normalized substring)。N=1000(dev_first1000,**全部是 2-hop**,gold
decomposition 皆 2 步)。註:本 oracle 實驗只能做在 MuSiQue —— 四個資料集
中唯一附 gold sub-questions。MuSiQue dev 按題型排序(全 dev 2417 題 =
2-hop 1252 / 3-hop 760 / 4-hop 405),canonical protocol「dev 前 1000、
不 shuffle」(對齊 DeepRAG)機械性得到全 2-hop —— 非本實驗篩選,
全篇 MuSiQue 數字同此口徑;>2-hop 的覆蓋由主表 2Wiki 承擔
(bridge_comparison 4-hop,228/1000)。2-hop 單一 bridge 也讓下方條件化
分析的因果鏈最乾淨;hop 更多時 bridge 錯誤逐跳複利,seek–gold 差距預期
更大,但 +17pp headroom 僅主張於 2-hop。

六個 arm:

| arm | 子問題來源 | `#N` 佔位符代入 |
|---|---|---|
| `orig` | 無(原問題直接查) | — |
| `seek` | SEEK decomposer(canonical trace `musique_1000_verify_v2_pathaware.jsonl`) | **預測的**中間答案(推論時實際看到的) |
| `gold` | MuSiQue 官方 `question_decomposition` | **gold** 中間答案(oracle 拆解 + oracle 中間答案) |
| `gold_raw` | 同上 | 不代入(保留 `#1`,隔離「拆解文字本身」的貢獻) |
| `seek_goldres` | SEEK decomposer 的 `raw_sub_q` | **gold** 中間答案(隔離 SEEK 拆解文字品質) |
| `base` | **未微調 Llama-3-8B-Instruct**(同 base、同 prompt、無 LoRA;`decompose_base_musique.py`) | **gold** 中間答案(近似對位;解析失敗 153 題退回原問題) |
| `base_raw` | 同上 | 不代入 |
| `orig+seek` / `orig+gold` | 上述聯集 | |

子問題 arm = 每個子問題各取 top-k 後聯集(pipeline 實際看到的量)。

## 主表(hit@k %)

| k | orig | seek | **gold** | gold_raw | seek_goldres | base | base_raw | orig+seek | orig+gold |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8.5 | 22.0 | **30.8** | 8.7 | 33.9 | 41.2† | 10.6 | 24.5 | 32.6 |
| 3 | 14.1 | 34.7 | **49.0** | 16.0 | 50.0 | 49.3† | 17.6 | 38.3 | 50.8 |
| 5 | 19.0 | 38.8 | **55.8** | 19.8 | 54.1 | 52.9† | 20.6 | 43.9 | 57.8 |
| 8 | 23.6 | 42.7 | **60.8** | 23.8 | 60.1 | 56.3† | 24.5 | 47.6 | 62.5 |
| 10 | 25.6 | 46.0 | **65.8** | 25.5 | 64.9 | 58.1† | 25.9 | 50.9 | 67.6 |

† **base arm 有檢索預算 confound**:union 口徑下每個子問題各取 top-k,base 平均拆
2.98 步(SEEK 2.03、gold 2.00)→ base 看到約 1.5 倍的段落。k=1 時 base 41.2 >
gold 30.8 純粹是這個效應(3 個查詢 vs 2 個)。同預算下 base 的拆解文字並不比較好
(k 越大、預算差被稀釋後,base 一路低於 gold 與 seek_goldres)。

(答案比對含/不含 title 兩種口徑差 <0.3pp,結論不變;與 6/24 報告的
orig 16.9 / seek 37.4 有 ~1–2pp 實作差 —— 當時腳本未留存,本表六個 arm
出自同一份實作、內部一致,比較以本表為準。)

## 支撐統計:差距來自中間答案,不是拆解文字

- **`gold_raw` ≈ `orig`(19.8 vs 19.0 @5)**:不把 `#1` 換成實體,gold 拆解的
  檢索力和原問題一樣差 —— 2-hop 最終答案幾乎只能靠 step-2 查詢
  (`<bridge entity> >> relation`)撈到,而 step-2 查詢的品質取決於代入的
  bridge entity。**拆解的檢索增益本質上是「拆解+逐步作答」的鏈,不是拆解
  文字本身。**
- **SEEK step-1 中間答案 EM vs gold = 41.9%**(997 題可比;2026-07-06 以
  SQuAD normalization 全 1000 題重算 = 42.4%,分桶 424/576,差異僅
  normalization 細節)。
- **條件化 hit@5(2026-07-06 修正:gold 改用同子集算,原版拿子集 seek 比
  全集 gold 有選擇偏誤)**:

  | SEEK step-1 中間答案 | n | seek | gold(同子集) | seek_goldres | orig |
  |---|---:|---:|---:|---:|---:|
  | 答對 | 424 | 56.4 | 58.5 | 57.3 | 20.0 |
  | 答錯 | 576 | 25.9 | 53.8 | 51.7 | 18.2 |

  同子集下排序處處正常(gold ≥ seek_goldres ≥ seek,沒有 seek > gold 的
  假象)。**關鍵讀法**:答錯子集上 gold 幾乎不掉(53.8 vs 全集 55.8)——
  題目本身沒有變難、gold 證明答案檢索得到——但 seek 崩到 25.9。
  → **seek 與 gold 的 17pp 差距幾乎完全由中間答案錯誤解釋**;SEEK 的拆解
  文字(KG-triple 風格)與 gold 同構,問題出在 bridge entity 答錯後
  step-2 查詢跟著歪(retrieval drift)。

## 未微調 base 模型拆解的行為(「為什麼要 train decomposer」)

同 base(Llama-3-8B-Instruct)、同 DECOMP_INSTRUCTION prompt、greedy,只差沒掛 LoRA:

| | SEEK decomposer(LoRA) | 未微調 base |
|---|---|---|
| 步數分布(全 2-hop) | **963/1000 拆 2 步** {1:3, 2:963, 3:33, 4:1} | **只有 149/1000 拆 2 步** {0:153, 2:149, 3:414, 4:185, 5:66, 6:23, 7:7, 8–11:3} |
| 格式解析失敗 | ~0 | **153/1000(15.3%)**,推論時整條 decompose path 直接失效 |
| 風格 | KG-triple 短查詢(`X >> relation`),BM25-friendly | 長自然語言句,常過度拆解、偶爾自帶(可能錯的)parametric 實體 |

**訓練買到的不是「查詢文字更會檢索」——是結構可靠性**:oracle 中間答案下
base 的文字檢索力其實不差(52.9 @5,還吃了 1.5 倍預算紅利),但實際推論時
(1) 15.3% 題目直接沒有可用拆解;(2) 2-hop 拆成 3–11 步 → 要答對的中間答案
變多、drift 逐跳複利(SEEK 2 步時中間答案 EM 已只有 41.9%,鏈更長只會更糟);
(3) 步數多 = 檢索/生成次數多,推論成本直接上升。

## 結論(給簡報/論文)

1. **p38 的主張成立且被 oracle 強化**:拆解檢索 38.8% >> 原問題 19.0%(@5);
   若拆解鏈完美(gold),可到 **55.8%** —— decomposition-as-retrieval 的
   headroom 還有 +17pp。
2. **decomposer 文字品質不是瓶頸**(gold_raw ≈ orig;seek_goldres 54.1 ≈ gold
   55.8),**瓶頸是中間答案正確率(41.9%)與其引發的 step-2 query drift**。這與
   `decomp_failure_taxonomy.md`(retrieval_gap_both 占 68%)互相印證,也回應
   老師 #4「用 MuSiQue 訓練完會怎麼拆/遇到什麼問題」:拆得像 gold,
   但答錯 bridge 之後整條鏈歪掉。
3. **為什麼要 train decomposer**:未微調 base 過度拆解(85% 步數錯)+ 15.3%
   格式失敗;LoRA 訓練買到的是步數/格式的結構可靠性與短查詢風格,
   不是檢索魔法。完整階梯(@5,oracle 中間答案):gold 55.8 ≥ seek_goldres
   54.1 > base 52.9(且 base 多吃 1.5 倍預算);實際可部署(預測中間答案):
   seek 38.8;不拆:19.0。
