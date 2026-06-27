# 實驗:MAX_ENTROPY 與答對與否的關係

> 這是一份完整的實驗說明(目的、資料、方法、結果、檔案位置)。
> 結論摘要見 [README.md](README.md);數字表見 [RESULTS.md](RESULTS.md)。

---

## 1. 目的

驗證一個假設:**模型在生成答案時的「token 熵」高低,能不能預測這題會不會答對?**

具體的訊號是 **MAX_ENTROPY**(`mean_max_ent`)——每一題模型輸出序列中「最猶豫的那個 token」的熵。
若假設成立:

- 低 MAX_ENTROPY → 模型「有把握」→ 傾向答對
- 高 MAX_ENTROPY → 模型「在猶豫」→ 傾向答錯

這同時是 SEEK 拿熵當 gate / skip-decompose router 的理論依據:熵必須真的帶有
「模型知不知道自己在說什麼」的訊號,gate 才有意義。

---

## 2. 資料來源

### 2.1 用的檔案
`/home/S113065528/Research/main_methodology/results/{dataset}_1000_msample.jsonl`

四個資料集,每個 1000 題 dev(共 4000 題):
`2wiki_1000_msample.jsonl`、`hotpotqa_1000_msample.jsonl`、
`musique_1000_msample.jsonl`、`popqa_1000_msample.jsonl`

### 2.2 這些檔案怎麼產生的
由 [`main_methodology/scripts/probe_multisample.py`](../../main_methodology/scripts/probe_multisample.py)
產生(driver:`scripts/run_m2b_multisample.sh`)。設定:

| 項目 | 值 |
|------|------|
| Base model | Llama3-8B-Instruct |
| Adapter (LoRA) | `checkpoints/llama3-8b-decomposer-musique-2wiki-popqa`(3-source decomposer) |
| 檢索 | 每題 BM25 top-k=5 context(等同 always-skip / M1 single-step RAG 設定,**有檢索、非閉書**) |
| 取樣數 | n_sample = 15 |
| 取樣參數 | temperature = 0.7, top_p = 0.95, max_new_tokens = 48, do_sample |

### 2.3 每個欄位的定義
對每一題,先用 top-k context 組 prompt,取樣 15 條答案;對每條答案的每個非 EOS token
算 token 熵 `H_t = Σ_v −p(v)·log p(v)`(`torch.special.entr`,對 top-p 遮罩的 −inf 安全):

| 欄位 | 意義 |
|------|------|
| **`mean_max_ent`** | **本實驗用的 MAX_ENTROPY**:每條樣本取「max over token 的 H_t」,再對 15 條取平均 |
| `mean_mean_ent` | 每條取 mean-token 熵,再對 15 條平均 |
| `mean_min_lp` | 每條取 min token log-prob,再對 15 條平均 |
| `disc_entropy` | 15 條答案「字串分布」的 Shannon 熵(答案層級,不是 token 層級) |
| `sc_agree` | 多數決同意比例 = 最大票數 / 15 |
| `majority` | 正規化後票數最高的答案 |
| `gold` | 標準答案 |

---

## 3. 方法(本實驗自己做的分析)

程式:[`code/analyze_maxent_correctness.py`](code/analyze_maxent_correctness.py)
(對錯判定與 AUC 邏輯沿用 `experiment/self_certainty/scripts/exp4`、`exp5b`)。

1. **判定對錯**:取 `majority` 與 `gold`,用 normalized EM 或 token-F1 ≥ 0.5 判對(`check_correct`)。
   每題得到 0/1 標籤。
2. **配對訊號**:每題的 (MAX_ENTROPY, 對/錯)。
3. **分桶**:把 4000 題依 MAX_ENTROPY 由低到高排序,切成等數量桶(equal-frequency,
   主圖 10 桶、長條圖 8 桶),每桶算準確率,附 95% Wilson 信賴區間。
4. **統計量**:
   - Spearman(MAX_ENTROPY, error)——預期為正
   - point-biserial(MAX_ENTROPY, correct)——預期為負
   - AUROC(把「低熵」當成「預測答對」的分數,Mann–Whitney U)
5. **跑法**:`conda activate thesis && python code/analyze_maxent_correctness.py`

---

## 4. 結果

Pooled(n=4000):答對均熵 0.268 vs 答錯均熵 0.602;
Spearman(ent,error) = **+0.326**(p≈1e-99),AUROC = **0.686**。
分桶準確率由低熵 ~58% 單調降到高熵 ~13%。四個資料集方向一致(PopQA 最強,AUROC 0.741)。
完整數字見 [RESULTS.md](RESULTS.md)。

### 重要 caveat
這是**整體 / SC 答案層級**的正相關。在真 multi-hop(2Wiki / MuSiQue)上,
低熵尾端會有「**confident-but-wrong**」(自信卻答錯)——見
`main_methodology/results/wiki21M_router_round2.md` 的 router 負面發現。
若進 thesis 要補一句這個限制,以免與 router 結論衝突。

---

## 5. 檔案位置一覽

```
Research/experiment/maxent_correctness/
├── EXPERIMENT.md   ← 本文件(完整方法)
├── README.md       ← 結論 + 執行方式
├── RESULTS.md      ← 四資料集 + pooled 統計表
├── code/
│   └── analyze_maxent_correctness.py
└── figures/
    ├── fig1_main_maxent_vs_accuracy.png          主圖:準確率隨熵單調下降(折線+CI)
    ├── fig2_maxent_vs_error_positive.png         正相關視角:錯誤率隨熵上升
    ├── fig2b_bar_accuracy_down_error_up.png      長條圖:答對率↓ / 答錯率↑ 並排對比
    ├── fig3_distribution_correct_vs_incorrect.png 答對/答錯兩群的熵分布
    └── fig4_per_dataset_grid.png                 四資料集各自的準確率 vs 熵

來源資料(唯讀,非本資料夾):
  Research/main_methodology/results/{2wiki,hotpotqa,musique,popqa}_1000_msample.jsonl
產生來源資料的程式:
  Research/main_methodology/scripts/probe_multisample.py  (+ run_m2b_multisample.sh)
```
