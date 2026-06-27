# SEEK 資料集說明

> 更新日期:2026-05-21
> 專案:SEEK(Sub-question Entropy-driven Efficient Retrieval)
> 用途:SEEK 第一階段「Data Construction」—— 建構 **SFT decomposer**(把複雜多跳問題拆成原子子問題)的訓練資料。

---

## 1. 三個資料集與其角色

SEEK 的 decomposer 要學會把多跳問題拆成帶依賴關係的子問題序列。我們用三個標準多跳 QA 資料集當訓練/評測來源:

| 資料集 | train | dev | 角色 |
|---|---:|---:|---|
| **MuSiQue** | 19,938 | 2,417 | 主力訓練。原生 `question_decomposition` gold,已抽 4000 筆 SFT |
| **2WikiMultiHopQA** | 167,454 | 12,576 | 訓練。`evidences` 三元組轉 gold 分解,已抽 4000 筆 SFT |
| **HotpotQA** | 90,447 | 7,405 | **僅下游評估**。無任何 gold 分解,不拿來訓練 decomposer |

**目前進度:** MuSiQue + 2Wiki 各 4000 筆 gold SFT 已建好。HotpotQA 經評估無 gold 分解訊號,定位為 SEEK 全流程的 out-of-domain 評估集,不參與 decomposer 訓練。

---

## 2. 為什麼需要分解資料

SEEK 的推論流程(見 proposal p.10):

```
複雜多跳問題 → a.Decompose(SFT decomposer)→ 子問題序列
            → b.Probe Entropy → c.Gate(低熵走參數記憶 / 高熵叫檢索)
```

第一步 Decompose 需要一個輕量 SFT decomposer。本資料夾的所有資料就是為了訓練/評估這個 decomposer。

---

## 3. GPT-OSS-20B 分解能力評測

我們用本地的 **GPT-OSS-20B**(mxfp4,~13GB,單張 L40S,TP=1)對 MuSiQue 全量 train(19,938 筆)做分解,結果:

| 指標 | 數值 | 說明 |
|---|---|---|
| parse_rate | **93.5%** (18,645/19,938) | 輸出能解析成合法 JSON 分解 |
| count_match | **66.1%** (12,318/18,645) | 生成步數 == gold 步數(佔已解析) |
| step_answer EM | 65.1% | 各步答案對 gold(僅步數吻合列) |
| psi_acc | 78.4% | 各步支持段落索引對 gold |
| final_answer EM / F1 | 53.2% / 65.5% | 最終答案對 gold —— **不採用** |

**關鍵判斷:** 「20B 分解能力準不準」要看 **parse_rate + count_match + 拓樸合法性**,不該看 final-answer EM。EM 會隨「回答能力」浮動,而回答能力是模型相依的,跟「分解能力」是兩回事 —— 資料建構階段只在意分解對不對。

> 早先看到的「59% 準確率」是誤導:raw train 按 hop 排序,前 2048 筆全是 2hop,那個數字只反映 2hop 子集。

---

## 4. 兩條建構路線(OSS-20B vs GOLD)

針對 MuSiQue,我們做了兩個 4000 筆 SFT 候選集:

### 路線 A:OSS-20B 分解 + 步數吻合篩選 〔已棄用〕
- 全量分解 → `filter_count_match.py`(parse OK + depends_on 邊合法 + 步數吻合)→ 12,313 筆存活(61.8%)
- `sample_sft.py` 按自然分佈抽 4000 筆
- **問題:** 12.9% 的依賴拓樸跟 gold 不一致;4hop 類別品質差(4hop3 拓樸只 15% 對)

### 路線 B:MuSiQue GOLD 分解 〔採用 ✅〕
- 直接從 raw train 按自然分佈抽 4000 筆,gold 分解依定義 100% 正確
- 把 gold `question_decomposition` 轉成 decomposer 訓練 schema
- **決定:先用這份 gold 4000 筆訓練,驗證 MuSiQue 單資料集能否拉升 performance**

---

## 5. 檔案清單(`datasets/musique/`)

### ✅ 現役檔案

| 檔案 | 筆數 | 說明 |
|---|---:|---|
| `gold/musique_gold_sft_4000.jsonl` | 4,000 | **訓練用 SFT 集**。gold 分解,自然 hop 分佈 |
| `raw/train.jsonl` | 19,938 | MuSiQue 原始 train |
| `raw/dev.jsonl` | 2,417 | MuSiQue 原始 dev(評測用) |
| `oss-20b/musique_full_decomposition.jsonl` | 19,938 | 20B 全量分解輸出(評測 20B 能力用) |

### 參考 / 中間產物

| 檔案 | 筆數 | 說明 |
|---|---:|---|
| `oss-20b/musique_full_decomposition_count_matched.jsonl` | 12,313 | 步數吻合篩選後存活 |
| `oss-20b/musique_sft_4000.jsonl` | 4,000 | 路線 A 候選集(已被 gold 取代) |
| `sft/train.jsonl` / `sft/dev.jsonl` | 19,938 / 2,417 | 早期簡易 SFT 轉檔(`input`/`output`,`#N` 格式) |

### 過時 / 可清理

| 檔案 | 筆數 | 說明 |
|---|---:|---|
| `oss-120b/selected_2048_with_oss120b_decomposition.jsonl` | 2,048 | 舊的「只分解、無子答案」run(本機其實無 120B) |
| `oss-20b/musique_train_2048*.jsonl` | 2048/1162/878 | 2048 筆種子批 + EM 篩選舊版(EM 已棄用) |
| `oss-20b/musique_train_100.jsonl`、`*smoketest*` | 100 / 10 / 0 | 煙霧測試 |

---

## 6. GOLD SFT 集規格(`musique_gold_sft_4000.jsonl`)

每列保留 raw 欄位,並新增 `gold_decomposition`(decomposer 訓練目標 schema):

```json
{
  "step_id": 1,
  "sub_question": "Hello Love >> performer",   // 保留 MuSiQue #N 佔位符
  "depends_on": [],                            // 從 #N 還原的依賴邊
  "answer": "Hank Snow",
  "paragraph_support_idx": 2
}
```

**hop 類別組成(完全貼合 raw train 自然分佈):**

| hop | 筆數 | 佔比 |
|---|---:|---:|
| 2hop | 2,884 | 72.1% |
| 3hop1 | 750 | 18.8% |
| 3hop2 | 130 | 3.2% |
| 4hop1 | 130 | 3.2% |
| 4hop2 | 26 | 0.7% |
| 4hop3 | 80 | 2.0% |

---

## 7. 2WikiMultiHopQA gold 分解(`datasets/2wiki/gold/`)

`evidences` 三元組 `(fact, relation, entity)` 直接轉 decomposer schema:
- `sub_question` = `"<fact> >> <relation>"`(與 MuSiQue gold 同風格)
- 依賴偵測(type-independent):某步 `fact` 等於前面某步 `entity` → chain hop,subject 改寫成 `#N`
- `answer` = entity;`paragraph_support_idx` = None(2Wiki 無此標註)

167,454 筆全部可用(evidences 覆蓋率 100%),34.8% 步驟為 chain hop。
4000 筆 SFT 依 `type` 自然分佈抽樣:

| type | 筆數 | 佔比 |
|---|---:|---:|
| compositional | 1,827 | 45.7% |
| comparison | 1,241 | 31.0% |
| bridge_comparison | 827 | 20.7% |
| inference | 105 | 2.6% |

檔案:`2wiki_gold_sft_4000.jsonl`、`2wiki_decomp_lf.json`(LLaMA-Factory,已註冊為 `twowiki_decomp`)。

---

## 8. 下一步

1. 訓練 decomposer:可選 MuSiQue 單獨、或 MuSiQue + 2Wiki 合併(共 8000 筆)
2. 評估 decomposer + 接 SEEK 推論流程(Probe Entropy + Gate)
3. HotpotQA 留作下游 QA 評估

---

## 相關腳本(`scripts/`)

| 腳本 | 功能 |
|---|---|
| `run_decomposition.py` | 用 vLLM 跑 GPT-OSS-20B 分解 |
| `eval_decomposition.py` | 評測分解品質(parse / count / EM / psi) |
| `filter_count_match.py` | 步數吻合篩選(不含 EM) |
| `sample_sft.py` | 從 OSS-20B 篩選結果抽自然分佈樣本(路線 A) |
| `sample_gold_sft.py` | 從 raw 抽 gold 自然分佈樣本(路線 B,現役) |
| `filter_decomposition.py` | EM 篩選(舊版,已棄用) |
