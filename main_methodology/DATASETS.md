# SEEK 資料集與評估設定

本文件描述目前論文採用的資料設定。SEEK 的 decomposer 使用三個來源、共
12,000 筆 SFT 樣本訓練；完整系統則在四個 QA 資料集上評估。

## 1. 正式設定總覽

| 資料集 | 類型 | Decomposer 訓練 | 完整系統評估 |
|---|---|---:|---:|
| **MuSiQue** | 多跳 QA | 4,000 筆 gold decomposition | 1,000 題 |
| **2WikiMultiHopQA** | 多跳 QA | 4,000 筆由 `evidences` 轉換的 decomposition | 1,000 題 |
| **PopQA** | 單跳 QA | 4,000 筆單步 decomposition；排除評估題 | 1,000 題 |
| **HotpotQA** | 多跳 QA | 不參與訓練 | 1,000 題 |

因此，正式 decomposer 是：

```text
MuSiQue 4K + 2Wiki 4K + PopQA 4K = 12K SFT
```

訓練設定位於
`configs/llama3_8b_decomposer_lora_3ds.yaml`，其中 LLaMA-Factory 的資料集名稱為
`musique_decomp,twowiki_decomp,popqa_decomp`。HotpotQA 沒有加入 decomposer
訓練，可用來觀察模型對未見訓練來源的泛化。

## 2. 下載與註冊正式 12K 資料

正式版本已公開於
[`Chien1011/SEEK-decomposer-12k`](https://huggingface.co/datasets/Chien1011/SEEK-decomposer-12k)。
下載並驗證固定 revision：

```bash
SEEK_DATA_DIR=/tmp/SEEK-decomposer-12k
hf download Chien1011/SEEK-decomposer-12k \
  --repo-type dataset \
  --revision 3dcb5c057a82b6610b7b303855e273bb93f9bab7 \
  --local-dir "$SEEK_DATA_DIR"
(cd "$SEEK_DATA_DIR" && sha256sum -c SHA256SUMS)
```

複製到 LLaMA-Factory：

```bash
mkdir -p data/seek_decomposer_12k
cp -R "$SEEK_DATA_DIR/data/." data/seek_decomposer_12k/
```

將以下三筆合併至 LLaMA-Factory 的 `data/dataset_info.json`：

```json
{
  "musique_decomp": {
    "file_name": "seek_decomposer_12k/musique/train.json"
  },
  "twowiki_decomp": {
    "file_name": "seek_decomposer_12k/twowiki/train.json"
  },
  "popqa_decomp": {
    "file_name": "seek_decomposer_12k/popqa/train.json"
  }
}
```

## 3. 12K decomposer 資料如何建構

### MuSiQue（4K）

MuSiQue train split 有 19,938 題，dev split 有 2,417 題。訓練樣本直接採用
原始 `question_decomposition`，並依原始 hop 類別的自然比例抽取 4,000 題。
`#N` 佔位符會保留，用來表示子問題之間的依賴關係。

| hop 類別 | 筆數 | 比例 |
|---|---:|---:|
| 2hop | 2,884 | 72.1% |
| 3hop1 | 750 | 18.8% |
| 3hop2 | 130 | 3.2% |
| 4hop1 | 130 | 3.2% |
| 4hop2 | 26 | 0.7% |
| 4hop3 | 80 | 2.0% |

主要產物：

- `datasets/musique/gold/musique_gold_sft_4000.jsonl`：保留完整 gold 欄位的抽樣結果。
- `datasets/musique/gold/musique_decomp_lf.json`：轉成 LLaMA-Factory 可讀格式。

建構腳本為 `scripts/sample_gold_sft.py` 與 `scripts/build_lf_dataset.py`。

### 2WikiMultiHopQA（4K）

2Wiki train split 有 167,454 題，dev split 有 12,576 題。每題的
`evidences` 三元組 `(fact, relation, entity)` 會轉成一個原子步驟：

```text
fact >> relation
```

若目前步驟的 `fact` 等於前一步產生的 `entity`，會以 `#N` 表示依賴；其餘步驟
保留為獨立的 comparison hop。4,000 題依原始 `type` 比例抽樣：

| type | 筆數 | 比例 |
|---|---:|---:|
| compositional | 1,827 | 45.7% |
| comparison | 1,241 | 31.0% |
| bridge_comparison | 827 | 20.7% |
| inference | 105 | 2.6% |

主要產物為
`datasets/2wiki/gold/2wiki_gold_sft_4000.jsonl` 與
`datasets/2wiki/gold/2wiki_decomp_lf.json`；建構腳本為
`scripts/build_2wiki_gold.py` 與 `scripts/build_lf_dataset.py`。

### PopQA（4K）

PopQA 是單跳資料集。每個 `(subject, property, object)` 樣本轉成一個步驟：

```text
#1: subject >> property
```

這些樣本讓 decomposer 學會：當問題本身已經是原子問題時，輸出單一步驟即可，
不必強制過度分解。`scripts/build_popqa_sft.py` 會先排除正式評估使用的 1,000 個
question IDs，再以固定 seed 抽取 4,000 題，輸出
`datasets/popqa/gold/popqa_decomp_lf.json`。

### HotpotQA（僅評估）

HotpotQA train split 有 90,447 題，dev split 有 7,405 題。它沒有提供與 MuSiQue
相同形式、可直接作為訓練目標的 gold 子問題序列，因此不加入 12K SFT；正式
實驗只使用其準備後的 1,000 題評估集。

## 4. SFT 樣本格式

LLaMA-Factory 使用 Alpaca 格式，每個樣本包含 `instruction`、`input` 和 `output`：

```json
{
  "instruction": "Decompose the following multi-hop question ...",
  "input": "the original question",
  "output": "#1: first atomic question\n#2: question using #1"
}
```

decomposer 只預測有順序的子問題，不預測各步答案。子答案是在 SEEK 推論期間由
基座模型產生。

## 5. 四資料集評估 protocol

正式結果在 HotpotQA、2WikiMultiHopQA、MuSiQue 與 PopQA 各 1,000 題上計算，
共 4,000 題。檢索使用完整 Wikipedia `wiki` 索引（約 21M passages），四個資料集
統一使用 `top_k=8`；這與只在資料集提供的 distractor paragraphs 中選取證據是
不同 protocol，兩者結果不可混用。

Git 已包含這 4,000 題的 compact question/gold manifests，位置是
`datasets/evaluation/`；可用 `scripts/prepare_eval_subsets.py` 從官方來源重新建立。
含 paragraphs 的 prepared dev files 及完整上游資料集仍屬外部資產。

推論時，SEEK 先產生子問題及暫定答案，再從 next-token distribution 計算
**Tail Leakage**。當 Tail Leakage 達到閾值時才為該步檢索，最後合成分解路徑答案，
並在 direct 與 decomposition 路徑之間選擇輸出。

論文採用的 accuracy-first 操作點為：

```text
Tail Leakage tau = 0.002305
Macro EM / F1    = 41.88 / 48.90
```

可由 Git 內附的最小 trace bundle 離線重算；請參考 `results/README.md`。

## 5. 本機資料目錄

大型原始資料、轉換後 SFT 與含檢索段落的 dev files 不納入 Git。完整重跑時預期
使用下列目錄：

```text
datasets/
├── musique/{raw,gold,...}/
├── 2wiki/{raw,gold,...}/
├── popqa/{raw,gold,...}/
└── hotpotqa/{raw,...}/
```

相關腳本：

| 腳本 | 功能 |
|---|---|
| `scripts/prepare_musique.py` | 將 MuSiQue 準備成 SEEK 使用格式 |
| `scripts/prepare_2wiki.py` | 將 2Wiki 準備成 SEEK 使用格式 |
| `scripts/prep_popqa.py` | 建立 PopQA 1,000 題評估檔及檢索段落 |
| `scripts/sample_gold_sft.py` | 抽取 MuSiQue 4K gold decomposition |
| `scripts/build_2wiki_gold.py` | 由 2Wiki evidences 建立 4K decomposition |
| `scripts/build_popqa_sft.py` | 建立 PopQA 4K 單步 SFT 並排除評估題 |
| `scripts/build_lf_dataset.py` | 轉換為 LLaMA-Factory Alpaca 格式 |

早期的 GPT-OSS-20B 生成分解、count-match 與 EM 篩選檔案只保留作歷史分析，
不屬於目前 12K decomposer 的訓練資料。
