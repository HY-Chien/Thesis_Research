# SEEK — 復現指南(Reproduction Guide)

**SEEK**(Self-certainty Evaluated Evidence-gated Knowledge-retrieval):以 sub-question 為決策單位、
tail-leakage surprisal 為信心訊號的 adaptive RAG 框架。本目錄是碩士論文(`../Thesis/`)的完整實驗程式碼與結果。

- 正式結果的唯一權威彙整:**`RESULTS_MASTER.md`**(操作點、主表、所有 ablation 出處)
- 腳本索引與命名慣例:`scripts/README.md`;結果檔結構:`results/README.md`
- 資料集說明:`DATASETS.md`

---

## 0. 系統需求

| 項目 | 需求 |
|------|------|
| GPU | 推論/訓練 Llama-3-8B(bf16 + LoRA):單卡 ≥24GB VRAM(實驗使用 2×GPU 分 shard) |
| 環境 | conda env `thesis`(Python 3.11,**torch 2.10.0+cu128** — 勿裝預設 cu130) |
| 檢索 | Elasticsearch 7.10.2 + `wiki` 索引(psgs_w100,21M passages,對齊 DeepRAG) |
| 論文編譯 | TeX Live 2026(pdflatex + biber) |

## 1. 環境安裝

```bash
conda env create -f environment_thesis.yml   # 建出名為 thesis 的環境
conda activate thesis
# 或用 pip 對照:pip install -r requirements_thesis.txt

# 訓練 decomposer 需要 LLaMA-Factory(pin 至實驗使用的 commit):
git clone https://github.com/hiyouga/LLaMA-Factory ~/LLaMA-Factory
cd ~/LLaMA-Factory && git checkout 7e20db5735443861fad91766614b3415ff8966a4
pip install -e ~/LLaMA-Factory
```
註:`environment_thesis.yml` 不含 llamafactory(dev 版無法從 PyPI 安裝),依上式從 git 裝。

## 2. 版控之外、必須另外取得的資產

`.gitignore` 排除大檔(`*.jsonl` `*.log` `*.safetensors` 等),**git clone 後缺以下內容**。
打包給別人時請附上(或依「重建方式」自建):

| 資產 | 位置 | 大小 | 重建方式 |
|------|------|------|---------|
| LoRA decomposer(正典) | `checkpoints/llama3-8b-decomposer-musique-2wiki-popqa/` | ~百 MB | §4-b 重新訓練 |
| 原始資料集 + 12K SFT | `datasets/{musique,2wiki,hotpotqa,popqa}/` | 數 GB | 官方載點 + `scripts/prepare_*.py` |
| wiki-21M ES 索引 | 外部(`~/Adaptive-RAG/elasticsearch-7.10.2` 的 data dir) | ~數十 GB | Adaptive-RAG repo `retriever_server/build_index.py`(psgs_w100) |
| 基座模型 | HF `meta-llama/Meta-Llama-3-8B-Instruct` | 16GB | HF 下載(需授權) |
| 實驗 traces / logs | `results/traces/`(721MB)、`logs/`(254MB) | ~1GB | 唯一原始證據,無法重建,**務必隨包附上** |

```bash
# 打包範例(程式碼走 git,大檔走 tar):
tar czf seek_assets.tar.gz results/traces logs checkpoints datasets
```

## 3. 復現路徑 A:從既有 traces 重現分析與圖表(免 GPU,建議先走這條)

所有論文表格與分析都是離線讀 `results/traces/` 產生:

```bash
conda activate thesis
python scripts/analyze_tail_leakage_variants.py      # 例:訊號 battery(→ results/reports/)
python scripts/make_threshold_tradeoff_fig.py        # 例:論文 ch5 圖(→ ../Thesis/figures/)
python ../Thesis/make_ch3_signal_figs.py             # ch3 Fig 3.4–3.7(數字內建,無外部依賴)
```

各 analyze/replay 腳本對應的論文章節見 `scripts/README.md` 的實驗家族表;
注意:2026-07-31 整理後 traces 移到 `results/traces/{dataset}/`,舊腳本內寫死的平鋪路徑需對應調整。

## 4. 復現路徑 B:完整重跑 pipeline(需 GPU + ES)

**a. 資料準備(12K 三來源 SFT)**
```bash
python scripts/prepare_musique.py && python scripts/prepare_2wiki.py
python scripts/build_popqa_sft.py          # PopQA 已剔除 1,000 筆評測題
python scripts/sample_gold_sft.py          # 合成 12K(MuSiQue/2Wiki/PopQA 各 4K)
```

**b. 訓練 decomposer(LoRA on Llama-3-8B-Instruct)**
```bash
llamafactory-cli train configs/llama3_8b_decomposer_lora_3ds.yaml
# cross-backbone 消融:scripts/qwen25_decomposer_lora.yaml
```

**c. 啟動 Elasticsearch(heap 需 ≥8g,1g 會 429)**
```bash
ES_JAVA_OPTS="-Xms8g -Xmx8g" ~/Adaptive-RAG/elasticsearch-7.10.2/bin/elasticsearch -d
curl -s localhost:9200/_cat/indices | grep wiki   # 確認 wiki 索引存在
```

**d. 主 pipeline:兩階段(trace 生成 → 離線 headline 鏈)**

階段一:dual-path trace 生成(GPU;entropy gate 產生並儲存每步的 no-retrieval / retrieved 答案與訊號):
```bash
python scripts/run_seek_dual.py \
  --adapter checkpoints/llama3-8b-decomposer-musique-2wiki-popqa \
  --dev <dataset 的 1000 題評測 jsonl> \
  --mode C --corpus wiki --top-k 8 --title-mode pathaware \
  --signal max_ent --tau 0.3 --no-evidence --fallback auto
# 批次啟動器參考:scripts/run_wiki_deeprag_setting.sh(以 "ds:devfile" 形式傳參)
```

階段二:論文 headline 鏈(對儲存的 trace 做離線重播,對應 RESULTS_MASTER §4.15–4.19):
```bash
python scripts/rerun_prompt_fix.py   --ds <ds> ...    # prompt fix A:step 回答前置原題
python scripts/rerun_tlsum_fixA.py   --ds <ds> --out <jsonl>
#   ↳ tl_sum step gate,內建 TLSUM_TAU=0.002305(論文 accuracy-first 操作點)
python scripts/replay_verifier_tlsum_fixA.py ...      # all-tl_sum verifier
python scripts/rescue_value_answers.py ...            # value→entity synthesis guard(離線)
```
**注意 `--signal` 沒有 `tl_sum` 選項**——tl_sum 操作點一律由階段二的 replay 腳本套用;
各腳本的精確閾值與參數以 `RESULTS_MASTER.md` canonical 區塊為準。synthesis 一律 LoRA-off。

**e. 驗收**:對照 `RESULTS_MASTER.md` 主表(honest wiki-21M,n=1000×4,
macro EM/F1 = 41.62/48.66,vs DeepRAG 38.45/45.75 三勝一負)。

## 5. 論文編譯

```bash
cd ../Thesis
~/texlive/2026/bin/x86_64-linux/pdflatex -interaction=nonstopmode main.tex
~/texlive/2026/bin/x86_64-linux/biber main
~/texlive/2026/bin/x86_64-linux/pdflatex -interaction=nonstopmode main.tex   # ×2
# 差異標註版同法編 main_diff.tex
```

## 6. 已知的坑

- **ES heap**:預設 1g 會大量 429,務必 8g(2026-06-04 修正)。
- **synthesis 必須關 LoRA**:開著會長答案劣化(2Wiki 37.7→62.1 修正的由來)。
- **torch 版本**:必須 cu128;`pip install torch` 預設 cu130 與本機 driver 不合。
- **PopQA top_k**:2026-06-17 舊 canonical 用 top_k=3,現行 headline 鏈四資料集一律 top_k=8。
- **distractor pool 與 honest wiki-21M 是兩套 protocol**,數字不可混表(RESULTS_MASTER §2 vs §3)。
