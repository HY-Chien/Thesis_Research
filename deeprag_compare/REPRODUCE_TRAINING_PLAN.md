# DeepRAG 訓練復現 Plan（在 A100 機器上跑，量測真實訓練 wall-clock）

> 目的：把 §4.8 的 DeepRAG 訓練成本從「估算」升級為「同硬體實測」。
> Deliverable：**在同一組 A100 上** 得到
> `SEEK decomposer = X 分` vs `DeepRAG 完整 pipeline（建資料 + SFT + DPO）= Y 小時`。
> 寫於 2026-07-01。本檔只是計畫，不需 GPU；真正執行在 A100 機器。

---

## 0. 先決定：忠實復現 or 近似復現（取決於卡數）

| 硬體 | 72B follow-up teacher | 是否忠實 | 論文如何交代 |
|---|---|---|---|
| **4×A100-80GB** | Qwen2.5-72B **bf16**（TP=4，或 TP=2 + 8B 另兩張） | ✅ 忠實 | 直接報實測 |
| **2×A100-80GB** | Qwen2.5-72B **int4（AWQ/GPTQ）**（~40GB，1 張）+ Llama-3-8B（另 1 張） | ⚠️ 近似 | 明註「因硬體限制，follow-up teacher 採 int4 量化；主策略模型與訓練皆為原精度」 |

> 兩種都能拿到有效的訓練 wall-clock。int4 只影響 follow-up 子問題的生成品質（teacher 角色），
> 不影響「訓練要花多久」這個我們真正要量的數字。A100 無 FP8 硬體，量化只能走 int4/int8。

---

## 1. 復現目標鎖定

- **復現對象 = 釋出模型 `xinyan233333/DeepRAG-8b`，其 base 為 Llama-3-8B-Instruct**（與 SEEK decomposer 同底）。
- 因此**全程主模型統一用 Llama-3-8B-Instruct**（見 §4 要修的 repo 不一致處）。
- repo：`~/DeepRAG`（訓練/資料建構腳本都在這）。訓練用 `~/LLaMA-Factory`。

---

## 2. 前置需求（在 A100 機器上先確認）

```bash
# 2.1 模型
#   - Llama-3-8B-Instruct：本機已有一份於
#       /home/S113065528/Adaptive-RAG/hf_cache/models--meta-llama--Meta-Llama-3-8B-Instruct
#     （A100 機器上請確認路徑，或重新下載）
#   - Qwen2.5-72B-Instruct：需下載（bf16 ~145GB；2 張卡走 int4 版，如 Qwen2.5-72B-Instruct-AWQ ~40GB）
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct
huggingface-cli download Qwen/Qwen2.5-72B-Instruct        # 4 張卡（bf16）
# huggingface-cli download Qwen/Qwen2.5-72B-Instruct-AWQ   # 2 張卡（int4）

# 2.2 磁碟：72B bf16 需 ~150GB 空間；建構資料輸出 construct/ 另留 ~數 GB
df -h .

# 2.3 依賴（DeepRAG repo）
bash ~/DeepRAG/install.sh            # spacy / elasticsearch==7.17.9 / beir 等
pip install vllm                     # 服務用；版本需與 CUDA/torch 相容

# 2.4 Elasticsearch + retriever（tree search 每步要檢索）
#   DeepRAG 用 ES 7.x；index 用其 retriever_tools 下腳本建（wiki dump）。
#   若沿用 Adaptive-RAG 既有 ES/index，需確認 schema 與 DeepRAG retriever_bm25 相容。
```

---

## 3. 硬體 profile 與服務啟動（**已修埠號衝突**）

> 原 repo bug：`launch/run.sh` 把 8B 開在 8000-8007，`launch/run-72b.sh` 把 72B 也開在 8000 → 撞。
> 下面重新分配埠號，且 data-construct 的兩個 URL 要對應改。

### Profile A — 4×A100-80GB（忠實 bf16）

```bash
# GPU 0,1 → 72B follow-up teacher（TP=2, port 8000）
CUDA_VISIBLE_DEVICES=0,1 nohup python3 -m vllm.entrypoints.openai.api_server \
  --model $HF/Qwen2.5-72B-Instruct --tensor-parallel-size 2 --port 8000 \
  > deploy_qwen72b_8000.log 2>&1 &

# GPU 2,3 → Llama-3-8B 主策略模型（各 1 張，port 8001 / 8002）
CUDA_VISIBLE_DEVICES=2 nohup python3 -m vllm.entrypoints.openai.api_server \
  --model $HF/Meta-Llama-3-8B-Instruct --tensor-parallel-size 1 --port 8001 \
  > deploy_llama_8001.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python3 -m vllm.entrypoints.openai.api_server \
  --model $HF/Meta-Llama-3-8B-Instruct --tensor-parallel-size 1 --port 8002 \
  > deploy_llama_8002.log 2>&1 &
```

### Profile B — 2×A100-80GB（近似 int4）

```bash
# GPU 0 → 72B int4 follow-up teacher（port 8000）
CUDA_VISIBLE_DEVICES=0 nohup python3 -m vllm.entrypoints.openai.api_server \
  --model $HF/Qwen2.5-72B-Instruct-AWQ --quantization awq --tensor-parallel-size 1 --port 8000 \
  > deploy_qwen72b_awq_8000.log 2>&1 &

# GPU 1 → Llama-3-8B 主策略模型（port 8001）
CUDA_VISIBLE_DEVICES=1 nohup python3 -m vllm.entrypoints.openai.api_server \
  --model $HF/Meta-Llama-3-8B-Instruct --tensor-parallel-size 1 --port 8001 \
  > deploy_llama_8001.log 2>&1 &
```

> 起完用 `curl http://0.0.0.0:8000/v1/models` 逐一確認上線再往下。

---

## 4. 要修的 repo 不一致（改成 Llama-3-8B 忠實復現）

DeepRAG repo 混用 Qwen2.5-7B / Llama-3-8B。復現釋出的 Llama-3-8B 版需統一：

1. **`scripts/data-construct/stage1.sh`**
   - `REMOTE_URL`（主策略/小模型，生中間答案）→ 指向 §3 的 8B 埠（8001[,8002]）。
   - `FOLLOW_UP_REMOTE_URL`（follow-up）→ 指向 72B 埠（8000）。
   - `--model_name_or_path` 由 `Qwen2.5-7B-Instruct` → `Meta-Llama-3-8B-Instruct`（僅影響 tokenizer/template）。
   - `--num_processes` 依 A100 併發調（先設 16 試跑，穩了再拉高）。
2. **`scripts/data-construct/stage2.sh`** 同步把 URL / `--model_name_or_path` 改成 8B。
3. **`scripts/data-construct/filter-stage1/tokenize_sft.py`、`filter-stage2/tokenize_dpo.py`**
   - `AutoTokenizer.from_pretrained("hf_models/Qwen2.5-7B-Instruct")` → Llama-3-8B。
4. **`scripts/training/stage1.sh`** `--model_name_or_path hf_models/Qwen2.5-7B-Instruct` → Llama-3-8B；
   `--template qwen25` → `llama3`。
5. **`scripts/training/stage2.sh`** `--model_name_or_path` = stage1 產出的 SFT checkpoint（本就 `--template llama3`，OK）。
6. **`filter-stage1/reformat.py`** 內有寫死的絕對路徑 instruction.txt → 改成 repo 內 `ICL-prompt/instruction.txt`。

---

## 5. 執行步驟（**每段都計時**）

> 統一用 `/usr/bin/time -v ... 2>> timing_deeprag.log` 或前後 `date +%s` 記秒數。
> 建一個 `timing_deeprag.log` 累積所有階段耗時。

```bash
cd ~/DeepRAG
LOG=~/DeepRAG/timing_deeprag.log ; : > $LOG
ts(){ echo "[$1] $(date -Is)" | tee -a $LOG; }

# --- Stage I：SFT 資料建構（tree search）→ 最長的一段 ---
ts "S1_construct_start"
bash scripts/data-construct/stage1.sh                 # 輸出 construct/sft/*
ts "S1_construct_end"
python scripts/data-construct/filter-stage1/extract.py      # 篩正確 response
python scripts/data-construct/filter-stage1/tokenize_sft.py # → construct/sft/tokenized_dataset
ts "S1_filter_end"

# --- Stage I：SFT 訓練（全參數 8B）---
ts "S1_sft_train_start"
bash scripts/training/stage1.sh                       # → checkpoint/.../sft/hotpot-wikihop
ts "S1_sft_train_end"

# --- Stage II：DPO 資料建構（用 SFT 後模型再跑 inference）---
#   先把 launch 的 8B 服務換成 SFT checkpoint（重起 vLLM 指向新權重），再：
ts "S2_construct_start"
bash scripts/data-construct/stage2.sh                 # 輸出 construct/dpo/*
python scripts/data-construct/filter-stage2/make_pair.py
python scripts/data-construct/filter-stage2/tokenize_dpo.py # → construct/dpo/tokenized_dataset
ts "S2_construct_end"

# --- Stage II：DPO 訓練（全參數 8B，8 epochs）---
ts "S2_dpo_train_start"
bash scripts/training/stage2.sh                       # → checkpoint/.../dpo/hotpot-wikihop
ts "S2_dpo_train_end"
```

---

## 6. 公平對照：在**同一台 A100** 重測 SEEK decomposer

> §4.8 現有的 37.9 分是 L40S 量的；要跟 A100 上的 DeepRAG 比，SEEK 也得在 A100 重跑一次。

```bash
cd ~/LLaMA-Factory && conda activate thesis
# canonical = 12K 三源 LoRA。用既有 config，把 dataset 指到三源、確認 model 路徑對 A100：
#   config: Research/main_methodology/configs/llama3_8b_decomposer_lora.yaml
#   dataset 改為: musique_decomp,twowiki_decomp,popqa_decomp   （= 12K）
#   model_name_or_path 改為 A100 上 Llama-3-8B-Instruct 路徑
/usr/bin/time -v llamafactory-cli train \
  Research/main_methodology/configs/llama3_8b_decomposer_lora.yaml 2>> ~/seek_decomposer_timing.log
# 對照參數（已知）：LoRA r16/α32, 3 epochs, 2250 steps, per_device_bs4 × ga4, lr1e-4, template llama3
```

---

## 7. 最後要填的對照表（實測後回填 §4.8）

| 階段 | SEEK | DeepRAG |
|---|---|---|
| 資料建構（tree search / OSS teacher） | 一次性 teacher 生成（可另計） | **Stage I + II construct 實測** |
| 梯度訓練 | LoRA SFT 3ep（**A100 實測 __ 分**） | 全參數 SFT 1ep + DPO 8ep（**實測 __ 小時**） |
| 服務叢集需求 | 單卡 | 72B + 8B 併存 |
| 總離線 wall-clock | __ | __ |

---

## 8. 風險 / 注意

- **最大不確定 = Stage I tree-search 時間**：取決於 rollout 深度、`num_processes`、ES 檢索延遲。先用 **小批（如 200 題）試跑估單題成本**，再外推全量，避免盲跑一整天。
- **ES/index**：DeepRAG 的 retriever 需對應 wiki index；若沿用 Adaptive-RAG 的 ES 要先驗證欄位相容，否則 tree search 每步檢索會壞。
- **SFT checkpoint 換服務**：Stage II 建資料前，記得把 vLLM 的 8B 服務重起指向 SFT 後權重（否則等於用未訓模型建 DPO pair）。
- **int4 profile 的誠實標註**：若走 Profile B，論文/答辯務必寫明「follow-up teacher 為 int4 量化」，其餘（主模型、訓練精度）維持原樣。
- **先試跑再全量**：建議先跑「200 題 construct → 小 SFT」全鏈路 smoke test，確認 pipeline 通、計時 hook 正確，再跑全量。

---

## 9. 若中途放棄也不虧

即使只完成到 **Stage I construct 的小批試跑 + 兩階段訓練的實測時間**，就已經能把 §4.8 的
「數量級估算」替換成「部分實測 + 明確外推」，比目前更硬。完整跑完則可直接寫成
「同硬體實測對照」，答辯無懈可擊。
