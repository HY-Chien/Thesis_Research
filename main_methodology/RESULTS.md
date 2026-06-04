# SEER 實驗結果

> 更新日期:2026-05-21

---

## 1. SEER Decomposer — 第一版(MuSiQue gold,LoRA)

### 訓練設定
- 底模:Llama-3-8B-Instruct
- 資料:`musique_gold_sft_4000.jsonl`(4000 筆 MuSiQue gold,自然 hop 分佈)
- 方法:LoRA(rank 16,all-linear),lr 1e-4,cosine,3 epochs
- 輸出格式:純 `#N:` 文字,依賴關係用 `#N` 佔位符
- 訓練時間:13 分鐘(單張 L40S),train_loss 0.25 → 0.04
- Checkpoint:`checkpoints/llama3-8b-decomposer-musique/`

### 評估結果(MuSiQue dev,2417 題)

| 指標 | 數值 | 說明 |
|---|---|---|
| parse_ok | **100.0%** (2417/2417) | 輸出全為合法 `#N:` 格式 |
| deps_valid | **100.0%** (2416/2417) | 依賴邊全部指向更早步驟 |
| count_match | **82.2%** (1986/2417) | 生成步數 == gold 步數 |
| topo_match | **95.4%** (1895/1986) | 步數吻合時的拓樸吻合率 |

### 各 hop 細分

| hop | n | parse% | count% | topo% |
|---|---:|---:|---:|---:|
| 2hop | 1252 | 100% | 97.6% | 99.8% |
| 3hop1 | 568 | 100% | 71.8% | 96.1% |
| 3hop2 | 192 | 100% | 80.2% | 72.1% |
| 4hop1 | 246 | 100% | 46.3% | 98.2% |
| 4hop2 | 64 | 100% | 40.6% | 76.9% |
| 4hop3 | 95 | 100% | 65.3% | 64.5% |

### 解讀
- **格式完美**:100% 合法 `#N`、100% 依賴邊合法,輸出零後處理。
- **拓樸可信**:步數對的時候,拓樸 95.4% 都對。
- **明顯優於原始 OSS-20B**:count_match 82.2% vs OSS-20B 的 66.1%(同一 MuSiQue 上)。
- **弱點 = 深層題步數崩塌**:4hop1/4hop2 只有 40–46% 步數對,模型傾向合併步驟。
  原因:4hop 在自然分佈訓練資料只佔 ~6%,模型看得少。

### 後續可能改善方向
1. 4hop 類別過採樣(犧牲自然分佈)
2. 加大 epoch / LoRA rank
3. 維持現狀(82% count + 95% topo,SEER 第一版可能已足夠)

---

## 2. SEER Decomposer — 第二版(MuSiQue + 2Wiki 合併,LoRA)

### 訓練設定
- 同 v1,資料改為 `musique_decomp` + `twowiki_decomp` 合併 = **8000 筆**
- 訓練時間:26 分鐘(單張 L40S),train_loss 0.16
- Checkpoint:`checkpoints/llama3-8b-decomposer-musique-2wiki/`

### 評估結果

| dev 集 | parse_ok | deps_valid | count_match | topo_match |
|---|---|---|---|---|
| MuSiQue (2417) | 100.0% | 99.9% | **81.6%** | **94.3%** |
| 2Wiki (12576) | 100.0% | 100.0% | **98.7%** | **92.5%** |

MuSiQue 對比 v1(82.2% / 95.4%):合併 2Wiki 後幾乎無損(-0.6% / -1.1%,雜訊範圍內)。
2Wiki 表現極佳 —— count_match 98.7%,因 2Wiki 結構規律、evidences 模板化。

### MuSiQue dev 各 hop

| hop | n | count% | topo% |
|---|---:|---:|---:|
| 2hop | 1252 | 98.0% | 99.8% |
| 3hop1 | 568 | 71.3% | 93.3% |
| 3hop2 | 192 | 71.9% | 71.7% |
| 4hop1 | 246 | 48.0% | 97.5% |
| 4hop2 | 64 | 39.1% | 40.0% |
| 4hop3 | 95 | 63.2% | 56.7% |

### 2Wiki dev 各 type

| type | n | count% | topo% |
|---|---:|---:|---:|
| comparison | 3040 | 97.5% | 99.9% |
| compositional | 5236 | 99.9% | 95.1% |
| inference | 1549 | 99.9% | 92.0% |
| bridge_comparison | 2751 | 97.1% | 79.6% |

弱點:MuSiQue 深層題(4hop)步數仍崩塌;2Wiki bridge_comparison 的 4 步雙鏈拓樸偶爾連錯。

---

## 3. 對照基準:GPT-OSS-20B 原始分解(MuSiQue train 全量)

| 指標 | 數值 |
|---|---|
| parse_rate | 93.5% |
| count_match | 66.1%(of parsed) |
| step_answer EM | 65.1% |
| psi_acc | 78.4% |
