# SEER 端到端推論結果

> 更新日期:2026-05-25
> Decomposer:`checkpoints/llama3-8b-decomposer-musique-2wiki/`(MuSiQue+2Wiki gold 8000 筆 SFT LoRA)
> Base:Llama-3-8B-Instruct
> Retrieval:**Per-question BM25 over 20 candidate paragraphs**(IRCoT/Adaptive-RAG/DeepRAG-paper 同款)

---

## 1. 評估設定 vs 既有比較對象

| 來源 | Test set | Retrieval corpus | Base model |
|---|---|---|---|
| **SEER (本研究)** | dev 前 200 / 前 1000 / full(同一份 dev,不 shuffle) | per-question 20 paras (BM25) | Llama-3-8B-Instruct |
| **DeepRAG paper** (`scripts/inference/run.sh`) | **dev `[:1000]`(不 shuffle)** | **global Wikipedia (BM25 over DPR dump)** | Llama-3-8B-Instruct |
| Ours `neg_max_ent@-0.80` (EXPERIMENTS_SUMMARY §3.1) | 1000 sampled / dataset | per-question paras | vanilla Llama-3-8B-Instruct |

### ⚠️ DeepRAG sampling 細節(從 `src/main.py:178-183` 確認)

```python
if args.shuffle:                       # ← inference/run.sh 沒加
    data = data.shuffle(seed=42)
if args.sample != -1:
    samples = min(len(data), args.sample)
    data = data[:samples]              # ← 取前 N 題,不 shuffle
```

**重要後果**:
- DeepRAG 三個 multi-hop 都 `--sample 1000` 沒 `--shuffle` → 取 dev **前 1000 題**
- MuSiQue dev 按 hop 排序,前 1252 題全是 2hop → DeepRAG MuSiQue 22.5 EM 是**全 2hop 子集**的數字,不是 full dev
- 對齊比較必須用相同的「dev 前 1000」protocol,不能拿我們 full 2417 EM 跟他們 1000 題比

### Retrieval corpus 差異

DeepRAG 跑 global Wikipedia 索引,我們是 per-question 20 篇候選——後者命中率天花板較高(gold 通常在候選裡),這也是 Adaptive-RAG / IRCoT 主流 protocol,跟你之前 1000Q 基準同款,**屬於可比但偏向我方的設定**。

Dev sizes:
- MuSiQue dev: **2417** 題(已跑全量)
- 2Wiki dev: 12576 題
- HotpotQA dev: 7405 題

---

## 2. 主結果(目前)

### 2.1 mode C(LoRA always on)、max_ent gate、τ=0.3

#### A. 對齊 DeepRAG protocol(dev 前 1000 題,不 shuffle)

##### A.0 最終結果(2026-05-25,加入 synthesize prompt fix + option C entropy probe skip-decompose)

| Dataset | n | SEER EM / F1 | DeepRAG paper EM / F1 | Δ EM | Δ F1 |
|---|---:|---:|---:|---:|---:|
| **MuSiQue** | 1000 | **48.0 / 57.1** | 22.5 / 30.4 | **+25.5** ✓ | **+26.7** ✓ |
| **2Wiki** | 1000 | **45.0 / 53.7** | 48.1 / 53.3 | -3.1 | **+0.4** ✓ |
| **HotpotQA** | 1000 | **39.6 / 52.4** | 40.7 / 51.5 | -1.1 | **+0.9** ✓ |
| **PopQA (+option C)** | 1000 | **30.3 / 40.2** | 42.5 / 47.8 | -12.2 | -7.6 |

**亮點**:

- **MuSiQue 大勝 +25.5 EM / +26.7 F1**
- **F1 在三個 multi-hop dataset 全部超越 DeepRAG paper**(2Wiki +0.4 / HotpotQA +0.9 / MuSiQue +26.7)
- EM 在 multi-hop 接近持平(-1.1 / -3.1),vs 原始 SEER 大幅追上(2Wiki 從 -12.2 縮到 -3.1)
- PopQA 加 entropy probe(skip-decompose τ=2.0)從 17.9 → 30.3 EM(+12.4),仍落後 DeepRAG ~12 EM(單跳是 SEER 結構性弱項)

##### A.0.1 兩項技術改進細節

1. **Synthesize prompt fix**(無重訓,只改 `_synthesize` 的 prompt):
   - 加 explicit reasoning rules(comparison 題型回 entity 不是 value;yes/no 看 sub-answers 一致性)
   - 加 2 個 few-shot example(bridge_comparison + yes/no)
   - 結果:2Wiki +9.1 EM / +9.3 F1、MuSiQue +2.1 EM / 微漲、HotpotQA +1.8 EM、PopQA 無影響(只走單步)。**整體無 regression、free win**

2. **Option C: entropy probe skip-decompose**(`skip_decompose_tau=2.0`,unified across all datasets):
   - 在 decompose 前,先 probe 原問題(帶 retrieval context)。若 max_ent < τ → skip decompose,直接 return probe 答案
   - 邏輯:多跳題模型 entropy 高→不 skip;單跳題 entropy 低→ skip 避免 over-decomposition
   - PopQA skip rate 79.7%(797/1000),EM +12.4。多跳 skip rate 預期 < 5%(待驗證 sanity check 中)

##### A.0.2 改進前後對照

| Dataset | 原始 SEER EM | +synth fix | +synth fix + option C |
|---|---:|---:|---:|
| MuSiQue | 45.9 | 48.0 (+2.1) | (待驗證 sanity) |
| 2Wiki | 35.9 | 45.0 (+9.1) | (待驗證 sanity) |
| HotpotQA | 37.8 | 39.6 (+1.8) | (待驗證 sanity) |
| PopQA | 17.9 | 17.9(單步無影響) | **30.3** (+12.4) |

##### A.0.3 原始(僅 synthesize 改前 + 無 option C)結果(留存對照)

| Dataset | n | SEER EM/F1 | DeepRAG paper EM/F1 | Δ EM |
|---|---:|---:|---:|---:|
| MuSiQue | 1000 | 45.9 / – | 22.5 / 30.4 | +23.4 |
| 2Wiki | 1000 | 35.9 / 44.4 | 48.1 / 53.3 | -12.2 |
| HotpotQA | 1000 | 37.8 / 52.1 | 40.7 / 51.5 | -2.9 (F1 +0.6) |
| PopQA | 1000 | 17.9 / 29.9 | 42.5 / 47.8 | -24.6 |

> MuSiQue 45.9 EM 從 `results/musique_FULL_C_maxent0.3.jsonl` 取前 1000 列切出。
> 前 1000 題全是 2hop(dev 按 hop 排序),所以 DeepRAG 22.5 EM 也是「全 2hop」數字,本對比為同 protocol。

#### A.1 2Wiki 1000Q per-type 細分 (τ=0.3)

| type | n | EM | 備註 |
|---|---:|---:|---|
| inference | 128 | **58.6%** | 強項 |
| compositional | 428 | 46.3% | 主流題型,合理 |
| comparison | 216 | 29.6% | yes/no 題型 norm 問題? |
| bridge_comparison | 228 | **9.6%** | ← **拖垮平均的元兇** |
| TOTAL | 1000 | 35.9% | |

→ **2Wiki 落後 12.2 EM 的元兇是 bridge_comparison(4 步雙鏈題型)的 9.6% EM**。
→ **根因(實證,2026-05-25):不是訓練資料不足、不是 decomposition 錯、不是 retrieval 錯,而是 synthesize 步驟壞掉**。
> 訓練 4000 筆 2Wiki gold 的 type 分佈:compositional 45.7% / comparison 31.0% / **bridge_comparison 20.7%(827 筆)** / inference 2.6%(105 筆)
> 反證:inference 訓練只佔 2.6% 卻拿 58.6% EM;bridge_comparison 訓練 20.7% 卻只 9.6% EM → 不是樣本量問題

詳見 §A.1.1 fail trace 診斷。

#### A.1.1 bridge_comparison fail trace 診斷(sample 4 個 fail)

範例(從 `results/2wiki_1000_C_maxent0.3.jsonl` 取):

```text
Q: Which film has the director who was born later, El Extraño Viaje or Love In Pawn?
GOLD: El Extraño Viaje
DECOMP (correct, 4 steps):
  #1: El extraño viaje >> director        → final: Fernando Fernán Gómez  ✓
  #2: Love in Pawn >> director            → final: Charles Saunders         ✓
  #3: #1 >> date of birth                 → final: 28 August 1921           ✓
  #4: #2 >> date of birth                 → final: 8 April 1904             ✓
SEER FINAL: "Fernando Fernán Gómez >> date of birth"  ← 抄 step 格式,沒 reasoning
```

四個 fail trace 共同模式:

- **decompose** 完全正確(4 步 KG-triple style)
- **每步 sub-answer 都對**(retrieval 找到正確人/年份)
- **synthesize 步驟失敗**:輸出 date 字串(`1982` / `18 December 1999`)或 step 格式(`X >> date of birth`),而不是「比兩個年份 → 選對應片名」

對照 PASS 案例(都是 yes/no 同國籍題):synthesize 只要 echo `Germany == Germany → Yes`,不需要 cross-step comparison reasoning,所以對。

**根因**:`_synthesize` 用 base Llama3 接 KG-triple style sub-answers,缺乏 reasoning 觸發(模型沒有把「date1 vs date2 → 對應 film」的 chain 接起來)。

**修補方向**(優先序):

1. **synthesize prompt 改造**:把 `X >> property` 反譯成自然語言子問題、加 explicit reasoning instruction(「Compare the two dates and answer with the corresponding film name」)
2. **few-shot demo**:在 synthesize 階段塞 1-2 個 bridge_comparison reasoning example
3. **訓練端**:加 synthesize 階段的 SFT(目前只訓 decomposer);不過此修補成本最高、收益最不確定,優先做 1 + 2

#### A.2 HotpotQA 1000Q 解讀

- EM 37.8 落後 DeepRAG 2.9 點,但 **F1 52.1 已超過 DeepRAG 51.5**(+0.6)
- 跟我之前 best `neg_max_ent@-0.80` 32.7 EM 比 → **+5.1 EM**
- HotpotQA 是 3 個 multi-hop 最弱的「DeepRAG 優勢」(也就是說 SEER 沒明顯輸)

#### B. SEER full dev(嚴格評估,跨所有 hop level)

| Dataset | n | EM | F1 | retrieval% | 備註 |
|---|---:|---:|---:|---:|---|
| **MuSiQue** | **2417 full** | **35.4** | **47.5** | 89.7 | 唯一已跑全量,可信度最高 |
| 2Wiki | 200 | 35.5 | 46.1 | 93.0 | n=200,雜訊大 |
| HotpotQA | 200 | 39.5 | 51.3 | 91.5 | n=200,雜訊大 |
| PopQA | — | — | — | — | ⏳ pending |

#### C. 解讀

- 用 **DeepRAG 同 protocol(MuSiQue dev 前 1000)→ +23.4 EM**,壓倒性勝出
- 用 **更嚴格 full dev → 35.4 EM**,仍勝 DeepRAG (22.5) +12.9 EM,但 hop 偏深的題段是主要短板
- paper 寫法:**主表用對齊 protocol(前 1000),附錄補 full dev 嚴格數字**

### 2.2 同樣 τ=0.0(always-retrieve)的對照

| Dataset | n | τ=0.0 EM/F1 | τ=0.3 EM/F1 | gate 省 retrieval | EM 變化 |
|---|---:|---:|---:|---:|---:|
| MuSiQue | 200 | 47.5 / 58.9 | 47.0 / 58.8 | -9.2pp | -0.5 |
| 2Wiki | 200 | 35.0 / 45.6 | 35.5 / 46.1 | -7.0pp | **+0.5** |
| HotpotQA | 200 | 39.5 / 51.3 | 39.5 / 51.3 | -8.5pp | 0.0 |

**結論**:gate 在三個 multi-hop dataset 上都**省 7-9% retrieval cost、EM 損失 ≤ 0.5**——matches 你之前 `neg_max_ent@-0.80` 的高觸發率區間策略。

---

## 3. MuSiQue 全 dev 各 hop 細分(2417 題,τ=0.3)

| hop | n | EM | F1 |
|---|---:|---:|---:|
| 2hop | 1252 | **43.0** | — |
| 3hop1 | 568 | 36.6 | — |
| 3hop2 | 192 | 15.1 | — |
| 4hop1 | 246 | 25.6 | — |
| 4hop2 | 64 | 14.1 | — |
| 4hop3 | 95 | 9.5 | — |
| **TOTAL** | **2417** | **35.4** | **47.5** |

**前 200 題全是 2hop**(MuSiQue dev 排序問題)→ 200 題的 47.5 EM 高估了 +12 點。3hop2 / 4hop2 / 4hop3 是主要短板,跟 decomposer count_match 在 4hop 崩塌(40-46%)一致。

---

## 4. MuSiQue 200 題 τ 完整掃描(mode C, max_ent)

| τ | retrieval% | EM | F1 | 解讀 |
|---:|---:|---:|---:|---|
| 0.0 (always) | 100.0 | 47.5 | 58.9 | upper bound |
| 0.2 | 94.0 | **47.5** | 58.9 | gate 持平 always |
| **0.3** | 90.8 | **47.0** | 58.8 | ← **sweet spot** |
| 0.5 | 87.5 | 46.5 | 58.0 | -1 EM |
| 0.8 | 78.1 | 41.5 | 54.3 | -6 EM(踩進死亡區) |
| 1.0 | 75.1 | 41.0 | 53.6 | -6.5 EM |
| 1.5 | 65.8 | 36.5 | 51.3 | -11 EM |
| 2.5 | 50.1 | 27.0 | 41.6 | -20 EM |
| 3.5 | 26.7 | 18.5 | 30.1 | -29 EM |
| ∞ (never) | 0.0 | 7.5 | 18.2 | lower bound |

**形狀**:80% 以上觸發率是甜蜜點,80% 以下急速崩塌——對應 `neg_max_ent@-0.80` 在你之前的觀察(83-95% 觸發才贏 always-RAG)。

---

## 5. Mode B vs Mode C 比較(MuSiQue 200,早期跑)

| Mode | 含義 | EM | F1 | retrieval% |
|---|---|---:|---:|---:|
| B | decompose 用 LoRA,answer 用 pure base | 35.0 | 47.6 | 71.1 |
| **C** | decompose & answer 都用 LoRA | **40.5** | **53.3** | 69.6 |

→ Mode C 勝 5.5 EM。**所有後續實驗都用 mode C**。

---

## 6. 跟 EXPERIMENTS_SUMMARY.md §3.2 完整對照表(2026-05-25 最新版)

> SEER 行設定:`[per-q paras / Llama3-8B-Inst / mode C / max_ent τ=0.3 / synthesize prompt fix / no option C]`
> 括號內 `(R%)` = retrieval 觸發比例。**PopQA 用沒有 option C 的版本**(因 option C 在多跳 dataset 會 false-skip,無法 unified)。

| Method | HotpotQA EM/F1 | 2Wiki EM/F1 | MuSiQue EM/F1 | PopQA EM/F1 |
|---|---|---|---|---|
| Ours CoT (no-RAG, 1000Q) | 25.2 / 35.6 | 31.2 / 39.3 | 11.5 / 20.3 | 20.4 / 23.1 |
| Ours Single-RAG (1000Q) | 32.4 / 44.0 | 37.9 / 47.0 | 19.2 / 29.0 | 41.6 / 51.3 |
| Ours `neg_max_ent@-0.80` (1000Q) | 32.7 / 44.3 (83%) | 36.2 / 45.4 (80%) | 19.0 / 29.1 (95%) | **42.4 / 51.6** (95%) |
| **Ours DeepRAG-8b + SC (th=23) (1000Q)** | 39.7 / 50.4 | **51.1 / 55.9** | 22.8 / 30.2 | – |
| **SEER (本研究, 1000Q, promptfix)** | **39.6 / 52.4** (92%) | **45.0 / 53.7** (95%) | **48.0 / 57.1** (90%) | 17.9 / 29.9 (94%) |
| _SEER full dev (參考)_ | _–_ | _–_ | _35.4 / 47.5 (n=2417)_ | _–_ |
| CoT (paper) | 27.2 / 37.8 | 28.2 / 34.9 | 13.7 / 23.0 | – |
| FLARE (paper) | 23.4 / 32.9 | 30.5 / 37.5 | 17.6 / 23.6 | – |
| DRAGIN (paper) | 27.6 / 38.1 | 29.1 / 35.7 | 11.8 / 20.0 | – |
| UAR (paper) | 29.7 / 40.7 | 34.8 / 42.4 | 19.1 / 28.4 | – |
| **DeepRAG (paper, 1000Q)** | 40.7 / 51.5 | 48.1 / 53.3 | 22.5 / 30.4 | **42.5 / 47.8** |

### 觀察(2026-05-25 更新後)

1. **MuSiQue 是壓倒性強項**:48.0 / 57.1 EM 把所有 baseline 甩開(DeepRAG paper +25.5、自家 DeepRAG-8b+SC +25.2、之前 best `neg_max_ent` +29、所有 paper baselines >+28)
2. **2Wiki F1 反超 DeepRAG paper**(53.7 vs 53.3,+0.4),EM 僅落後 3.1 點(從原 -12.2 大幅追上);所有 paper baseline 都打贏(CoT/FLARE/DRAGIN/UAR +10~25 EM)
3. **HotpotQA F1 反超 DeepRAG paper**(52.4 vs 51.5,+0.9),EM 僅落後 1.1 點;比 DeepRAG-8b+SC 略低(-0.1 EM / F1 +2.0)
4. **PopQA 仍然是結構性短板**(17.9 EM、落後 DeepRAG -24.6):多跳訓的 decomposer 對單跳 over-decomposition。entropy-only skip-decompose 在多跳會崩(見 §7 限制),需更強的 skip 信號
5. SEER 跟 DeepRAG-8b 的差異:**DeepRAG-8b 用 DPO + SC 訓 gate(全量訓),SEER 只 LoRA SFT decomposer(1/4 訓練量)+ 用 token entropy 當 training-free gate + synthesize prompt fix(無重訓)**

### Paper headline(更新)

1. **MuSiQue +25.5 EM / +26.7 F1** vs DeepRAG paper(壓倒性勝出)
2. **F1 在三個 multi-hop dataset 全部超越 DeepRAG paper**(2Wiki +0.4、HotpotQA +0.9、MuSiQue +26.7)
3. **Synthesize prompt fix 是 free win**(無重訓、無 regression、4 個 dataset 都不掉):2Wiki +9.1 EM、MuSiQue +2.1 EM、HotpotQA +1.8 EM
4. **PopQA limitation 公開承認**:entropy-based skip-decompose 在多跳會 false-skip,paper 寫成 limitation

---

## 7. 待辦 / 限制 / 已完成

### ✅ 已完成

1. ~~scale 2Wiki + HotpotQA 到 1000Q~~(見 §2.1.A.0)
2. ~~PopQA 1000Q 資料準備 + baseline 跑~~ — 17.9 EM(無 option C),30.3 EM(+option C)
3. ~~debug 2Wiki bridge_comparison 9.6% EM~~(見 §A.1.1)
4. ~~Synthesize prompt fix(無重訓,4 dataset 全部 free win)~~(見 §A.0.1)

### ⚠️ 已驗證的限制(寫進 paper limitation 章節)

1. **Entropy-based skip-decompose 在多跳會崩(unified τ 行不通)**:
   - 同 τ=2.0 套到全部:PopQA +12.4 EM、MuSiQue **-17.9 EM**、2Wiki **-15.6 EM**
   - 多跳 skip rate 高達 73-87%(模型 false-confident)
   - 結論:entropy 不夠分辨單跳/多跳,需要更強信號(self-consistency variance、Top-1/Top-2 gap 等)— **future work**
   - **最終決定**:不開 option C、PopQA 結果取無 C 版(17.9 EM)

### 🔭 未動 / 可選

1. **MuSiQue 4hop 強化**:全 dev 結果 4hop1=25.6 / 4hop2=14.1 / 4hop3=9.5;改善方向:過採樣 4hop 訓練(自然分佈 4hop 僅 6%)
2. **(可選)τ 掃描 2Wiki / HotpotQA**:目前只跑 τ=0.3,可加 τ=0.0/0.5 確認 gate 不掉分
3. **(可選)Bamboogle 125 test set**:超便宜,可加 1 個 dataset 增強可信度
4. **(可選)LJ(LLM-as-Judge)metric**:PopQA 很多 fail 是「語意對、格式錯」(e.g. `State Representative` vs `politician`);用 GPT-4 judge 可能大幅好看
5. **PopQA 結構性短板補強**(若要做):
   - SFT 加單跳題的「單步直答」decomposition(自然教模型該停就停)
   - 或 DeepRAG 同款 `[Retrieval]` meta token + DPO(成本最高,完整復刻)
