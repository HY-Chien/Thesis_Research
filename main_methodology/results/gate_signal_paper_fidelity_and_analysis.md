# Gate Signal 統整分析：MAQA / TARG 論文設定核對 + 四資料集全訊號比較

日期：2026-07-02。對應數據：`results/gate_signal_replay_analysis.md`（4 datasets × 8 signals × n=1000，含 paired-bootstrap 95% CI）。
圖：`results/figures/gate_signal_replay/expA0_signal_distribution.png`、`expA_auroc_need.png`、`expB_budget_curves.png`。

## 1. 動機

審視「為什麼 gate 用 entropy」時的疑慮：印象中 MAQA 與 TARG 兩篇都主張 entropy 較差。
本文件核對 (a) 兩篇論文實際怎麼說、(b) 我們的實作是否忠於論文設定、(c) 在我們的
sub-question 設定下四個資料集的完整比較結果。

## 2. 論文核對

### MAQA (Yang, Yoo & Lee, NAACL 2025 Findings, arXiv:2408.06816)

- **性質**：UQ 評測 benchmark（multi-answer QA 的 data uncertainty），**不是** adaptive-retrieval 論文。
- **margin 定義**：第一個 answer token 的 top-1 − top-2 **softmax probability** margin（"we use
  the logit values of the first token of each answer"；M = p_max1 − p_max2）。
- **實作核對：一致**。`seek/answerer.py` 的 `maqa_margin` = 第一個非 EOS 生成 token 的
  top1−top2 softmax 機率差。
- **論文結論：entropy 是 white-box 方法中最好的**，margin 較差（world knowledge 多答案任務
  AUROC：entropy 71.18 vs margin 63.82）。⇒「MAQA 說 entropy 不好」是誤記；
  我們 entropy > maqa_margin 的結果與 MAQA 論文**同向**。

### TARG (arXiv:2511.09803, "Retrieval as a Decision")

- **性質**：training-free 檢索 gating。對原問題先做 no-context greedy draft，取前 k 個 token
  的 logits 算 uncertainty，超過 τ 才檢索。
- **margin 定義**：U_mar = (1/k) Σ exp(−g_t/β)，g_t = top1−top2 **logit** gap；預設 k=20、β=3。
- **實作核對：一致**。`answerer.py` 的 `targ_unc` = prefix-mean exp(−logit_gap/β)，sweep 與
  replay trace 都用 `--targ-prefix-k 20 --targ-beta 3.0`，且同樣在無 context 的 tentative
  answer 上計算。`targ_logit_gap`（raw gap 平均）為輔助變體。
- **論文主張**：instruction-tuned 模型分佈過尖 → prefix **mean entropy** 失去鑑別力、
  over-trigger；margin 較佳。實驗多為單跳（NQ-Open、TriviaQA、PopQA；Qwen2.5-7B、
  Llama-3.1-8B-Instruct）。
- **與我們設定的差異**（解讀分歧的關鍵）：
  1. TARG 的 draft 是對整題的自由續寫（≥20 tokens）；我們的 span 是 answer-phrase-only
     （常僅 2–5 tokens），prefix 平均涵蓋的 token 結構不同。
  2. TARG 只比 **mean** entropy；我們的主訊號是 **max_ent**（逐 token 熵的最大值），
     TARG 未測。且在我們的設定連 mean_ent 都贏 targ_unc（見下表）。
  3. TARG 是 question-level 單次 gate；我們是 decomposition 後 per-sub-question gate。

## 3. 四資料集完整結果（AUROC(need)，n=1000/dataset）

| signal | hotpotqa | 2wiki | musique | popqa | macro |
|---|---:|---:|---:|---:|---:|
| **max_ent** | **0.684** | **0.541** | 0.783 | 0.749 | **0.689** |
| min_lp | 0.659 | 0.401 | **0.790** | **0.758** | 0.652 |
| mean_ent | 0.625 | 0.436 | 0.773 | 0.717 | 0.637 |
| maqa_margin | 0.596 | 0.416 | 0.750 | 0.705 | 0.617 |
| targ_logit_gap | 0.549 | 0.437 | 0.695 | 0.650 | 0.583 |
| mean_margin | 0.572 | 0.369 | 0.730 | 0.651 | 0.580 |
| targ_unc | 0.562 | 0.378 | 0.719 | 0.644 | 0.575 |
| max_margin | 0.522 | 0.448 | 0.643 | 0.621 | 0.559 |

Bootstrap 檢定（1000 resamples，paired；詳見 replay md）：

- **hotpotqa**：max_ent 顯著贏所有 margin/TARG 訊號（vs maqa_margin Δ=+0.088
  [+0.040,+0.137]；vs targ_unc Δ=+0.122 [+0.065,+0.180]）。
- **popqa**（TARG 自己的資料集）：entropy family 顯著贏 targ_unc（Δ=+0.106
  [+0.059,+0.152]，P(Δ≤0)<0.001）— 在共同資料集上直接反轉 TARG 的結論。
- **musique**：min_lp ≈ max_ent ≈ mean_ent（差異不顯著）；maqa_margin 接近
  （Δ CI 含 0）；targ/mean/max_margin 顯著較差。
- **2wiki**：全部訊號都爛（多數 AUROC < 0.5，比隨機還反向）；max_ent 只是「最不爛」。
  2wiki 的 gate 訊號問題是資料集本身（見 RESULTS_MASTER §4.6 既有觀察），不是訊號選擇問題。

## 4. 結論：為什麼選 entropy 站得住腳

1. **與 MAQA 一致**：MAQA 論文本來就說 entropy 最好；我們複現其 margin 設定後也得到同向結果。
2. **TARG 的主張在我們的設定不成立**：忠實複現（k=20, β=3, no-context）後，targ_unc 在
   四個資料集全輸 max_ent，包括 TARG 自己用的 PopQA。合理解釋：TARG 的訊號是為「整題
   20-token 自由續寫 draft」設計；我們的 gate 作用在「分解後 sub-question 的短 answer span」，
   span 短時 prefix-mean 幾乎退化為 1–3 個 token 的平均，而 max_ent 恰好抓到 span 內最不確定
   的那個 token（通常是 entity 的第一個 token），鑑別力更強。
3. **max_ent 是 macro 最佳訊號**（0.689），且在沒有任何資料集上被其他訊號顯著擊敗
   （musique/popqa 上 min_lp 略高但差異不顯著，Δ CI 皆含 0）。
4. 論文寫法建議：不要說「文獻說 entropy 差」——正確的敘事是「MAQA 支持 entropy；TARG 在
   單跳 draft 設定主張 margin，但我們在 sub-question 短 span 設定下忠實複現其訊號
   （k=20, β=3），entropy 仍顯著較佳（含 TARG 自用的 PopQA），故採 max_ent」。

## 5. Caveats

- Replay 是 question-level 模擬（整題全檢索或全不檢索），部署 gate 是 per-step；
  step-1-only AUROC 欄位為 chain-independent 的 robustness check，排序結論相同。
- End-to-end tau sweep（deployment 操作點）另由 `run_dual_margin_sweep.sh` 粗格點
  （每訊號 5–7 個 τ）進行中，預計 7/3–7/4 補齊 musique/popqa。
- 2wiki 上所有訊號皆近於或低於隨機，任何 gate 在該資料集都應搭配 always-RAG 保底。

## 6. 產物

- 數據：`results/gate_signal_replay_analysis.md`（重跑：`python scripts/replay_gate_signal_analysis.py --figures`）
- 新 trace：`results/musique_dual_targ_unc_tau0p10.jsonl`、`results/popqa_dual_targ_unc_tau0p10.jsonl`
  （命名相容 sweep，會被 skip 邏輯視為已完成）
- 腳本變更：`replay_gate_signal_analysis.py` 加入 paired bootstrap CI（AUROC CI、ΔAUROC vs
  max_ent、P(Δ≤0)）；`run_dual_margin_sweep.sh` 改粗格點（原密格點留在註解，可無縫加密）。
