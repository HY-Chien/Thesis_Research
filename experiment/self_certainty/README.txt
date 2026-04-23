===========================================================
  Self-Certainty (SC) 實驗目錄說明
  /home/S113065528/Adaptive-RAG/experiment/self_certainty/
===========================================================

目錄結構：
  scripts/   — Python 程式碼
  results/   — 實驗輸出的 JSON 資料
  logs/      — 執行 log
  figures/   — 圖片輸出

===========================================================
scripts/ — 程式碼
===========================================================

[主要實驗]

exp0_sc_vs_ppl_2wiki.py
  ★ 目前主力實驗腳本（多資料集版）
  SC vs PPL 在 Best-of-N 選取上的比較。
  每題 sample N 個 CoT 候選答案，用 Borda 投票（p=0.3）選最佳答案。
  支援 --dataset 參數：2wiki / nq / hotpotqa / musique / squad / trivia
  用法：python exp0_sc_vs_ppl_2wiki.py --dataset nq --max_questions 250 --n_candidates 8

exp1_stability.py
  SC vs PPL 在問題改寫（paraphrase）下的穩定性實驗。
  固定模型答案 A，分別用「原始問題」「改寫問題」「擴展問題」計算 SC 與 PPL，
  觀察哪個指標對 prompt 形式變化更不敏感（更穩定）。

exp2_best_of_n.py
  早期版本的 Best-of-N 實驗（NQ，direct answer prompt）。
  已被 exp0_sc_vs_ppl_2wiki.py 取代，保留作為參考。

[輔助腳本]

exp0_sc_vs_ppl_nq.py
  舊版 NQ 直接回答（non-CoT）實驗的 placeholder，已封存不再使用。

exp0_sc_vs_ppl_multi.py
  批次跑多個資料集的舊版腳本，已被 exp0_sc_vs_ppl_2wiki.py 的 --dataset 參數取代。

[繪圖腳本]

plot_exp1_by_correctness.py
  把 exp1 結果依答案正確/錯誤分組，畫 SC 與 PPL 的穩定性比較圖。
  輸出：figures/exp1_stability_by_correctness.png

plot_exp2.py
  畫 exp2 結果（舊版 Best-of-N NQ）的圖。

plot_multi_results.py
  畫多資料集批次比較的圖（AUC、accuracy summary table）。

plot_results.py
  舊版通用繪圖腳本，早期開發用。

[其他]

rewrite_sc.py
  針對改寫問題計算 SC 的早期實驗腳本（exp1 的前身）。

self_certainty_filter.py
  過濾/篩選 SC 分數的工具，找出 SC 極端值（flagged samples）。

===========================================================
results/ — JSON 資料
===========================================================

[exp0 系列：Best-of-N SC vs PPL]

exp0_nq_results.json
  ★ NQ 最新實驗結果（250題 x N=8，CoT+Borda）
  包含每題所有候選答案的 sc, ppl, sc_per_token, ppl_per_token

exp0_nq_summary.json
  ★ NQ 最新實驗摘要（各方法準確率、AUC）

exp0_sc_vs_ppl_2wiki_results.json
  2WikiMultiHopQA 完整實驗結果（250題 x N=8）

exp0_sc_vs_ppl_2wiki_summary.json
  2WikiMultiHopQA 實驗摘要

exp0_2wiki_original_results.json
  exp1 用：2wiki 原始問題的 SC/PPL 分數

exp0_2wiki_rewritten_results.json
  exp1 用：2wiki 改寫問題的 SC/PPL 分數

exp0_nq_original_results.json
  exp1 用：NQ 原始問題的 SC/PPL 分數

exp0_nq_rewritten_results.json
  exp1 用：NQ 改寫問題的 SC/PPL 分數

exp0_sc_vs_ppl_nq_results.json
  舊版 NQ 直接回答（non-CoT）實驗結果，已封存

exp0_sc_vs_ppl_results.json
  最早期的小量測試結果，已封存

exp0_multi_summary.json
  多資料集批次實驗的摘要表格

[exp1 系列：穩定性實驗]

exp1_stability_results.json
  NQ 穩定性實驗完整結果：每題含 sc_orig, sc_rewritten, sc_expansion,
  ppl_orig, ppl_rewritten, ppl_expansion, fixed_answer

[exp2 系列：早期 Best-of-N]

exp2_nq_results.json
  舊版 exp2 NQ 結果（non-CoT，已被 exp0 取代）

exp2_summary.json
  舊版 exp2 摘要

[其他]

rewrite_sc_comparison.json
  早期改寫 SC 比較的原始資料（exp1 前身）

self_certainty_scores.json
  早期少量測試的 SC 原始分數

self_certainty_scores_flagged.json
  self_certainty_filter.py 標記的異常 SC 樣本

===========================================================
logs/ — 執行 log
===========================================================

exp0_nq.log            NQ 最新實驗（250題 x N=8）執行過程
exp0_sc_vs_ppl_2wiki.log  2wiki 250題實驗執行過程
exp0_multi.log         多資料集批次實驗執行過程
exp0_sc_vs_ppl.log     早期測試執行過程
exp1_stability.log     exp1 穩定性實驗執行過程
exp2_best_of_n.log     exp2 舊版實驗執行過程
rewrite_sc.log         改寫 SC 早期腳本執行過程
self_certainty_filter.log  過濾腳本執行過程

===========================================================
figures/ — 圖片
===========================================================

[exp0：Best-of-N 分布與比較]

exp0_sc_vs_ppl_distribution_2wiki.png
  2wiki 所有 candidate 的 SC / -PPL 分布圖（正確 vs 錯誤）

exp0_sc_vs_ppl_distribution_nq.png
  NQ 版本的分布圖

exp0_accuracy_comparison_2wiki.png
  ★ 2wiki 各選取方法準確率比較 bar chart（無 Greedy）

exp0_accuracy_comparison_nq.png
  ★ NQ 各選取方法準確率比較 bar chart（無 Greedy）

exp0_token_trajectories_2wiki.png
  2wiki 正確/錯誤 candidate 的 per-token SC 與 NLL 軌跡

exp0_token_trajectories_nq.png
  NQ 版本的 per-token 軌跡圖

[exp1：穩定性]

exp1_stability.png
  SC 與 log-PPL 在原始 vs 改寫問題的散點圖 + 中位數相對變化量

exp1_stability_comparison.png
  同時比較 rewritten 與 expansion 兩種改寫方式的穩定性

exp1_stability_by_correctness.png
  依答案正確/錯誤分組的穩定性比較（SC 穩定 ~5%，PPL 不穩定 ~60%）

[exp2：舊版 Best-of-N]

exp2_accuracy_comparison.png  舊版 exp2 準確率比較
exp2_nq_distribution.png      舊版 exp2 NQ 分布圖
exp2_summary_table.png        舊版 exp2 摘要表

[其他]

exp0_sc_vs_ppl_distribution.png  最早期測試分布圖
exp0_separation.png              早期 SC/PPL 分離度圖
exp0_token_trajectories.png      早期 token 軌跡圖
exp0_multi_auc_comparison.png    多資料集 AUC 比較
exp0_multi_auc_stability.png     多資料集 AUC 穩定性圖
exp0_multi_summary_table.png     多資料集摘要表
showcase_sc_stable_ppl_not.txt   SC 穩定性展示樣本文字說明

===========================================================
