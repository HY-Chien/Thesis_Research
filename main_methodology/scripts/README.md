# scripts/ 索引

> 2026-07-31 整理。檔名**未改動**(`.sh` 以 `python scripts/xxx.py` 呼叫、RESULTS_MASTER.md 也引用原名,改名會斷)。
> 執行紀錄(`.log`/`.done`/`.out`/`.nohup`)已移至 `../logs/script_runs/`。

## Code-review 起點（現役）

| 目的 | 入口 |
|---|---|
| Live dual-path SEEK | `run_seek_dual.py`（`--signal tail_leakage`） |
| 一鍵執行完整離線驗證 | `run_offline_reproduction.sh` |
| 單一路徑 / gate 診斷 | `run_seek.py` |
| 重建四份 1,000 題 evaluation JSONL | `prepare_eval_subsets.py` |
| 產生 Tail Leakage 雙分支預測結果 | `rerun_tlsum_fixA.py` |
| 固定鏈重新 synthesis | `replay_tlsum_fixA_step_gate_synthesis.py` |
| 重現 41.88 EM / 48.90 F1 | `analyze_tlsum_fixA_step_gate_final.py` |

只想檢查或執行正式方法時，先看上表與上層 `README.md`。其餘檔案保留作
論文 ablation、diagnostic 與歷史實驗紀錄，不是 quick-start
必跑步驟。

## 命名慣例(前綴 = 用途)

| 前綴 | 意義 |
|------|------|
| `run_*` | 執行 pipeline / 實驗(`.py` 為主程式,`.sh` 為 GPU 啟動器) |
| `probe_*` | 跑模型收集信心訊號(需 GPU) |
| `analyze_*` | 讀取已儲存的預測結果並產出分析報告（現役重現輸出到 `results/reproduced/`；歷史腳本可能仍指 `results/reports/`） |
| `replay_*` / `rerun_*` | 離線重播 verifier / gate 決策,不重跑生成 |
| `sweep_*` | 閾值掃描 |
| `make_*` / `plot_*` | 畫圖(部分直接輸出到 `Thesis/figures/`) |
| `build_*` / `prepare_*` / `prep_*` / `sample_*` / `filter_*` | 資料集與 SFT 訓練資料準備 |
| `recall_*` / `recompute_*` | 檢索品質評估 |
| `eval_*` / `test_*` | decomposer 評估 |

## 其他核心與支援檔案

- `run_seek.py` / `run_seek_dual.py` — SEEK 主 pipeline(單路徑 / dual-path);核心邏輯在 `../seek/`(`pipeline.py`, `answerer.py`, `retriever.py`, `verifier.py`)
- `rescue_value_answers.py` — comparison 題 value→entity 離線修正(論文 §synthesis guard)
- `qwen25_decomposer_lora.yaml` — Qwen-2.5-7B cross-backbone 訓練設定(LLaMA-Factory)

## 主要實驗家族(對照 RESULTS_MASTER.md 章節)

| 家族 | 相關檔案 |
|------|---------|
| 信心訊號 battery | `probe_confidence_battery*.py`, `analyze_confidence_battery.py`, `run_conf_battery_*.sh` |
| tail-leakage (tl_sum) | `analyze_tail_leakage_variants.py`, `analyze_tlsum_*.py`, `probe_direct_norag_tlsum.py`, `run_tlsum_fixA_*.sh` |
| step gate / τ sweep | `sweep_step_gate_signals.py`, `probe_step_gate_battery.py`, `run_tau_sweep.sh`, `tau_sweep_orchestrator.sh`, `seer_tau_table.py` |
| union / title ablation | `analyze_union_ablation_pathlocal.py`, `run_*_union_ablation.sh`, `run_tlsum_title_ablation_tau002305.sh` |
| 3-way RAG ablation | `ablate_rag_usefulness.py`, `run_3way_ablation.sh`, `make_3way_fig*.py` |
| verifier replay | `replay_*_verifier.py`, `replay_verifier_*.py`, `analyze_quadrant_confidence.py` |
| 論文圖表 | `make_answer_hit_fig.py`, `make_direct_decompose_2x2_fig.py`, `make_routing_gap_fig.py`, `make_threshold_*_fig.py` |

目前論文逐表／逐圖的精確命令、輸入、輸出與 external-asset 狀態統一列在
上層 [`THESIS_REPRODUCTION.md`](../THESIS_REPRODUCTION.md)。其中 path overlap、
routing 與 quadrant 圖共用 `accuracy_first_replay.py`，一律從 Git 內已儲存的
逐題結果重建 `tau=0.002305` 候選，不使用舊 threshold 的硬編碼數字。
