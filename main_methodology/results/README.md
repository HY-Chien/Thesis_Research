# SEEK 結果與最小復現資料

Git 內附一份約 **49 MB（47 MiB）** 的 canonical minimal trace bundle，讓 reviewer
在 clone repository 後，不需要 GPU、Llama 權重或 Elasticsearch，即可離線重算
論文的 Tail Leakage threshold analysis 與最終 EM/F1。

## Clone 後直接重算

從 `main_methodology/` 執行：

```bash
python scripts/analyze_tlsum_fixA_step_gate_final.py
```

腳本會直接讀取下列 bundled files，不需要建立 symlink 或另外指定輸入路徑。
產物寫入：

```text
results/reproduced/tlsum_fixA_step_gate_final.md
results/reproduced/tlsum_fixA_step_gate_final.json
```

`results/reproduced/` 是本機產生目錄，不納入 Git。accuracy-first 操作點應重現：

```text
Tail Leakage tau = 0.002305
Macro EM / F1    = 41.88 / 48.90
```

## Git 內附的正式結果檔案

令 `{ds}` 為 `hotpotqa`、`2wiki`、`musique` 或 `popqa`。

| 類別 | 路徑樣式 | 數量 | 用途 |
|---|---|---:|---|
| Final dual-path traces | `traces/{ds}/{ds}_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl` | 4 | 題目、gold、direct/decomposition 路徑與 verifier 所需欄位 |
| Direct confidence traces | `traces/{ds}/conf_battery_{ds}.jsonl` | 4 | 由 token probabilities 重算 direct-path Tail Leakage |
| Tail Leakage step traces | `traces/{ds}/tlsum_fixA_{ds}_shard{0,1}.jsonl` | 8 | 每一步 tentative/retrieved answer、`tl_sum` 與 `post_tl_sum` |
| Fixed-chain syntheses | `traces/other/tlsum_fixA_step_gate_synth_shard{0,1}.jsonl` | 2 | 各 threshold 對應的 final synthesis，供離線重播 |

共 18 個 JSONL。這是重算正式 accuracy-first 結果所需的最小集合，而不是所有研究
期間產生的 traces。

此外，修訂版論文的 failure taxonomy 需要下列 6 個 coverage traces（HotpotQA、
2Wiki、MuSiQue 各兩個）：

| 類別 | 路徑樣式 | 數量 | 用途 |
|---|---|---:|---|
| Original-query coverage | `traces/{ds}/{ds}_m3_wiki_detail.jsonl` | 3 | 原始問題 supporting-title coverage@10 |
| Decomposed-query coverage | `traces/{ds}/{ds}_final_tlsum_bunion_coverage.jsonl` | 3 | 固定 resolved sub-questions 的 union coverage@10 |

clone 後可再執行：

```bash
python scripts/analyze_skipped_retrieval_questions.py
python scripts/analyze_accuracy_first_failure_taxonomy.py
python scripts/analyze_confidence_battery.py
python scripts/analyze_tail_leakage_variants.py
python scripts/analyze_tlsum_confidence_story.py
```

前者只使用原本 18 個 canonical traces；後者使用同一組 traces 加上上述 6 個
coverage traces。後三者使用四份 bundled direct-confidence traces。這些現役
重現命令的輸出都寫入 `results/reproduced/`。

## `results/` 目錄

| 位置 | 內容 |
|---|---|
| `traces/{hotpotqa,2wiki,musique,popqa}/` | 依資料集分層的實驗 traces；Git 只追蹤上表的 canonical subset |
| `traces/other/` | 跨資料集的 synthesis 或 manifest；Git 只追蹤上表的兩個 synthesis shards |
| `reports/` | confidence、failure taxonomy、threshold sweep 等分析報告 |
| `tables/` | 最終表格與 machine-readable summaries |
| `figures/` | 由分析產生的圖 |
| `reproduced/` | reviewer 在本機重算的輸出（gitignored） |

完整實驗曾產生數百個歷史 sweep、ablation traces 與 logs，總量遠大於最小 bundle，
因此仍由 `.gitignore` 排除。這些大型歷史檔案不是執行上方 canonical reproduction
command 的必要條件。研究過程與其他消融結果的彙整可查閱上層
`../RESULTS_MASTER.md`；腳本索引位於 `../scripts/README.md`。
