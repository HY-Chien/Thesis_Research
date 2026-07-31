# Research — SEEK 碩士論文專案

SEEK(Self-certainty Evaluated Evidence-gated Knowledge-retrieval):
sub-question 層級、以 tail-leakage surprisal 為信心訊號的 adaptive RAG。

| 目錄 | 內容 | 入口文件 |
|------|------|---------|
| `main_methodology/` | 實驗程式碼、結果、正式數據彙整 | **`main_methodology/README.md`(復現指南)**、`RESULTS_MASTER.md`(權威結果) |
| `Thesis/` | 論文 LaTeX 原始檔與成品(main.pdf / 差異標註 main_diff.pdf) | `Thesis/README.md` |
| `deeprag_compare/` | DeepRAG 對照重跑 | — |
| `experiment/` | 早期實驗 | — |

- 環境:conda env `thesis`(規格見 `main_methodology/environment_thesis.yml`)
- 注意:`.gitignore` 排除 jsonl/log/權重等大檔;完整清單與打包方式見復現指南 §2
- `environment.yml`(root)是舊的 `prag` 環境,SEEK 復現請用 `environment_thesis.yml`
