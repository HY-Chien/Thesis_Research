# SEEK 論文：編譯與交付指南

本目錄包含目前的 NTHU 碩士論文 LaTeX 原始碼、正式 PDF，以及委員修訂標註版。
2026-08-12 已從乾淨目錄重新驗證：正式版與差異版均可由 bundled sources 完整編譯，
最終沒有缺檔、未定義引用或未定義交叉參照。

## 正式交付檔

| 檔案 | 用途 |
|---|---|
| `main.tex` / `main.pdf` | 正式論文來源與編譯成品 |
| `main_diff.tex` / `main_diff.pdf` | 委員修訂差異標註版與成品 |
| `revision_submission/revision_response.md` | 修訂項目與回覆說明 |
| `2025Titlepage.tex` | 目前正式版使用的封面 |
| `nthu_thesis.cls` | NTHU 論文類別與版面設定 |

目前兩份 PDF 均為 63 頁。若修改任何來源，請重新編譯並確認 PDF 與來源一起提交。

## 來源結構

| 路徑 | 內容 |
|---|---|
| `front/Eng_Abstract.tex` | 英文摘要 |
| `front/C_Abstract.tex` | 中文摘要 |
| `contents/chapter01.tex` … `chapter07.tex` | 第一章至第七章 |
| `back/appendix01.tex` | 附錄 |
| `back/references.bib` | BibLaTeX 參考文獻 |
| `figures/` | 正式論文使用的 PDF/PNG 圖 |
| `figure_sources/` | 13 張正式圖的 source/command manifest 與兩份 recovered Draw.io XML |
| `make_ch3_signal_figs.py` | 產生第三章 Tail Leakage / Rényi 訊號圖的來源腳本 |

`main.tex` 目前不載入 `back/appendix02.tex`、誌謝、Nomenclature 或三份行政表單
PDF。這些檔案可保留作日後使用，但不是目前正式 PDF 的編譯相依項。

## 建議編譯環境

使用含下列工具與套件的完整 TeX Live：

- `latexmk`、pdfLaTeX、Biber；
- `CJKutf8`、`biblatex`、`pdfpages` 及 class 載入的標準套件。

目前正式入口以 `CJK` / pdfLaTeX 排版，請不要直接改用 XeLaTeX。現行
`2025Titlepage.tex` 在 XeLaTeX 下無法解析 `CJK*` 環境，而且替換封面入口會改變
頁數與版面。

repository 內的 `fonts/` 是歷史資產；正式 pdfLaTeX build 不會讀取這些 TTF。
中文楷體由 TeX Live 的 CJK/Arphic 字型提供。

## 一鍵編譯

從本目錄執行：

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error main_diff.tex
```

`latexmk` 會在需要時自動執行 Biber 並重跑 pdfLaTeX。若只要清除中間檔：

```bash
latexmk -c main.tex
latexmk -c main_diff.tex
```

Git 只保留 `main.pdf` 與 `main_diff.pdf` 兩份成品；`.aux`、`.bcf`、
`.run.xml`、`.toc` 等中間檔都由 `.gitignore` 排除。

## 編譯後檢查

至少確認：

1. `main.pdf` 與 `main_diff.pdf` 都已更新；
2. log 沒有 `undefined citation`、`undefined reference` 或 missing file；
3. 目錄、圖表目錄、文獻編號與頁碼正確；
4. 新圖確實位於 `figures/`，大小寫與 `\\includegraphics` 完全一致。

少量字型 fallback、float 位置調整與重複 page anchor 是現行範本的非致命警告；
`main_diff.tex` 的 latexdiff 表格另有極小 overfull box。若出現新的錯誤或未定義
引用，則不應交付。

## 差異版的來源說明

`main_diff.tex` 是已展開的 standalone latexdiff 成品，可直接編譯。檔頭只記錄
baseline 的日期與用途；精確的 baseline source 不在 repository，因此 clone 後可以
重編現有差異版，卻不能只靠目前 repository 精確重新產生同一份 diff。若之後要建立新
diff，請先把選定的 baseline `main.tex` 以清楚檔名存入版控或記錄其 Git commit，
再執行 `latexdiff baseline/main.tex main.tex > main_diff.tex`。

## 與程式結果的關係

論文正式 headline 與修訂分析可從相鄰的 `main_methodology/` 重算：

```bash
cd ../main_methodology
python scripts/analyze_tlsum_fixA_step_gate_final.py
python scripts/analyze_skipped_retrieval_questions.py
python scripts/analyze_accuracy_first_failure_taxonomy.py
```

這三個命令所需的 canonical traces 已納入 Git，輸出寫到
`main_methodology/results/reproduced/`。模型權重、原始資料集與 Wikipedia
Elasticsearch index 只在 live inference 或訓練時需要，不屬於 LaTeX build。

完整的逐實驗、逐表格與全部 13 張圖的命令／資產狀態，請看
[`main_methodology/THESIS_REPRODUCTION.md`](../main_methodology/THESIS_REPRODUCTION.md)
與 [`figure_sources/README.md`](figure_sources/README.md)。`Thesis_Overview.pdf`
的可編輯原始檔目前仍找不到；兩張 prompt 圖則已從 PDF metadata 還原成獨立
`.drawio` 檔，不再只是 rendered PDF。
