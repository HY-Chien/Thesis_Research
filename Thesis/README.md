# NTHU 論文範本 — SEEK 使用指南

本目錄為清華大學 (NTHU) 圖書館 DSLIN 提供之 LaTeX 論文範本,已客製化為 **SEEK (Self-certainty Evaluated Entropy-gated Knowledge-retrieval)** 碩士論文骨架。本檔說明各資料夾用途與撰寫流程。

---

## 目錄結構

```
Thesis/
├── main.tex              ← 主檔,編譯這支
├── nthu_thesis.cls       ← NTHU 文件類別 (字型、間距、TOC、邊界)
├── 2025Titlepage.tex     ← 封面 (中英書名、學號、指導教授、年月)
├── README.md             ← 本檔
│
├── front/                ← 論文前置部分 (摘要、誌謝、官方表單)
├── contents/             ← 論文章節本文 (Chapter 1–6)
├── back/                 ← 附錄與參考文獻
├── figures/              ← 圖檔資源
└── fonts/                ← 中英文字型 (TTF)
```

---

## 各資料夾用途

### `front/` — 前置文件
| 檔案 | 用途 | 是否需修改 |
|------|------|-----------|
| `Eng_Abstract.tex` | 英文摘要 (對應 `\chapter*{Abstract}`) | ✅ 已填入 SEEK 摘要,可微調 |
| `C_Abstract.tex` | 中文摘要 | ✅ 需與英文對齊 |
| `acknowledgement.tex` | 誌謝 (實驗室、指導教授、家人) | ✅ 已填入基本骨架 |
| `Nomenclature.tex` | 符號/縮寫表 (如 RAG、LLM、SFT 等) | ⬜ 視需要新增 |
| `Authorized agreement.pdf` | **學位論文授權書** (3 頁,口試後簽完掃描換上) | 🔴 口試後替換 |
| `Advisor Approval Form.pdf` | **指導教授推薦書** | 🔴 口試後替換 |
| `Thesis Review Form.pdf` | **考試委員審定書** | 🔴 口試後替換 |

> 三份 PDF 由 `main.tex` 用 `\includepdf` 嵌入,**檔名不可改**;直接覆蓋即可。

### `contents/` — 章節本文
| 檔案 | 對應章 | 目前狀態 |
|------|--------|---------|
| `chapter01.tex` | 緒論 (Background / Motivation / Objectives) | ✅ 已有骨架,可再擴寫 |
| `chapter02.tex` | Related Work (Adaptive-RAG / FLARE / DeepRAG + 對照表) | ✅ 已更新為 v2 verifier 定位 |
| `chapter03.tex` | Methodology (SEEK v2: direct path + decompose path → training-free verifier) | ✅ 已完整撰寫 |
| `chapter04.tex` | Experiments (honest wiki-21M 主表、verifier/router ablation) | ✅ 已完整撰寫 (主結果=v2 no-evidence verifier) |
| `chapter05.tex` | Discussion (trigger rate、EM vs F1、failure case) | ✅ 已完整撰寫 |
| `chapter06.tex` | Conclusions & Future Work | ✅ 已完整撰寫 |

> **主結果設定:** 全篇以 **12K 三資料集 decomposer + no-evidence tuned v2 verifier** 為 "SEEK (Ours)"。主表使用 honest wiki-21M pool；distractor pool 只能作開發/消融，不可與 DeepRAG 直接混表。

每章透過 `main.tex` 的 `\input{contents/chapterXX}` 載入,**新增/刪除章節時記得同步 main.tex**。

### `back/` — 後置文件
| 檔案 | 用途 |
|------|------|
| `references.bib` | BibLaTeX 參考文獻資料庫,IEEE 樣式 (`\printbibliography`) |
| `appendix01.tex` | 附錄 A (例如:完整 prompt、超參數、額外實驗) |
| `appendix02.tex` | 附錄 B (目前 main.tex **未載入**,需要時加 `\input{back/appendix02}`) |

引用方式:`\cite{adaptiverag}`、`\cite{flare}`、`\cite{deeprag}` 等 key 已定義於 `references.bib`。

---

## 引用 (citation) 與編號怎麼運作

> **一句話:** 編號永遠由 LaTeX 自動產生,你**永遠不用手動去寫或改 `[1] [2]` 這些號碼**。改完 `\cite` 重編一次就好。

### 編號規則:按「首次引用順序」
`nthu_thesis.cls` 設定 `\usepackage[style=ieee]{biblatex}`,IEEE 樣式預設 `sorting=none`,意思是:**參考文獻依「在全文第一次被 `\cite` 的先後」編號**(`[1] [2] [3]…`),不是字母序。文獻表的排列也跟著這個順序。

### 改 `\cite` 會自動重排,且無害
- **新增引用** → deeprag 之類若被插在更前面首次出現,後面的號碼會自動往後遞補。
- **刪掉某個 `\cite`** → 若那是該文獻的「最早一次」引用,它的號碼會改由下一個引用位置決定,其餘自動補位。
- 以上**全部自動**:只要重新編譯,號碼一律重新算到自洽。文獻只要在**任何一處**還有被 `\cite`,就不會從文獻表消失;完全沒被引用的條目則不會出現(這是正常的,bib 裡放著備用沒關係)。

### 日常操作三步
1. **(只在新增文獻時)** 在 `back/references.bib` 加一筆,例如:
   ```bibtex
   @inproceedings{mykey,
     author    = {Last, First and ...},
     title     = {Paper Title},
     booktitle = {Proceedings of ...},
     year      = {2025}
   }
   ```
2. 在內文要引用處寫 `\cite{mykey}`;一次引多篇用逗號:`\cite{flare,dragin,seakr}`。
3. 跑 `latexmk -pdf -interaction=nonstopmode main.tex` → 號碼與文獻表自動更新。

> **何時會跑 biber:** 只有改了 `references.bib` 或增刪 `\cite` 時(`latexmk` 偵測 `.bcf` 變化自動決定,你不用管)。**只改內文**時不會重跑 biber,號碼也不變。
>
> **想全部換成字母序?** 把 cls 的 `style=ieee` 改成 `sorting=nyt`(name-year-title),號碼就改成依作者字母排、不隨引用位置變動——但那就不是 IEEE 慣例,**不建議**動。

### `figures/` — 圖檔
- `nthu_thesis.cls` 已設定 `\graphicspath{{./figures/}}`,因此 `\includegraphics{name.png}` 會自動從這裡找。
- 目前內含舊範本的範例圖 (`I2adsorption.png`、`ReactProbability.png`),**可全部換成 SEEK pipeline 圖、結果圖**。
- 建議圖檔:
  1. SEEK overall pipeline (decompose → probe → gate → synthesize)
  2. Entropy distribution 直方圖 (retrieve vs skip)
  3. Trigger rate per dataset 長條圖
  4. EM/F1 對 τ 的折線圖 (ablation)

### `fonts/` — 字型
- `chinese/` — BiauKai、Kaiti 系列 (NTHU 標準中文字型)
- `english/` — Times New Roman 四種樣式
- 由 `nthu_thesis.cls` 與 `CJK` 套件自動載入,**通常不需動**。

---

## 我要改什麼 → 去哪改

| 要改的東西 | 去哪改 | 怎麼用 |
|-----------|--------|--------|
| **內文 / 章節** | `contents/chapterXX.tex` | 直接編輯該章 |
| **參考文獻 (新增一篇)** | `back/references.bib` | 加一筆 `@article{key,...}`,內文用 `\cite{key}` 引用 |
| **圖片** | 把圖檔丟進 `figures/`,內文用 `\includegraphics[width=0.8\textwidth]{檔名.png}` | 已設 `graphicspath`,**只寫檔名、不用寫路徑** |
| **摘要 / 誌謝** | `front/Eng_Abstract.tex`、`front/C_Abstract.tex`、`front/acknowledgement.tex` | 直接編輯 |
| **封面 (書名/年月/教授)** | `main.tex` 開頭的 `\newcommand{\titleZH}{...}` 等 | 直接編輯 |
| **三份簽名 PDF** | 覆蓋 `front/` 裡同名 PDF(檔名不可改) | 口試後替換 |

---

## 編譯方式

> 環境已就緒:**TeX Live 2026 (scheme-full) 裝在 `~/texlive/2026`**,含 biber / CJKutf8 / biblatex / pdfpages。

### PATH 設定 (已處理好,僅供參考)
系統預設的 `which pdflatex` 會指到 **conda 殘留的壞版本**,且預設找不到 `biber`。此問題**已永久修正** — 下面這行已寫入 `~/.bashrc`,所以**新開的終端機自動生效,不必再手動設定**:
```bash
export PATH=/home/S113065528/texlive/2026/bin/x86_64-linux:$PATH
```
> 若是修改 `.bashrc` 之前就已開著的舊終端機,先跑一次 `source ~/.bashrc` 即可。驗證:`which latexmk` 應顯示 `.../texlive/2026/...` 而非 `.../miniconda3/...`。

### 一鍵編譯(推薦)
```bash
cd /home/S113065528/Research/Thesis
latexmk -pdf -interaction=nonstopmode main.tex
```
`latexmk` 會**自動**判斷要不要跑 biber、要重跑幾次 pdflatex,一個指令搞定。輸出 → **`main.pdf`**(用 VSCode 直接開即可)。

清掉中間檔(可選):`latexmk -c`(留 PDF) 或 `latexmk -C`(連 PDF 一起刪)。

### 手動四步(latexmk 出問題時的後備)
```bash
BIN=/home/S113065528/texlive/2026/bin/x86_64-linux
$BIN/pdflatex -interaction=nonstopmode main.tex   # 1. 生成 .aux/.bcf
$BIN/biber main                                    # 2. 處理參考文獻
$BIN/pdflatex -interaction=nonstopmode main.tex   # 3. 插入引用
$BIN/pdflatex -interaction=nonstopmode main.tex   # 4. 修正 TOC/交叉引用
```

### 常見疑問
- **每次看都要重編嗎?** 是。`.tex` 是原始碼,要編譯成 PDF 才看得到結果——但用 `latexmk` 一個指令就好,不必手動跑四遍。
- **每次都要重跑 biber 嗎?** 不用。biber 只有在你**改了 `references.bib` 或新增/刪除 `\cite`** 時才需要;`latexmk` 會自動偵測 `.bcf` 變化決定要不要跑,所以你不用自己判斷。只改內文時,它通常只重跑一兩次 pdflatex。
- **只改一個字也要全部重編嗎?** `latexmk` 只在偵測到來源變動時才重編,沒變的話會顯示 "up-to-date" 直接跳過。

---

## 撰寫流程 (建議順序)

### Phase 1 — 範本客製化 (已完成)
- [x] `main.tex` 填入學號、書名、年月、系所
- [x] `front/Eng_Abstract.tex` / `front/C_Abstract.tex` 寫 SEEK 摘要
- [x] `front/acknowledgement.tex` 寫誌謝
- [x] `contents/chapter01.tex` 緒論骨架
- [x] `contents/chapter02.tex` Related Work + 比較表
- [x] `back/references.bib` 補入 SEEK 引用

### Phase 2 — 章節擴寫 (進行中)
1. **Chapter 3 Methodology** — 對應 `Research/main_methodology/seek/pipeline.py`
   - 3.1 Problem Formulation
   - 3.2 Direct path + decomposition path
   - 3.3 Per-step entropy probe and decomposition-path retrieval gate
   - 3.4 Synthesis + value-to-entity guard
   - 3.5 Training-free verifier v2 (no evidence)
2. **Chapter 4 Experiments** — 對應 `Research/main_methodology/RESULTS_MASTER.md` 與 `results/*.jsonl`
   - 4.1 Honest wiki-21M setup
   - 4.2 Main table: v2 verifier vs DeepRAG
   - 4.3 Verifier evidence ablation、global entropy router failure
   - 4.4 Distractor-pool development results、oracle headroom、failure breakdown
3. **Chapter 5 Discussion** — EM vs F1 差距、over-decomposition、failure case
4. **Chapter 6 Conclusions** — 收尾與 future work

### Phase 3 — 圖表
- 把 `figures/` 中舊範本圖移除/取代
- 結果圖可從 `Research/main_methodology/notebooks/` 或 `RESULTS_SEEK.md` 匯出

### Phase 4 — 口試後收尾
- 換入三份簽名 PDF (`Authorized agreement.pdf`、`Advisor Approval Form.pdf`、`Thesis Review Form.pdf`)
- 確認封面 `\monthZH` / `\monthEN` 為實際口試月份
- 跑一次完整編譯 → 上傳圖書館 ETD 系統

---

## 常見坑

| 問題 | 解法 |
|------|------|
| 中文亂碼 | 確認用 `pdflatex` (不是 `xelatex`);CJK 套件需 `BiauKai` 字型 |
| 參考文獻空白 | 漏跑 `biber main`;檢查 `.bcf` 是否生成 |
| `Authorized agreement.pdf` 頁數不對 | `\includepdf` 的 `pages=1-3` 要與實際 PDF 頁數一致 |
| TOC 沒更新 | 至少跑 3 次 `pdflatex`;或用 `latexmk` |
| 圖找不到 | 確認檔在 `figures/`;`\includegraphics` 不要寫副檔名以外的路徑 |
| 行距太寬/太窄 | 在 `nthu_thesis.cls` 改 `\setstretch{1.5}` |

---

## 相關專案路徑

| 來源 | 路徑 |
|------|------|
| SEEK 主程式 | `/home/S113065528/Research/main_methodology/seek/` |
| 實驗結果 | `/home/S113065528/Research/main_methodology/results/` |
| 訓練設定 | `/home/S113065528/Research/main_methodology/configs/` |
| LoRA checkpoints | `/home/S113065528/Research/main_methodology/checkpoints/` |
| LLaMA-Factory | `/home/S113065528/LLaMA-Factory/` |

寫論文時可直接引用上述路徑的數字、圖表、訓練設定。
