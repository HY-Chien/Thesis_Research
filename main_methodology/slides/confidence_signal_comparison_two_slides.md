# Confidence Signal Comparison — 2-slide Draft

Purpose: summarize alternative confidence/retrieval-gate signals (margin, TARG, etc.) for advisor discussion.

---

## Which Confidence Signal Best Predicts Retrieval Need?

### Max entropy is the strongest macro signal in our sub-question setting

**Metric:** AUROC(need) = how well the signal ranks questions where retrieval helps  
Higher is better; 0.5 = random. n=1000 per dataset.

| Signal | Definition | HotpotQA | 2Wiki | MuSiQue | PopQA | Macro |
|---|---|---:|---:|---:|---:|---:|
| **max_ent (SEEK)** | max token entropy over the generated sub-answer | **0.684** | **0.541** | 0.783 | 0.749 | **0.689** |
| min_lp | worst token log-probability | 0.659 | 0.401 | **0.790** | **0.758** | 0.652 |
| mean_ent | mean token entropy | 0.625 | 0.436 | 0.773 | 0.717 | 0.637 |
| MAQA margin | first-token top1-top2 probability margin | 0.596 | 0.416 | 0.750 | 0.705 | 0.617 |
| TARG uncertainty | prefix mean exp(-logit gap / beta), k=20, beta=3 | 0.562 | 0.378 | 0.719 | 0.644 | 0.575 |
| max_margin | max token margin variant | 0.522 | 0.448 | 0.643 | 0.621 | 0.559 |

- **max_ent is best on macro AUROC** and is not significantly beaten by any other signal.
- Margin/TARG signals are weaker in this setting; they were designed for full-question drafts, while SEEK gates short decomposed sub-answers.
- **2Wiki is hard for every signal**: confident-but-wrong sub-answers make retrieval need close to random.

---

## End-to-End Accuracy: Margins Do Not Improve SEEK

### Tuned margin gates only match accuracy by retrieving almost always

**Metric:** final dual-path SEEK EM/F1 after sweeping each signal's threshold.  
Rows use the best completed 4-dataset operating point for each signal.
`min_lp` is omitted here because only replay/AUROC results are available; no completed dual-path tau sweep was run for it.

| Gate signal | Best tau | EM | F1 | Retrieval rate | Interpretation |
|---|---:|---:|---:|---:|---|
| **max_ent (SEEK)** | **0.30** | **40.52** | **47.61** | 94.2% | Best stable canonical setting |
| mean_margin | 0.99 | 40.4 | 46.9 | 97.1% | Similar EM, lower F1, more retrieval |
| MAQA margin | 0.998 | 40.3 | 46.9 | 99.1% | Near always-retrieve; no accuracy gain |
| TARG uncertainty | 0.10 | 39.4 | 46.4 | 97.5% | Lower than max_ent despite more retrieval |
| max_margin | 0.998 | 30.1 | 35.9 | 49.7% | Too aggressive; accuracy collapses |

- At the same nominal tau=0.3, margins under-retrieve and hurt multi-hop accuracy; after tau sweep, their best points require **95-99% retrieval**.
- Therefore margin signals are useful as ablations, but **not better replacements** for SEEK's max-entropy gate.
- Final wording: **we keep max_ent because it gives the best accuracy-cost trade-off in the decomposed sub-question setting.**

---

## Speaker Notes

這兩頁的口頭主線可以這樣講:

第一頁我不是直接比較最終 EM,而是先問 signal 本身會不會把「需要檢索的題」排在前面。這裡用 AUROC(need),0.5 是隨機。結果 max entropy 的 macro AUROC 是 0.689,是所有訊號裡最高。MAQA margin 是 0.617,TARG uncertainty 是 0.575,都比較低。這說明在 SEEK 的 decomposed sub-question 設定下,最大 token entropy 比 margin 更適合當 retrieval gate。

### Signal definitions

- **max_ent (SEEK):** 看生成答案過程中「最不確定的那個 token」的 entropy。直覺是:短答案只要有一個關鍵 entity token 很不確定,整個 sub-answer 就有風險,應該檢索。
- **min_lp:** 看生成答案裡機率最低的 token。它和 max_ent 很像,也是抓最弱的一個 token,但用的是 token log-probability,不是整個分布的 entropy。
- **mean_ent:** 把答案所有 token 的 entropy 平均。它量的是整段答案的平均不確定性,但在短 answer span 裡,平均會把關鍵 token 的高不確定性稀釋掉。
- **MAQA margin:** 只看第一個答案 token 的 top-1 和 top-2 softmax probability 差距。差距小代表模型在前兩個候選 token 間猶豫,所以較不確定。
- **TARG uncertainty:** 看答案前綴多個 token 的 top-1/top-2 logit gap,再轉成 uncertainty 後平均。這是 TARG 論文的 training-free gate 設計,原本主要用在原問題的 longer draft。
- **max_margin:** margin 類訊號的變體,取答案 token 中最不穩的 margin。理論上想模仿 max_ent 抓 worst token,但實驗上 macro AUROC 最低。

第二頁看 end-to-end accuracy。這裡我有對每個 signal sweep threshold,不是只拿 tau=0.3 硬比。結果 margin 類訊號最佳點大多要檢索 95% 到 99% 的 step,幾乎變成 always retrieve,但 EM/F1 仍沒有超過 max entropy。MAQA margin 最好約 40.3/46.9,TARG 約 39.4/46.4,都低於 max_ent 的 40.52/47.61。所以結論不是 margin 沒有用,而是它不能取代目前的 max_ent gate。

如果老師問 TARG 為什麼和論文不同,可以說: TARG 的設定是原問題 no-context draft,取 20 token prefix 的 logit gap;我們的 gate 是 decomposition 之後的短答案 span,常常只有幾個 token。對短 answer span 來說,max_ent 抓到最不確定的 entity token,比 prefix-average margin 更敏感。

