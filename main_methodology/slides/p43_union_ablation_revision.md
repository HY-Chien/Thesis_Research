# p43 revision — Original-question Evidence Union Ablation(修正版)

Feedback addressed(老師 6/25 口試檢討):
1. **表格比較對象錯誤** — 原本的表是 full-system before/after(混入 verifier 最後選 direct 的題);
   乾淨的 ablation 應該只看 decompose path 有沒有 union。
2. **「不走上路卻用上路的文章」的講法** — 需要說清楚為什麼這不會讓兩條 path 變成一條。
3. **補上路/下路選路百分比統計**。

All numbers below are recomputed offline from the existing traces by
`scripts/analyze_union_ablation_pathlocal.py` → `results/union_ablation_pathlocal_summary.md`
(N=1000 per dataset; gold = answer + aliases, same scoring as the original runs).

---

## Slide content (English, paste-ready)

### Title: Original-question Evidence Union Ablation

**Goal:** Test whether the **decomposition path** should preserve evidence retrieved with the
original question.

**Evidence union (decomposition path only):** for each sub-question, answer with
`sub-question top-k ∪ original-question top-k` (deduplicated) → merged context → answer the
sub-question. The direct path is unchanged.

**Main table — Effect on the decomposition path (decomposed candidate only):**

| Dataset | w/o union (EM / F1) | + union (EM / F1) | Gain (EM / F1) |
|---|---:|---:|---:|
| PopQA | 51.50 / 53.68 | **52.90 / 55.20** | +1.40 / +1.52 |
| HotpotQA | 25.00 / 33.42 | **30.50 / 39.49** | **+5.50 / +6.07** |
| 2Wiki | 47.20 / 53.63 | **50.20 / 56.65** | +3.00 / +3.03 |
| MuSiQue | 24.70 / 33.10 | **25.50 / 33.40** | +0.80 / +0.30 |
| **Average** | **37.10 / 43.46** | **39.78 / 46.19** | **+2.68 / +2.73** |

> Footnote (small, bottom-right): *Full-system scores also improve (e.g. HotpotQA 31.8→34.3 EM,
> 2Wiki 46.9→49.9), but they include the verifier's final path selection, so they are **not** the
> clean union ablation.* ← 原本 p43 的表移到這裡當小字,不再當主表。

**Path selection statistics (final verifier, direct / decompose):**

| | PopQA | HotpotQA | 2Wiki | MuSiQue | Pooled |
|---|---:|---:|---:|---:|---:|
| direct / decompose | 3.7% / 96.3% | 27.4% / 72.6% | 6.5% / 93.5% | 18.2% / 81.8% | **14.0% / 86.1%** |

**Why this is still two paths** (one-liner for the slide):

> The two paths share a **retrieval pool**, not a reasoning path: sharing is **one-way**
> (decomposition borrows the original-question documents; the direct path never sees anything
> from decomposition), and the decomposition path still runs its own sub-question retrieval,
> step-by-step answering, and final synthesis.

---

## Backup numbers(被追問時用)

**Subset check(老師字面上的口徑:只看最後走 decompose 的題):** 結論方向完全一致,
union 在每個 dataset 都是正增益,HotpotQA 最大、MuSiQue 最小:

| Dataset | N (decomp-routed) | w/o union EM | + union EM | ΔEM |
|---|---:|---:|---:|---:|
| HotpotQA | 726 | 31.13 | **38.57** | +7.44 |
| 2Wiki | 935 | 48.66 | **52.30** | +3.64 |
| MuSiQue | 818 | 28.00 | **28.61** | +0.61 |
| PopQA | 963 | 52.23 | **53.69** | +1.45 |

**Question-level flips on the decomposed candidate(union 救回/弄壞的題數):**

| Dataset | gained (0→1) | lost (1→0) | net |
|---|---:|---:|---:|
| HotpotQA | +80 | −25 | +55 |
| 2Wiki | +84 | −54 | +30 |
| MuSiQue | +43 | −35 | +8 |
| PopQA | +46 | −32 | +14 |

(lost cases 是 future work:rerank / limit union evidence,避免 context 變長帶進干擾。)

**選路比例在三個階段的變化**(union 讓 verifier 更信任 decompose;type-aware replay 再把
一部分題撥回 direct):

| Dataset | before union | after union | final (type-aware) |
|---|---:|---:|---:|
| HotpotQA | 18.1% / 81.9% | 13.2% / 86.8% | **27.4% / 72.6%** |
| 2Wiki | 6.8% / 93.2% | 4.8% / 95.2% | **6.5% / 93.5%** |
| MuSiQue | 8.8% / 91.2% | 7.3% / 92.7% | **18.2% / 81.8%** |
| PopQA | 5.5% / 94.5% | 3.0% / 97.0% | **3.7% / 96.3%** |

---

## Speaker notes(中文講稿,直接回應老師的兩個問題)

**1. 為什麼主表換掉了(比較對象修正):**
Evidence union 這個設計只改 decomposition path 每個 sub-question 的 context,direct path
完全沒動。所以乾淨的 ablation 應該固定看 decomposition path 的輸出(decomposed candidate)
有沒有 union 的差別。原本那張表放的是整個系統的最終分數,裡面混了 verifier 最後選 direct
的題——那些題根本沒用到 union,放進來會稀釋、也會混淆歸因。修正後的主表顯示 union 對
decompose path 本身平均 +2.68 EM,HotpotQA 最大(+5.50),因為 HotpotQA 的原題常帶有
比較詞和 disambiguating entity,拆成 sub-question 後最容易掉。

**2. 這樣還算兩條 path 嗎(回應「都參考同一批文章就沒有兩條 path」):**
我們不是把上路的「答案」或「決策」給下路,而是把「原問題檢索到的文件」補進下路的每一步。
共享是**單向**的:下路借用 `r(q)` 的文件,上路完全看不到下路的任何東西。而且下路仍然有
自己獨立的三件事:sub-question 檢索(title-on)、逐步作答、最後 synthesis——答案的生成
路徑完全不同。所以兩條 path 的差別是「怎麼推理」,不是「看哪個文件池」。這個設計的動機
是 decomposition 會把原題的全域約束拆散(context loss),union 是把丟掉的線索補回來,
定位是 **original-question evidence preservation**,不是 path 合併。

**3. 選路統計(老師要求補的數字):**
最終系統大多數題走下路 decompose(pooled 86%),上路 direct 是 minority path:PopQA 只有
3.7%、2Wiki 6.5%,HotpotQA 最高 27.4%(它的 direct/decompose 兩路互補性最強,2×2 分析裡
direct-only 111 題 vs decompose-only 113 題幾乎打平)。這也直接回答「兩條 path 會不會變成
一條」:如果真的變成一條,verifier 的選擇就不會有資訊量;實際上兩路答案不同、且選路比例
隨 dataset 特性明顯變化。

---

## Data sources & reproduction

- Traces: `results/{ds}_1000_verify_v2_pathaware.jsonl`(w/o union)、
  `results/{ds}_1000_verify_v2_pathaware_directctx.jsonl`(+ union)、
  `results/{ds}_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl`(final,`replay_choice`)。
- Offline analysis: `python scripts/analyze_union_ablation_pathlocal.py` →
  `results/union_ablation_pathlocal_summary.md`。
- Re-run traces: `scripts/run_{ds}_union_ablation.sh`(`run_seek_dual.py --direct-context-union`)。
- 口徑出處:`RESULTS_MASTER.md` §2026-06-23 original-question evidence union。
