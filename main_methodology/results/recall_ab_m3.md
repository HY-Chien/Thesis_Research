# M3 — Does decomposition retrieve the gold passages better than the original question?

**Setup.** For each question we retrieve from its own distractor paragraph pool
with BM25 and measure recall of the gold (`is_supporting`) passages.

- **A** — original full question, one query, top-k.
- **B-matched** — SEER sub-questions; each paragraph scored by its *best-matching*
  sub-q, global top-k. **Same k budget as A → the fair comparison.**
- **B-union** — each sub-q takes its own top-k, then union (what SEER actually
  feeds the answerer; budget up to `n_subq × k`, shown for context only).

Decomposition trace = canonical 12K three-source decomposer (`C_maxent0.3_3ds_terse`).
Gold = 2 supporting passages/question. `recall` is micro (over gold passages);
`full-cov` = fraction of questions with **all** gold passages retrieved.

PopQA is **N/A**: single-hop, no `is_supporting` gold labels.

## Results (recall@k / full-coverage@k, 1000 q each)

### 2WikiMultiHopQA (pool ~10)
| k | A recall | B-match | B-union | A full-cov | B-match | B-union | Δrecall (Bm−A) |
|---|---|---|---|---|---|---|---|
| 3 | 63.8% | **82.6%** | 96.7% | 34.5% | **63.4%** | 92.7% | **+18.9** |
| 5 | 75.7% | **95.4%** | 98.3% | 51.6% | **89.6%** | 96.2% | **+19.7** |
| 10 | 100% | 100% | 100% | 100% | 100% | 100% | +0.0 † |

### HotpotQA (pool ~10)
| k | A recall | B-match | B-union | A full-cov | B-match | B-union | Δrecall (Bm−A) |
|---|---|---|---|---|---|---|---|
| 3 | 71.0% | **78.2%** | 88.5% | 46.1% | **60.3%** | 79.1% | **+7.2** |
| 5 | 81.4% | **87.1%** | 94.2% | 64.8% | **76.2%** | 89.1% | **+5.7** |
| 10 | 100% | 100% | 100% | 100% | 100% | 100% | +0.0 † |

### MuSiQue (pool ~20)
| k | A recall | B-match | B-union | A full-cov | B-match | B-union | Δrecall (Bm−A) |
|---|---|---|---|---|---|---|---|
| 3 | 46.4% | **70.0%** | 82.6% | 15.5% | **53.0%** | 72.2% | **+23.6** |
| 5 | 56.1% | **78.0%** | 88.0% | 27.0% | **64.5%** | 79.8% | **+21.9** |
| 10 | 71.8% | **87.5%** | 94.2% | 48.7% | **78.0%** | 89.1% | **+15.7** |

† k=10 is trivial on 2wiki/hotpot — the pool is only 10 passages, so top-10 = whole
pool. Only k=3,5 carry signal there; MuSiQue (pool 20) keeps signal at k=10.

## Takeaways

1. **Decomposition retrieves the gold substantially better, at matched budget.**
   B-matched beats A by **+5.7 to +23.6 recall** and **+12 to +37 full-coverage**
   across all informative k. The intuition "decompose → retrieve more accurately"
   is now empirically grounded.
2. **Full-coverage is where it matters most** — multi-hop answering needs *every*
   hop's passage. MuSiQue@5: A gets all gold for only **27%** of questions, B for
   **64.5%**. The original question systematically misses the later-hop passage.
3. **Effect scales with hop-bridging difficulty:** MuSiQue (deep multi-hop) ≫
   2Wiki ≫ HotpotQA. HotpotQA's second hop often shares surface entities with the
   question, so a single query already retrieves it — exactly where decomposition
   helps least.
4. **This is a lower bound.** B uses SEER's *resolved* sub-qs, which substitute the
   model's intermediate answers; when an early hop is wrong the later sub-q is
   contaminated — yet B still wins. A gold-decomposition (oracle) arm would only
   widen the gap.

**Caveat / extension.** Measured in the per-question distractor pool, not the
open-domain 21M-passage corpus (DeepRAG setting). Open-domain recall is a follow-up
(needs ES up; see ES-heap-fix). Reproduce: `bash scripts/run_m3_recall.sh`.
