# SEER vs DeepRAG — same-pool (open-domain wiki) comparison

**Why this file.** Our headline table (`RESULTS_SEER.md` §6, the `synthOFF` numbers)
is measured on the **per-question distractor pool** (`--corpus paragraphs`, 10–20
candidate paragraphs that always contain the gold — the Adaptive-RAG/IRCoT
protocol). DeepRAG's reported numbers are on the **open-domain wiki corpus**
(psgs_w100, 21M passages, BM25 top-3). Putting them in one table and claiming
"we beat DeepRAG" is apples-to-oranges. This file gives the **same-pool** head-to-head.

Our same-pool runs: `results/{ds}_1000_wiki_top3.jsonl` — canonical 12K 3-source
decomposer, Mode C, max_ent τ=0.3, `--corpus wiki` top-3, post-synthOFF
(created 2026-06-03, 6h after the synthOFF fix; same pipeline). dev first 1000,
no shuffle (matches DeepRAG's `--sample 1000` no-shuffle protocol).

## Open-domain wiki pool (the fair, same-pool comparison)

**Best unified config = k=8** (`*_wiki_k8.jsonl`; PopQA single-hop uses top-3
`*_wiki_top3.jsonl`). F1 uses answer_aliases for MuSiQue/PopQA (matches stored EM).

| Dataset | Ours EM/F1 (k=8) | DeepRAG paper EM/F1 | ΔEM | ΔF1 | verdict |
|---|---|---|---|---|---|
| PopQA | 50.7 / 52.9 | 42.5 / 47.8 | **+8.2** | **+5.1** | win |
| MuSiQue | 22.7 / 30.5 | 22.5 / 30.4 | **+0.2** | **+0.1** | ≈ tie |
| 2WikiMultiHopQA | 42.9 / 48.8 | 48.1 / 53.3 | −5.2 | −4.5 | close |
| HotpotQA | 25.2 / 33.4 | 40.7 / 51.5 | **−15.6** | −18.1 | lose |

**On DeepRAG's actual pool, with the best unified config: win PopQA, tie MuSiQue,
lose 2Wiki (close) and HotpotQA (badly).** Not a sweep, but not "lose all multi-hop"
either — MuSiQue is a micro-win on both EM and F1.

### Strict top-3 match (k=3, `*_wiki_top3.jsonl`) — exactly DeepRAG's retrieve_topk

| Dataset | Ours EM/F1 (k=3) | ΔEM vs DeepRAG |
|---|---|---|
| 2WikiMultiHopQA | 38.7 / 44.6 | −9.4 |
| HotpotQA | 25.1 / 32.9 | −15.6 |
| MuSiQue | 20.3 / 30.5 | −2.2 |
| PopQA | 50.7 / 52.9 | +8.2 |

k=8 is our tuned operating point; DeepRAG retrieves top-3 *per step* but iterates,
so neither k is a single "fair" match. We report k=8 as headline, k=3 as strict floor.

## For contrast — distractor pool (our headline, NOT DeepRAG's pool)

| Dataset | Ours `synthOFF` EM/F1 | pool-change EM gain |
|---|---|---|
| 2WikiMultiHopQA | 62.1 / 69.9 | +23.4 vs wiki |
| HotpotQA | 41.3 / 52.9 | +16.2 vs wiki |
| MuSiQue | 44.3 / 53.7 | +24.0 vs wiki |
| PopQA | 48.7 / 51.2 | −2.0 vs wiki |

The distractor pool inflates the multi-hop numbers by ~+16 to +24 EM, because the
gold passage is guaranteed to be in the 10–20 candidates. PopQA is ~flat (single-hop;
open wiki has comparable single-hop coverage).

## Caveats / to verify

- DeepRAG numbers are **their paper's**, not our re-run on an identical harness —
  small EM/F1-normalization differences possible.
- DeepRAG **PopQA 42.5/47.8** provenance = `RESULTS_SEER.md` §6; verify against the
  DeepRAG paper (PopQA may not be in their main table).
- DeepRAG used full DPO + SC-trained gate; SEER uses LoRA-SFT decomposer + a
  training-free entropy gate (~1/4 the training).
- Our open-domain retrieval is plain BM25 top-3 — the multi-hop gap is partly a
  **retrieval-quality** gap, not only an answering gap (cf. M3: decomposition helps
  recall, but M3 was measured in the distractor pool; an open-domain M3 would show
  whether decomposition closes this gap).

**Decision needed for the thesis:** (a) make open-domain the main table (honest,
but we lose multi-hop) and explain; (b) close the open-domain gap first; or
(c) keep distractor pool as main but separate the tables by protocol and stop
claiming a DeepRAG win on multi-hop.
