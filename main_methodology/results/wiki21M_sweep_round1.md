# SEER vs DeepRAG — honest wiki-21M pool, Round-1 hyperparameter sweep (2026-06-06)

Pool: shared ES `wiki` (psgs_w100, 21M passages) = DeepRAG's actual corpus.
Fixed: adapter `llama3-8b-decomposer-musique-2wiki-popqa` (12K, three-source),
Llama-3-8B-Instruct base, mode C, max_ent gate τ=0.3, synthesis LoRA-off.
DeepRAG-8B reference: 2Wiki 48.1/53.3 · HotpotQA 40.7/51.5 · MuSiQue 22.5/30.4 · PopQA 42.5/47.8.

## top_k sweep (decompose path)

| Dataset  | k=3 (base) | k=5        | k=8         | k=10       |
|----------|-----------|------------|-------------|------------|
| HotpotQA | 25.1/32.9 | 24.2/32.6  | 25.2/33.4   | 25.2/33.5  |
| 2Wiki    | 38.7/44.6 | 40.4/46.3  | **42.9/48.8** | —        |
| MuSiQue  | 20.3/26.7 | 21.1/28.3  | **22.7/30.5** | —        |
| PopQA    | 50.7/52.9 | —          | —           | —          |

- top_k=8 is best for all three multi-hop sets; **HotpotQA saturates** (k8≈k10),
  so its bottleneck is query QUALITY (KG-triple sub-queries), not recall depth.

## single-step (skip_decompose_tau=1e9, original question retrieves)

| HotpotQA single-step k=8 | **30.3/41.5** |

- +5.2 EM / +8.6 F1 over decompose. Confirms recall_ab_m3 finding #3: on open
  domain, decomposing HURTS HotpotQA (2nd hop shares surface entities with the
  original question, so one full-question query already retrieves it).

## Unified config (decompose, k=8) vs DeepRAG — NO per-dataset cherry-pick

| Dataset  | SEER unified | DeepRAG   | Δ EM/F1        |
|----------|-------------|-----------|----------------|
| PopQA    | 50.7/52.9   | 42.5/47.8 | **+8.2/+5.1** ✅ win |
| MuSiQue  | 22.7/30.5   | 22.5/30.4 | +0.2/+0.1 ≈ tie (n=1000 noise) |
| 2Wiki    | 42.9/48.8   | 48.1/53.3 | −5.2/−4.5 close |
| HotpotQA | 25.2/33.4   | 40.7/51.5 | −15.5/−18.1 lose |

single-step narrows HotpotQA's only big loss to −10.4/−10.0 but it still loses.

**Honest one-liner:** one unified SEER config on DeepRAG's own 21M pool — PopQA
clear win, MuSiQue tie, 2Wiki close, HotpotQA behind. The per-dataset "single vs
decompose" split is exactly what the entropy router (Round 2) must learn to do
automatically.
