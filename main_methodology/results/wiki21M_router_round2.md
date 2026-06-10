# SEER Round-2 — single-step confirm + adaptive entropy router (wiki-21M, 2026-06-07)

Pool wiki-21M, 12K decomposer, mode C, gate τ=0.3, top_k=8 (PopQA baseline was k=3).
Router = `--skip-decompose-tau τ`: probe ORIGINAL question; if max_ent < τ skip
decomposition (single-step RAG), else decompose. Higher τ → more single-step.
DeepRAG-8B: 2Wiki 48.1/53.3 · HotpotQA 40.7/51.5 · MuSiQue 22.5/30.4 · PopQA 42.5/47.8.

## (A) single-step confirm — does forced single-step hurt real multi-hop?

| Dataset  | decompose k8 | single-step k8 | Δ EM      |
|----------|-------------|----------------|-----------|
| HotpotQA | 25.2/33.4   | **30.3/41.5**  | **+5.1** (single-step WINS) |
| 2Wiki    | 42.9/48.8   | 18.0/27.6      | **−24.9** (collapse) |
| MuSiQue  | 22.7/30.5   | 10.2/17.6      | **−12.5** (collapse) |

→ Confirmed hard: forced single-step is catastrophic for true multi-hop
(2Wiki/MuSiQue) but WINS on HotpotQA. "Decompose vs single-step" is strongly
dataset-dependent — exactly the decision a router should make per query.

## (B) adaptive router — can ONE global τ make that decision?

EM, skip% in parens. Reference: decompose k8 / single-step k8.

| Dataset  | τ=0.5 (skip)  | τ=0.8 (skip)  | τ=1.2 (skip)  | all-decomp | all-single |
|----------|---------------|---------------|---------------|-----------|-----------|
| HotpotQA | **32.7 (40%)**| 32.6 (58%)    | 32.2 (72%)    | 25.2      | 30.3      |
| 2Wiki    | 39.9 (22%)    | 35.2 (40%)    | 29.3 (57%)    | **42.9**  | 18.0      |
| MuSiQue  | 20.8 (20%)    | 19.6 (32%)    | 17.2 (49%)    | **22.7**  | 10.2      |
| PopQA    | 47.7 (42%)    | 43.1 (61%)    | 40.1 (76%)    | 50.7(k3)  | —         |

### Two findings, one positive one negative

**POSITIVE — HotpotQA: query-level entropy routing genuinely works.**
router 32.7 > all-single 30.3 > all-decompose 25.2. Mixing (decompose the
hard 60%, single-step the easy 40%) beats EITHER pure policy. Entropy carries
real per-query signal here.

**NEGATIVE — 2Wiki/MuSiQue/PopQA: ANY skip hurts; best is τ→0 (never skip).**
Monotone: more skip → worse. These sets have no "single-step-able" subset; their
low-entropy questions are NOT easy, they are **confident-but-wrong** (links to
[[research_b1b2b3_framing]]). So a single global τ cannot win: HotpotQA wants
high skip, the other three want zero skip.

### Best single global policy is still all-decompose k8

τ=0.5 router vs DeepRAG: HotpotQA −8.0 (best yet, beats single-step −10.4),
2Wiki −8.2, MuSiQue −1.7, PopQA +5.2. It trades HotpotQA's big gain for small
losses elsewhere — net DeepRAG win/tie count does NOT improve over all-decompose
(PopQA win, MuSiQue tie, 2Wiki/HotpotQA lose). The router is a strong ANALYSIS
result (why naive routing is insufficient), not a leaderboard win.

## Takeaways for the thesis
1. Decompose-vs-single is dataset-dependent (A) and, within a routable dataset,
   per-query entropy routing helps (HotpotQA B).
2. A single global entropy τ cannot route across datasets — low entropy on true
   multi-hop = confident hallucination, not easiness.
3. Best unified config remains all-decompose k8. To actually narrow the DeepRAG
   gap, the open lever is retrieval QUERY quality (use original question instead
   of KG-triple sub-q), not the gate.
