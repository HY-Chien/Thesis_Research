# DeepRAG vs SEEK Decomposition: Case-Level Comparison

Date: 2026-07-08

Question: when DeepRAG is wrong and SEEK is right on the *same* question, what did DeepRAG's decomposition do wrong? Are there concrete mis-decomposition / over-decomposition examples?

## Setup

- DeepRAG traces: `deeprag_compare/results/learned/{ds}_deeprag-8b_learned.jsonl` (per-step `history` with `follow_up`, `answer`, `retrieved_with`).
- SEEK traces: `main_methodology/results/{ds}_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl` (final candidate, tau=0.3 type-aware replay; `replay_em` used for SEEK correctness).
- Joined by normalized question text; overlap is the full 1000 questions per dataset (998 on MuSiQue).
- Analysis script: scratchpad `join_deeprag_seek.py` / `analyze_wins.py` (session 2026-07-08).

## Quadrant counts (DeepRAG EM × SEEK replay EM)

| Dataset | Both right | DeepRAG only | SEEK only | Both wrong |
|---|---:|---:|---:|---:|
| HotpotQA | 257 | 139 | **98** | 506 |
| 2Wiki | 340 | 162 | **160** | 338 |
| MuSiQue | 154 | 83 | **92** | 669 |

The "SEEK only" column (DeepRAG wrong, SEEK right) is analyzed below. Note the DeepRAG-only column is comparable in size — these are *complementary* failure modes, not one-sided dominance.

## DeepRAG failure taxonomy within the SEEK-win set

Counts are non-exclusive (a case can hit several categories).

| Signal | HotpotQA (n=98) | 2Wiki (n=160) | MuSiQue (n=92) |
|---|---:|---:|---:|
| Format near-miss (DeepRAG F1 ≥ 0.5; answer essentially right) | 21 | 17 | 14 |
| Degenerate final answer (`yes`/`no`/`[No related Info]`) | 24 | 29 | 27 |
| Over-decomposition (≥5 steps) | 5 | 39 | 5 |
| Decomposition spiral: leaked CoT contaminates next follow-up | 31 | 11 | 24 |
| Zero retrievals (all-parametric, hallucinated sub-answers) | 13 | 55 | 0 |
| DeepRAG steps > SEEK steps + 1 (both decomposed) | 7 | 18 | 10 |

Interpretation:

1. **Over-decomposition is a real, countable DeepRAG failure mode on 2Wiki** (39/160 SEEK-wins have ≥5 steps; DeepRAG step histogram in this set: 2 steps ×91, 3 ×24, 5 ×26, 6 ×13). The dominant pattern is comparison questions where DeepRAG appends *redundant terminal steps* ("Which director was born later, X or Y?", then re-asks the original question) answered **parametrically** — and the parametric comparison flips an otherwise-correct chain. SEEK's decomposer emits exactly the 4 factual lookups and leaves the comparison to the final answerer.
2. **Decomposition spiral**: once an intermediate answer is wrong, DeepRAG's next `follow_up` is generated conditioned on the polluted chain — the question field literally contains fragments of the previous erroneous CoT ("However, the context provided…\nFollow up: …"). 24–34% of SEEK-win cases show this contamination. SEEK's decomposition is generated *up front* from the original question only, so a bad intermediate answer cannot corrupt the sub-question structure.
3. **Parametric over-confidence (2Wiki)**: 55/160 SEEK-win cases had *zero* retrievals — DeepRAG's retrieve-or-answer decision chose parametric for every step and hallucinated (e.g., wrong director for an obscure film). SEEK retrieves for every sub-question by design.
4. **Honest caveats**: ~15–21% of these "DeepRAG losses" are EM formatting artifacts (`Murray Hill, New Jersey` vs `Murray Hill`, `Norse mythology` vs `Norse`) — exclude them when citing counts. And SEEK's own decomposition is sometimes bad in the win set too; it is saved by the direct path or by original-question evidence union (examples below).

## Case studies

### A. Over-decomposition with parametric flip (2Wiki comparison questions)

**Q: Which film has the director who was born earlier, Buffet Froid or Lost In Africa?** (gold: Buffet Froid)

DeepRAG, 6 steps, only 1 retrieval:
1. [retrieval] director of Buffet Froid → Bertrand Blier ✓
2. [parametric] birth date of Blier → 20 March 1939 ✓
3. [parametric] director of Lost In Africa → **Claude Lelouch** (hallucination; actual: Jamie Uys)
4. [parametric] birth date of Lelouch → 30 October 1937
5. [parametric] who born earlier → Claude Lelouch (consistent with its own wrong facts)
6. [parametric] re-ask original → **Lost In Africa** ✗

SEEK, fixed 4-step template, all retrieved: Blier / Jamie Uys / 14 Mar 1939 / 30 May 1921 → answerer outputs **Buffet Froid** ✓. Same pattern in *My Own Private Idaho vs Lucignolo*, *Garde À Vue vs Ivide Thudangunnu*, *The Last Company vs Anand Ashram* (there DeepRAG's step-1 retrieval itself grabbed "Francis Ford Coppola" as director of the 1930 German film The Last Company). Steps 5–6 are pure over-decomposition: they add no information and are where the flip happens.

### B. Redundant reverse-lookup steps + format loss

**Q: In which city are the headquarters of the American research company where Ravi Sethi worked?** (gold: Murray Hill)

DeepRAG, 6 steps: gets Bell Labs *and* Avaya Labs at step 1, then explores both HQ cities, then adds two **reverse lookups** ("Which company's HQ is in Murray Hill?" / "…in Santa Clara?") that answer nothing new, finally disambiguates at step 6. Prediction "Murray Hill, New Jersey" → EM 0 (F1 0.5). SEEK: 2 steps (`Ravi Sethi >> employer` → Bell Labs; `>> headquarters location`) → "Murray Hill" ✓. Over-decomposition here doesn't flip the answer, but the longer chain produces an over-specified answer string.

Same shape: **Michael Pollock / Royal Navy** — DeepRAG has the answer at step 1, then asks "informal name of the Royal Navy" three more times and degrades to "the Navy" (EM 0).

### C. Decomposition spiral after a wrong first hop

**Q: Who was known by his stage name Aladin and helped organizations improve their performance as a consultant?** (gold: Eenasul Fateh)

DeepRAG step 1 resolves Aladin → "Cole Gribble" (wrong entity, an Indonesian soap-opera child actor), then spends 5 more steps trying to make Cole Gribble a consultant; follow-ups 3–6 are visibly contaminated with prior CoT text and it ends at "no". SEEK: `Aladin >> real name` → Eenasul Fateh → `>> employer` ✓. The spiral is the interaction of (a) iterative decomposition conditioned on the chain and (b) no mechanism to back out of a wrong entity.

Similar: **Small Cowper Madonna** (MuSiQue) — DeepRAG's chain drifts to the Uffizi, then asks "what *state* is the Uffizi in?" → "Florida" → final "Italy". SEEK retrieves the correct museum (National Gallery of Art) → Washington, D.C. ✓.

### D. Structurally wrong decomposition (lost bridge relation)

**Q: Who sings Home Alone Tonight *with* the singer of You Can Crash My Party Anytime?** (gold: Karen Fairchild)

DeepRAG decomposes into two *independent* lookups — "who sings You Can Crash My Party Anytime" (Luke Bryan) and "who sings Home Alone Tonight" (Luke Bryan) — dropping the "with" (duet-partner) relation, and answers Luke Bryan. SEEK's step 2 keeps the placeholder binding: "who sings home alone tonight **with #1**" → Karen Fairchild ✓. This is a genuine mis-decomposition, not a retrieval error. Same failure shape in *Turn Me On / Chasing Pirates* (DeepRAG answers the intersection artist instead of the writer).

### E. Fiction/reality confusion in the sub-answer chain

**Q: Who was the spouse of the leading lady in Gone With the Wind?** (gold: Laurence Olivier)

DeepRAG: leading lady → "Scarlett O'Hara" (the character) → spouse → "Rhett Butler". SEEK resolves the same sub-question to Vivien Leigh (actress) → Laurence Olivier ✓. Decomposition structure identical — the difference is the per-step answerer + retrieval grounding.

### F. SEEK wins even when its own decomposition is bad (dual-path / union rescue)

Honest examples to keep the story symmetric:

- **Shiners' War** (HotpotQA, gold: French Canadians): SEEK's decomposition also fails (step 3 → "United Empire Loyalist") but the **direct path** answers correctly and the verifier picks it. DeepRAG has no direct fallback and ends at "Irish Catholic immigrants". 29/98 HotpotQA SEEK-wins were won on the direct path — decomposition was unnecessary for these, and DeepRAG's forced iterative chain actively hurt.
- **Kansas City Wizards** (gold: Arrowhead Stadium): SEEK's decomposition is poor (`Kansas City Wizards >> named after` → "Wizards") yet the union-augmented retrieval still surfaces the right evidence and step 2 answers Arrowhead Stadium. DeepRAG's clean-looking 2-step chain picks the wrong venue from retrieval.

## Takeaways for the thesis narrative

1. **Concrete over-decomposition evidence exists and is quantifiable**: 2Wiki comparison questions, ≥5-step chains in 39/160 SEEK-wins, with the error introduced precisely in the redundant terminal comparison steps (answered parametrically). Good slide material: case A.
2. **Iterative decomposition has a failure mode SEEK structurally avoids**: chain-conditioned follow-up generation lets one wrong hop corrupt all later sub-questions (spiral + CoT leakage into the question field). SEEK's up-front decomposition + per-step retrieval bounds the damage.
3. **DeepRAG's retrieve-or-not gate is a liability on 2Wiki** (55/160 zero-retrieval hallucination losses) — supports the SEEK choice of always retrieving per sub-question.
4. **Symmetry warning**: DeepRAG-only wins are of similar magnitude (139/162/83). Use these cases to argue the failure modes *differ* and that SEEK's dual-path + union mitigates decomposition risk — not that SEEK dominates per-question.
5. When citing counts, subtract format near-misses (~15–21% of the win set).
