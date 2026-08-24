# Journal extension plan

## Proposed scope

Working title: **SEEK: Interpretable Dual-Path Adaptive Retrieval with
Confidence-Guided Decomposition**

The journal article should present the complete SEEK system. It should not be a
longer copy of the ICLR manuscript; it needs substantial new evidence and a
clearly broader contribution.

## Content split

### ICLR conference paper

- Tail Leakage definition and entropy decomposition.
- Representative confidence comparison.
- Step-level adaptive retrieval in SEEK as a case study.
- Concise end-to-end QA, retrieval, and routing results.
- Main limitations and confidently wrong boundary.

### Journal extension

- Full decomposer construction and training analysis.
- Controlled factorial ablations for decomposition, title matching,
  original-question evidence union, and synthesis.
- Full-dev and hop-stratified evaluation.
- Multiple model families and model sizes.
- Multiple seeds or uncertainty intervals.
- Dense/hybrid retrieval and reranking comparisons.
- Complete direct/decomposition overlap, oracle routing, verifier variants,
  and evidence-aware verification.
- Retrieval calls, latency, memory, and offline adaptation cost.
- Extended failure taxonomy and qualitative cases.
- Calibration, top-k sensitivity, and theoretical analysis of Tail Leakage.

## Suggested journal structure

1. Introduction
2. Related work
3. Tail Leakage
4. The SEEK framework
5. Experimental protocol
6. Confidence and calibration
7. Retrieval and decomposition ablations
8. Routing and evidence-aware verification
9. Efficiency and scalability
10. Failure analysis
11. Limitations and conclusion

## Submission order

Do not submit a substantially similar journal manuscript while the ICLR paper
is under review. Prepare experiments and a private draft in parallel, then
select a journal and adapt its official template only after the conference
submission is no longer under archival review or after the journal version is
clearly established as a substantial extension under the venue's policy.
