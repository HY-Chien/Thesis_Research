# ICLR 2027 submission checklist

> This is a prospective checklist for an anonymous draft. As of 2026-08-24,
> the manuscript has not been submitted and has not been published.

## Policy and format

- [x] Use the official ICLR 2027 LaTeX style.
- [x] Keep the review PDF anonymous and leave the final-copy command commented.
- [x] Use US Letter and embedded fonts.
- [x] Keep initial main text within 9 pages (current endpoint: page 7).
- [x] Put references before the appendix.
- [x] Include the required AI use statement.
- [ ] Confirm the AI statement with every coauthor before submission.
- [ ] Create or update every author's OpenReview profile before the abstract
  deadline; no author may be added after that deadline.
- [ ] Anonymize supplementary code, paths, metadata, repository history, and
  links.
- [ ] Confirm no substantially similar conference or journal paper is under
  review during the ICLR review period.

## Evidence required before treating this as a final submission

- [ ] Calibrate the step-gate and verifier thresholds on a disjoint calibration
  split, then report results on untouched evaluation examples. The current
  operating point was chosen from the reported development traces.
- [ ] Run at least one additional backbone. The pending Qwen-2.5-7B experiment
  is the highest-priority robustness test.
- [ ] Evaluate the full development sets or provide stratified results,
  especially all 2--4 hop MuSiQue questions. The current first-1,000 slice is
  entirely two-hop.
- [ ] Run a controlled retrieval comparison with identical title fields for
  original and decomposed queries. The current Hit@5 table measures the full
  path policy and cannot isolate decomposition wording.
- [ ] Add a separate original-question evidence-union ablation.
- [ ] Reproduce key baselines in the same harness, or retain the explicit
  “reported results” qualification and avoid SOTA claims.
- [ ] Recompute entropy-component values from complete logits if they remain a
  mechanism claim.
- [ ] Add uncertainty estimates or repeated runs for small differences.

## Artifact consistency

- [ ] Repair old figure scripts that still point to moved result files.
- [ ] Ensure every script highlights the final step-gate threshold 0.002305;
  0.008461 should appear only as a threshold-sweep point.
- [ ] Reconcile routing/oracle source reports with the final all-Tail-Leakage
  counts.
- [ ] Do not reuse the old three-way figure until its decomposition-path metric
  and caption are made consistent.
- [ ] Regenerate any retained Matplotlib PDFs with Type 42 fonts.
- [ ] Export an anonymous minimal code-and-trace bundle and run it from a clean
  environment.

## Final editorial pass

- [ ] Verify every reported number against a canonical JSON/report.
- [ ] Verify all claims use “development set” where appropriate.
- [ ] Keep final routing framed as a small conservative fallback (+0.28 macro
  EM), not the main source of the gain.
- [ ] Keep efficiency claims modest: the accuracy-first gate retrieves 93.84%
  of decomposed steps.
- [ ] Ask a fluent English reader and a technical coauthor to review the final
  PDF.
