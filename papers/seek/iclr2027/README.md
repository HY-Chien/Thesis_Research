# ICLR 2027 draft

> **Anonymous working draft (2026-08-24). Not submitted and not published.**

Working title: **Tail Leakage: A Simple Confidence Signal for Adaptive
Retrieval**

This is a new conference manuscript, not a reformatted copy of the thesis.
Its main claim is the Tail Leakage confidence signal; SEEK is the evaluation
framework.

## Official requirements used here

- Abstract deadline: 2026-09-18 AOE.
- Full-paper and supplementary deadline: 2026-09-25 AOE.
- Initial main text: at most 9 pages.
- References and appendix: outside the main-text limit.
- Anonymous review: keep the final-copy command commented.
- AI use statement: required.

## Clean build

Run from this directory:

    latexmk -C
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

The official style files are stored locally beside **main.tex**. Do not edit the
style file or margins. The source does not depend on `../shared/figures/`.

## Draft status

The checked PDF has 13 US-Letter pages and compiles without missing citations,
missing references, or overfull boxes. It is anonymous and uses embedded Type
1 fonts. It also carries an explicit unsubmitted-draft banner; remove that
banner only if the paper is actually submitted. Content-level blocking checks
remain in **../SUBMISSION_CHECKLIST.md**.
