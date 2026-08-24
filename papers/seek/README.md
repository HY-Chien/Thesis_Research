# SEEK paper workspace

> **Status (2026-08-24): anonymous ICLR 2027 draft. This manuscript has not
> been submitted and has not been published.** The `iclr2027/` directory is a
> working submission candidate, not an archival publication.

This directory keeps the conference manuscript, shared evidence, and a future
journal extension separate from the thesis.

## Layout

- **iclr2027/**: anonymous ICLR 2027 draft using the official style files;
  includes the LaTeX source and the current compiled `main.pdf`.
- **shared/**: macros, bibliography, and an asset manifest. Local candidate
  figures are deliberately excluded from the GitHub artifact because the
  manuscript does not use them.
- **journal/**: content plan for a later expanded journal submission.
- **SUBMISSION_CHECKLIST.md**: pre-submission policy, evidence, and
  reproducibility checks; unchecked items are unresolved.

The conference paper intentionally focuses on Tail Leakage. The journal plan
retains the broader SEEK system, full routing analysis, and additional
experiments. The two versions should not be under archival review at the same
time.

## Clean build

From the repository root, copy the project to a clean temporary directory and
build it there:

    tmpdir="$(mktemp -d)"
    cp -a papers/seek/. "$tmpdir/seek-paper/"
    cd "$tmpdir/seek-paper/iclr2027"
    latexmk -C
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

The compiled draft is **iclr2027/main.pdf**. It uses US Letter paper, anonymous
authors, and the official ICLR 2027 style. The checked build has 13 pages: the
current main text ends on page 7, references occupy the rest of pages 7--8, and
the appendix starts on page 9. No file under `shared/figures/` is referenced by
the manuscript.
