# Asset manifest

The current anonymous ICLR 2027 draft uses LaTeX-native tables and a native
pipeline box. It does not reference any external figure file. Accordingly,
`shared/figures/` is intentionally excluded from the GitHub artifact; local
files once collected there are unreviewed candidates, not manuscript assets.

## Local candidates that must not be reused without regeneration

- **fig_threshold_tradeoff.png**: the local candidate predates the corrected
  accuracy-first generator; regenerate with the current thesis script before use.
- **fig_routing_gap_oracle.png** and **fig_quadrant_confidence.png**: local
  candidates predate the current row-level accuracy-first replay; regenerate
  rather than copying these files.
- **fig_same_entropy_risk.pdf**, **fig_leakage_monotonicity.pdf**,
  **fig_renyi_reweight_auroc.pdf**, and **fig_tail_leakage_error_bins.pdf**:
  old files contain Type 3 fonts; regenerate with Matplotlib PDF font type 42.
- The old thesis three-way candidate was omitted after a metric/caption
  mismatch. The current generator now plots final-system EM, but the paper
  draft still has no external figure dependency.

These names document local provenance issues only; their files are not required
to build the current manuscript and are not part of the stage list. Every
future figure promoted into the paper must record its canonical data file,
generator, threshold configuration, and creation command here.
