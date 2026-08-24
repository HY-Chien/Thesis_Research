# Thesis figure-source manifest

The current thesis references 13 distinct figure files. This directory records
the command or editable source for each one. Run plotting commands from
`Research/main_methodology/` unless stated otherwise.

| Thesis figure | Source | Inputs | Rebuild status |
|---|---|---|---|
| `Thesis_Overview.pdf` | No editable original or generator was found in the workspace | Existing raster-only PDF | Known source gap; preserve the canonical PDF |
| `prompt_Decomposer.pdf` | `figure_sources/prompt_Decomposer.drawio` | Editable XML recovered from the PDF Subject metadata | Open/export with diagrams.net |
| `prompt_Answer.pdf` | `figure_sources/prompt_Answer.drawio` | Editable XML recovered from the PDF Subject metadata | Open/export with diagrams.net |
| `fig_renyi_reweight_auroc.pdf` | `Thesis/make_ch3_signal_figs.py` | Script-documented illustrative/AUROC constants | Rebuildable |
| `fig_same_entropy_risk.pdf` | `Thesis/make_ch3_signal_figs.py` | Script-documented illustrative constants | Rebuildable |
| `fig_leakage_monotonicity.pdf` | `Thesis/make_ch3_signal_figs.py` | Script-documented illustrative constants | Rebuildable |
| `fig_tail_leakage_error_bins.pdf` | `Thesis/make_ch3_signal_figs.py` | Script-documented error-bin values | Rebuildable |
| `fig_answer_hit_retrieval.png` | `scripts/recompute_answer_containing_recall_4ds.py`, then `scripts/make_answer_hit_fig.py` | Four raw dev sets and external ES `wiki` index | Rebuildable with external assets |
| `fig_3way_ablation.png` | `scripts/analyze_tlsum_fixA_step_gate_final.py`, then `scripts/make_3way_fig.py` | Bundled canonical traces | Rebuildable after clone |
| `fig_threshold_tradeoff.png` | Headline analyzer, then `scripts/make_threshold_tradeoff_fig.py` | Bundled canonical traces | Rebuildable after clone |
| `fig_direct_decompose_2x2_counts.png` | `scripts/make_direct_decompose_2x2_fig.py` | Bundled accuracy-first fixed-chain replay | Rebuildable after clone |
| `fig_routing_gap_oracle.png` | `scripts/make_routing_gap_fig.py` | Bundled accuracy-first fixed-chain replay | Rebuildable after clone |
| `fig_quadrant_confidence.png` | `scripts/analyze_quadrant_confidence.py` | Bundled accuracy-first replay and direct Tail Leakage traces | Rebuildable after clone |

## Exact rebuild commands

The four Chapter 3 figures are written directly into `Thesis/figures/`:

```bash
cd Research/Thesis
python make_ch3_signal_figs.py
```

The clone-supported Chapter 5 and appendix figures can be regenerated into
the thesis directory with:

```bash
cd Research/main_methodology
python scripts/analyze_tlsum_fixA_step_gate_final.py
python scripts/make_3way_fig.py \
  --out ../Thesis/figures/fig_3way_ablation.png
python scripts/make_threshold_tradeoff_fig.py \
  --out ../Thesis/figures/fig_threshold_tradeoff.png
python scripts/make_direct_decompose_2x2_fig.py \
  --out ../Thesis/figures/fig_direct_decompose_2x2_counts.png
python scripts/make_routing_gap_fig.py \
  --out ../Thesis/figures/fig_routing_gap_oracle.png
python scripts/analyze_quadrant_confidence.py \
  --out ../Thesis/figures/fig_quadrant_confidence.png
```

The answer-containing figure additionally requires the four external raw
evaluation sets and a running Elasticsearch `wiki` index:

```bash
cd Research/main_methodology
python scripts/recompute_answer_containing_recall_4ds.py
python scripts/make_answer_hit_fig.py \
  --out ../Thesis/figures/fig_answer_hit_retrieval.png
```

Plotting scripts default to `results/reproduced/` so a reviewer does not
overwrite the tracked thesis figures accidentally. Passing `--out` as above
is the explicit update path.

## Editable-source caveat

The two prompt PDFs were exported by diagrams.net with their `mxfile` XML
embedded in PDF metadata. The standalone `.drawio` files here are decoded
copies of that XML. `Thesis_Overview.pdf` contains only a raster image and no
recoverable drawing metadata; the original editable source must be obtained
from the author if it still exists.
