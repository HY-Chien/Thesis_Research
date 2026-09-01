# Thesis Reproduction Guide

This document maps every SEEK experiment, table, and figure in the current
thesis to its executable entry point. Baselines quoted from earlier papers are
outside this repository and are intentionally excluded.

Run commands from `Research/main_methodology/` unless the command explicitly
changes directory.

## Availability Labels

| Status | Meaning |
|---|---|
| **Runs from repository** | All required input files are already in Git, the command runs on CPU, and its output was checked against the thesis |
| **Extra files required** | The command works, but it also needs model weights, original datasets, or the Wikipedia search index that are not stored in Git |
| **Historical result only** | The thesis result is recorded, but some original per-question outputs or the final aggregation script are unavailable |
| **Figure source missing** | The rendered figure exists, but its editable source or generation script was not found |

Generated analysis outputs go to `results/reproduced/`. GPU/live outputs go
to `results/generated/`. Both directories are ignored by Git.

## 1. Thesis Results and Entry Points

| Thesis content | Entry point | Inputs | Output | Hardware and availability |
|---|---|---|---|---|
| Ch. 4 SEEK main-result row; Ch. 5 retrieval-necessity and threshold-sensitivity tables | `analyze_tlsum_fixA_step_gate_final.py` | Saved predictions and confidence scores for four datasets in `results/traces/` | One Markdown report and one JSON file in `results/reproduced/` | CPU; **runs from repository** |
| Ch. 5 skipped-retrieval safety | `analyze_skipped_retrieval_questions.py` | The same saved step answers and final predictions | Markdown and JSON reports plus a JSONL file of affected questions | CPU; **runs from repository** |
| Appendix decomposition failure taxonomy and diagnostic flags | `analyze_accuracy_first_failure_taxonomy.py` | Saved main-result files plus six retrieval-coverage files | `decomp_failure_taxonomy_tau0p002305.{md,json}` | CPU; **runs from repository** |
| Ch. 5 confidence-signal AUROC battery | `analyze_confidence_battery.py` | Four saved prediction files containing confidence scores | `confidence_battery_analysis.md` | CPU; **runs from repository** |
| Ch. 5 entropy components and selective accuracy; Appendix confidently-wrong categories | `analyze_tlsum_confidence_story.py` | Four saved prediction files containing confidence scores | `tlsum_confidence_story.md` | CPU; **runs from repository** |
| Supporting Tail Leakage variants and paired bootstrap | `analyze_tail_leakage_variants.py` | Four saved prediction files containing confidence scores | `tail_leakage_variants.md` | CPU; **runs from repository** |
| Ch. 5 verifier choices, path overlap, routing effect, oracle headroom; Appendix quadrant confidence | Accuracy-first replay helper plus three analysis/plot scripts | Saved step answers, direct answers, decomposed answers, confidence scores, and final predictions | Three Markdown reports and three PNG figures | CPU; **runs from repository** |
| Ch. 5 answer-containing Hit@5 | `recompute_answer_containing_recall_4ds.py` | Four original development sets, saved SEEK predictions, and the Elasticsearch `wiki` index | `answer_containing_recall_4ds_open_domain.md` | CPU + Elasticsearch; **extra files required** |
| Ch. 5 MuSiQue gold/base decomposition recall | `decompose_base_musique.py`, then `recall_gold_decomp_musique.py` | Original MuSiQue data, base model, saved SEEK predictions, and the Elasticsearch `wiki` index | Base-model decompositions, per-question details, and a summary table | GPU + Elasticsearch; **extra files required** |
| Ch. 5 title-off/title-on/path-aware comparison | `run_tlsum_title_ablation_tau002305.sh` for all three settings | Base model, LoRA adapter, saved SEEK input files, and the Elasticsearch `wiki` index | Prediction JSONL files and logs under `results/generated/title_policy/` | GPU + Elasticsearch; **extra files required**; thesis aggregation is historical |
| Ch. 5 maximum entropy vs. Tail Leakage | Tail Leakage from the saved main-result analysis; maximum-entropy producer is `run_seek_dual.py --signal max_ent` | Base model, adapter, original development data, Elasticsearch, and the historical threshold | New prediction JSONL | Original per-question maximum-entropy outputs are unavailable; **historical result only** |
| Appendix repeated-sampling errors | `probe_multisample.py` or `run_m2b_multisample.sh` | Base model, adapter, and evaluation JSONL files containing source paragraphs | One repeated-sampling JSONL file per dataset | GPU; final historical aggregation is **historical result only** |
| Ch. 4 decomposer training | LLaMA-Factory config in `configs/` | Public [SEEK 12K dataset](https://huggingface.co/datasets/Chien1011/SEEK-decomposer-12k) and gated base model | Trained LoRA adapter | GPU; **extra files required** |
| Ch. 4 live final SEEK system | `run_seek_dual.py` | Base model, trained adapter, evaluation JSONL, and optionally Elasticsearch | Prediction JSONL path selected by `--out` | GPU; **extra files required** |

## 2. Offline Reproduction Using Included Result Files

Run every CPU analysis whose input files are already in this repository with:

```bash
bash scripts/run_offline_reproduction.sh
```

The complete script was rerun successfully on 2026-08-30. The commands below
show the individual entry points.

### Main Result, Gate Ablations, Skip Safety, and Taxonomy

```bash
python scripts/analyze_tlsum_fixA_step_gate_final.py
python scripts/analyze_skipped_retrieval_questions.py
python scripts/analyze_accuracy_first_failure_taxonomy.py
```

The accuracy-first row must be:

```text
Tail Leakage tau = 0.002305
HotpotQA          37.80 EM / 49.03 F1
2Wiki             49.30 EM / 56.22 F1
MuSiQue           26.50 EM / 34.63 F1
PopQA             53.90 EM / 55.73 F1
Macro             41.88 EM / 48.90 F1
Pooled retrieval  93.84%
```

The skipped-retrieval analysis must find 478 skipped steps out of 7,761 and
418 affected questions.

### Confidence Analyses

```bash
python scripts/analyze_confidence_battery.py
python scripts/analyze_tail_leakage_variants.py
python scripts/analyze_tlsum_confidence_story.py
```

These were rerun against a clean copy of the saved result files on 2026-08-24. The
representative signal battery, entropy-component AUROCs, selective-accuracy
table, and all 491 confidently-wrong cases match the current thesis.

### Path Overlap, Routing, Oracle, and Confidence Quadrants

```bash
python scripts/make_direct_decompose_2x2_fig.py
python scripts/make_routing_gap_fig.py
python scripts/analyze_quadrant_confidence.py
```

The shared `scripts/accuracy_first_replay.py` reconstructs candidates at
`tau=0.002305` from per-question result files; it does not embed the thesis counts.
Expected correctness quadrants are:

| Dataset | Both correct | Direct-only | Decomp-only | Both wrong |
|---|---:|---:|---:|---:|
| HotpotQA | 225 | 78 | 135 | 562 |
| 2Wiki | 144 | 36 | 353 | 467 |
| MuSiQue | 75 | 27 | 192 | 706 |
| PopQA | 346 | 32 | 194 | 428 |

Expected direct-choice counts are 171, 25, 121, and 10 respectively. The
quadrant-confidence report should give pooled AUROC 0.817 and 9.3% confidently
wrong after rounding.

## 3. GPU and External-Data Analyses

### Four-Dataset Answer-Containing Hit@5

Provide the following raw files:

```text
datasets/2wiki/raw/dev_first1000.jsonl
datasets/hotpotqa/raw/dev_first1000.jsonl
datasets/musique/raw/dev_first1000.jsonl
datasets/popqa/raw/dev.jsonl
```

Start or connect to Elasticsearch 7.10.2 with the `wiki` index, then run:

```bash
python scripts/recompute_answer_containing_recall_4ds.py \
  --ks 1 3 5 8 10 \
  --corpus wiki
```

The script joins the original gold answers and aliases with the saved SEEK
predictions. It
writes `results/reproduced/answer_containing_recall_4ds_open_domain.md`.

### MuSiQue Gold/Base Decomposition Diagnostic

```bash
python scripts/decompose_base_musique.py \
  --base meta-llama/Meta-Llama-3-8B-Instruct \
  --dev datasets/musique/raw/dev_first1000.jsonl \
  --out results/generated/musique_1000_base_decomp.jsonl

python scripts/recall_gold_decomp_musique.py \
  --dev datasets/musique/raw/dev_first1000.jsonl \
  --trace results/traces/musique/musique_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl \
  --base-decomp results/generated/musique_1000_base_decomp.jsonl \
  --detail results/generated/musique_gold_decomp_recall_detail.jsonl \
  --ks 1 3 5 8 10 \
  --corpus wiki
```

The second command prints the summary table and writes row-level details when
`--detail` is supplied.

### Title Handling

```bash
export BASE_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
export SEEK_ADAPTER=/path/to/seek-lora-adapter

bash scripts/run_tlsum_title_ablation_tau002305.sh off 0
bash scripts/run_tlsum_title_ablation_tau002305.sh global 0
bash scripts/run_tlsum_title_ablation_tau002305.sh pathaware 0
```

The portable launcher derives the project path from its own location and uses
the active `python` unless `PYTHON_BIN` is set. The thesis table predates
the current packaging pass; its final aggregation is retained as historical
historical documentation rather than being labelled as reproducible from a
new clone.

### Maximum-Entropy Live Arm

Use the same setup as the live SEEK evaluation below, changing the step signal
and corresponding verifier thresholds:

```bash
python scripts/run_seek_dual.py \
  --base meta-llama/Meta-Llama-3-8B-Instruct \
  --adapter /path/to/seek-lora-adapter \
  --dev /path/to/dev.jsonl \
  --mode C \
  --signal max_ent \
  --tau 0.3 \
  --corpus wiki \
  --top-k 8 \
  --title-mode pathaware \
  --direct-context-union \
  --no-evidence \
  --direct-entropy-tau 0.4 \
  --decomp-post-entropy-tau 0.8 \
  --typeaware-fallback \
  --fallback auto \
  --out results/generated/maxent_live.jsonl
```

This command produces a new live run. The original per-question outputs used
for the maximum-entropy thesis row are not stored in this repository.

### Repeated Sampling

```bash
export BASE_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
export SEEK_ADAPTER=/path/to/seek-lora-adapter

bash scripts/run_m2b_multisample.sh \
  0 \
  hotpotqa:/path/to/hotpotqa.jsonl \
  2wiki:/path/to/2wiki.jsonl \
  musique:/path/to/musique.jsonl \
  popqa:/path/to/popqa.jsonl
```

The launcher produces 15 samples/question at temperature 0.7. The current
appendix's identical-answer/zero-mean-max-entropy count was historically
aggregated and does not have a dedicated current report writer.

### Decomposer Training

Run from `main_methodology/`:

```bash
SEEK_DATA_DIR=/tmp/SEEK-decomposer-12k
hf download Chien1011/SEEK-decomposer-12k \
  --repo-type dataset \
  --revision 3dcb5c057a82b6610b7b303855e273bb93f9bab7 \
  --local-dir "$SEEK_DATA_DIR"
(cd "$SEEK_DATA_DIR" && sha256sum -c SHA256SUMS)

export LLAMA_FACTORY_DIR="$(cd ../.. && pwd)/LLaMA-Factory"

if [ ! -d "$LLAMA_FACTORY_DIR/.git" ]; then
  git clone https://github.com/hiyouga/LLaMA-Factory.git "$LLAMA_FACTORY_DIR"
fi

git -C "$LLAMA_FACTORY_DIR" checkout 7e20db5735443861fad91766614b3415ff8966a4
python -m pip install -e "$LLAMA_FACTORY_DIR"
python -m pip check

# Validate the setup without training.
bash scripts/train_decomposer.sh --check-only

# Full three-epoch thesis training run.
bash scripts/train_decomposer.sh
```

The wrapper registers the three datasets automatically. Dataset sources and
the original construction record are documented in `DATASETS.md`. The wrapper
uses GPU 0 by default so the effective batch size remains 16, matching the
thesis run.

### Live SEEK

The exact path-aware, original-question-union command is in the root README
and `main_methodology/README.md`. Use `--corpus paragraphs` for a smoke
test without ES, or `--corpus wiki` for the thesis retrieval setting.

## 4. All 13 thesis figures

| Figure | Exact producer/update command | Status |
|---|---|---|
| `Thesis_Overview.pdf` | No command; preserve the tracked PDF | **Figure source missing**: raster-only PDF |
| `prompt_Decomposer.pdf` | Open/export `../Thesis/figure_sources/prompt_Decomposer.drawio` in diagrams.net | Editable source recovered |
| `prompt_Answer.pdf` | Open/export `../Thesis/figure_sources/prompt_Answer.drawio` in diagrams.net | Editable source recovered |
| `fig_renyi_reweight_auroc.pdf` | `cd ../Thesis && python make_ch3_signal_figs.py` | Rebuildable; illustrative constants in script |
| `fig_same_entropy_risk.pdf` | Same Chapter 3 command | Rebuildable |
| `fig_leakage_monotonicity.pdf` | Same Chapter 3 command | Rebuildable |
| `fig_tail_leakage_error_bins.pdf` | Same Chapter 3 command | Rebuildable; source values documented in script |
| `fig_answer_hit_retrieval.png` | Run the four-dataset Hit@5 analysis, then `python scripts/make_answer_hit_fig.py --out ../Thesis/figures/fig_answer_hit_retrieval.png` | **Extra files required** |
| `fig_3way_ablation.png` | Run the main-result analysis, then `python scripts/make_3way_fig.py --out ../Thesis/figures/fig_3way_ablation.png` | **Runs from repository**; plots final-system EM |
| `fig_threshold_tradeoff.png` | Run the main-result analysis, then `python scripts/make_threshold_tradeoff_fig.py --out ../Thesis/figures/fig_threshold_tradeoff.png` | **Runs from repository**; highlights accuracy-first `tau=0.002305` |
| `fig_direct_decompose_2x2_counts.png` | `python scripts/make_direct_decompose_2x2_fig.py --out ../Thesis/figures/fig_direct_decompose_2x2_counts.png` | **Runs from repository** |
| `fig_routing_gap_oracle.png` | `python scripts/make_routing_gap_fig.py --out ../Thesis/figures/fig_routing_gap_oracle.png` | **Runs from repository** |
| `fig_quadrant_confidence.png` | `python scripts/analyze_quadrant_confidence.py --out ../Thesis/figures/fig_quadrant_confidence.png` | **Runs from repository** |

Plot scripts default to `results/reproduced/`, which prevents an ordinary
review run from modifying tracked thesis figures. Passing the explicit
`--out ../Thesis/figures/...` path updates the LaTeX assets.

The complete source mapping and recovered Draw.io files are in
`../Thesis/figure_sources/`.

## 5. LaTeX and paper figures

The thesis build is independent of the experiment environment:

```bash
cd ../Thesis
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error main_diff.tex
```

The ICLR 2027 working draft currently has no external figure dependency. Its
method overview is LaTeX-native in
`papers/seek/iclr2027/sections/method.tex`. Candidate images previously kept
under `papers/seek/shared/figures/` are stale/unreferenced and intentionally
excluded from the files prepared for review.
