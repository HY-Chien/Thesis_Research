# Thesis experiment and figure reproduction matrix

This document maps every SEEK experiment, table, and figure in the current
thesis to its executable entry point. Baselines quoted from earlier papers are
outside this repository and are intentionally excluded.

Run commands from `Research/main_methodology/` unless the command explicitly
changes directory.

## Status definitions

| Status | Meaning |
|---|---|
| **Verified clone** | Uses tracked canonical artifacts, runs on CPU, and was checked against the current thesis |
| **External assets** | The command is portable, but licensed model weights, raw data, or the external ES index are intentionally not committed |
| **Provenance only** | The result is retained in the thesis/research ledger, but its complete row-level inputs or final aggregation program are not available in a normal clone |
| **Manual source gap** | The rendered asset exists, but no editable original or generator was found |

Generated reviewer outputs go to `results/reproduced/`. GPU/live outputs go
to `results/generated/`. Both directories are ignored by Git.

## 1. Experiment and table matrix

| ID | Thesis item | Entry point | Inputs | Output | Resources/status |
|---|---|---|---|---|---|
| E1 | Ch. 4 SEEK main-result row; Ch. 5 retrieval-necessity table; threshold-sensitivity table | `analyze_tlsum_fixA_step_gate_final.py` | 18 canonical traces under `results/traces/` | `results/reproduced/tlsum_fixA_step_gate_final.{md,json}` | CPU; **verified clone** |
| E2 | Ch. 5 skipped-retrieval safety | `analyze_skipped_retrieval_questions.py` | Same fixed-chain artifacts as E1 | Markdown, JSON, and case JSONL in `results/reproduced/` | CPU; **verified clone** |
| E3 | Appendix decomposition failure taxonomy and diagnostic flags | `analyze_accuracy_first_failure_taxonomy.py` | E1 artifacts plus six bundled coverage traces | `decomp_failure_taxonomy_tau0p002305.{md,json}` | CPU; **verified clone** |
| E4 | Ch. 5 confidence-signal AUROC battery | `analyze_confidence_battery.py` | Four bundled `conf_battery` traces | `confidence_battery_analysis.md` | CPU; **verified clone** |
| E5 | Ch. 5 entropy components, selective accuracy; Appendix confidently-wrong categories | `analyze_tlsum_confidence_story.py` | Four bundled `conf_battery` traces | `tlsum_confidence_story.md` | CPU; **verified clone** |
| E6 | Supporting Tail Leakage metric variants and paired bootstrap | `analyze_tail_leakage_variants.py` | Four bundled `conf_battery` traces | `tail_leakage_variants.md` | CPU; **verified clone** |
| E7 | Ch. 5 verifier choices, path correctness overlap, routing effect, oracle headroom; Appendix quadrant confidence | Accuracy-first replay helper plus three analysis/plot scripts | Bundled step/final/synthesis/direct-confidence traces | Three Markdown reports and three PNGs | CPU; **verified clone** |
| E8 | Ch. 5 answer-containing Hit@5 | `recompute_answer_containing_recall_4ds.py` | Four raw dev sets, bundled final traces, ES `wiki` | `answer_containing_recall_4ds_open_domain.md` | CPU + ES; **external assets** |
| E9 | Ch. 5 MuSiQue gold/base decomposition recall | `decompose_base_musique.py`, then `recall_gold_decomp_musique.py` | Raw MuSiQue dev, base model, bundled final trace, ES `wiki` | Base decomposition JSONL, detail JSONL, stdout table | GPU + ES; **external assets** |
| E10 | Ch. 5 title-off/title-on/path-aware end-to-end comparison | `run_tlsum_title_ablation_tau002305.sh` for all three arms | Base model, LoRA adapter, bundled source traces, ES `wiki` | Per-arm JSONL/logs under `results/generated/title_policy/` | GPU + ES; **external assets**; thesis table aggregation is historical |
| E11 | Ch. 5 maximum-entropy vs Tail Leakage end-to-end comparison | Tail Leakage row from E1; maximum-entropy live producer is `run_seek_dual.py --signal max_ent` | Base model, adapter, raw dev, ES, historical operating point | Live JSONL | Maximum-entropy row-level historical traces are not bundled; **provenance only** |
| E12 | Appendix repeated-sampling errors | `probe_multisample.py` or portable launcher `run_m2b_multisample.sh` | Base model, adapter, evaluation JSONL with paragraphs | Per-dataset multisample JSONL | GPU; exact appendix zero-entropy aggregation has no dedicated current writer, so final table is **provenance only** |
| E13 | Ch. 4 decomposer training configuration | LLaMA-Factory config in `configs/` | Public [SEEK 12K dataset](https://huggingface.co/datasets/Chien1011/SEEK-decomposer-12k), gated base model | LoRA adapter checkpoint | GPU; **external assets** |
| E14 | Ch. 4 live final system | `run_seek_dual.py` | Base model, adapter, evaluation JSONL, optional ES | Prediction JSONL selected by `--out` | GPU; **external assets** |

## 2. Verified clone commands

Run every verified E1–E7 item with:

```bash
bash scripts/run_offline_reproduction.sh
```

The complete script was rerun successfully on 2026-08-30. The commands below
show the individual entry points.

### E1-E3: headline, gate ablations, skip safety, and taxonomy

```bash
python scripts/analyze_tlsum_fixA_step_gate_final.py
python scripts/analyze_skipped_retrieval_questions.py
python scripts/analyze_accuracy_first_failure_taxonomy.py
```

The E1 accuracy-first row must be:

```text
Tail Leakage tau = 0.002305
HotpotQA          37.80 EM / 49.03 F1
2Wiki             49.30 EM / 56.22 F1
MuSiQue           26.50 EM / 34.63 F1
PopQA             53.90 EM / 55.73 F1
Macro             41.88 EM / 48.90 F1
Pooled retrieval  93.84%
```

E2 must find 478 skipped steps out of 7,761 and 418 affected questions.

### E4-E6: confidence analyses

```bash
python scripts/analyze_confidence_battery.py
python scripts/analyze_tail_leakage_variants.py
python scripts/analyze_tlsum_confidence_story.py
```

These were rerun against a clean canonical bundle on 2026-08-24. The
representative signal battery, entropy-component AUROCs, selective-accuracy
table, and all 491 confidently-wrong cases match the current thesis.

### E7: path overlap, routing, oracle, and confidence quadrants

```bash
python scripts/make_direct_decompose_2x2_fig.py
python scripts/make_routing_gap_fig.py
python scripts/analyze_quadrant_confidence.py
```

The shared `scripts/accuracy_first_replay.py` reconstructs candidates at
`tau=0.002305` from row-level artifacts; it does not embed the thesis counts.
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

## 3. External-asset commands

### E8: four-dataset answer-containing Hit@5

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

The script joins raw gold answers/aliases with the tracked final traces. It
writes `results/reproduced/answer_containing_recall_4ds_open_domain.md`.

### E9: MuSiQue gold/base decomposition diagnostic

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

### E10: title handling

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
provenance rather than falsely labelled a clean-clone reproduction.

### E11: maximum-entropy live arm

Use the same live command as E14, changing the step signal and corresponding
verifier thresholds:

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

This is the live producer. The exact historical traces underlying the
maximum-entropy thesis row are not part of the minimal Git bundle.

### E12: repeated sampling

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

### E13: decomposer training

```bash
SEEK_DATA_DIR=/tmp/SEEK-decomposer-12k
hf download Chien1011/SEEK-decomposer-12k \
  --repo-type dataset \
  --revision 3dcb5c057a82b6610b7b303855e273bb93f9bab7 \
  --local-dir "$SEEK_DATA_DIR"
(cd "$SEEK_DATA_DIR" && sha256sum -c SHA256SUMS)

git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
git checkout 7e20db5735443861fad91766614b3415ff8966a4
python -m pip install -e .

# Copy the downloaded data and register musique_decomp, twowiki_decomp, and
# popqa_decomp in data/dataset_info.json as documented in DATASETS.md.
mkdir -p data/seek_decomposer_12k
cp -R "$SEEK_DATA_DIR/data/." data/seek_decomposer_12k/

llamafactory-cli train \
  /path/to/Thesis_Research/main_methodology/configs/llama3_8b_decomposer_lora_3ds.yaml
```

The three `dataset_info.json` entries and original construction provenance
are documented in `DATASETS.md`.

### E14: live SEEK

The exact path-aware, original-question-union command is in the root README
and `main_methodology/README.md`. Use `--corpus paragraphs` for a smoke
test without ES, or `--corpus wiki` for the thesis retrieval setting.

## 4. All 13 thesis figures

| Figure | Exact producer/update command | Status |
|---|---|---|
| `Thesis_Overview.pdf` | No command; preserve the tracked PDF | **Manual source gap**: raster-only PDF |
| `prompt_Decomposer.pdf` | Open/export `../Thesis/figure_sources/prompt_Decomposer.drawio` in diagrams.net | Editable source recovered |
| `prompt_Answer.pdf` | Open/export `../Thesis/figure_sources/prompt_Answer.drawio` in diagrams.net | Editable source recovered |
| `fig_renyi_reweight_auroc.pdf` | `cd ../Thesis && python make_ch3_signal_figs.py` | Rebuildable; illustrative constants in script |
| `fig_same_entropy_risk.pdf` | Same Chapter 3 command | Rebuildable |
| `fig_leakage_monotonicity.pdf` | Same Chapter 3 command | Rebuildable |
| `fig_tail_leakage_error_bins.pdf` | Same Chapter 3 command | Rebuildable; source values documented in script |
| `fig_answer_hit_retrieval.png` | E8, then `python scripts/make_answer_hit_fig.py --out ../Thesis/figures/fig_answer_hit_retrieval.png` | **External assets** |
| `fig_3way_ablation.png` | E1, then `python scripts/make_3way_fig.py --out ../Thesis/figures/fig_3way_ablation.png` | **Verified clone**; plots final-system EM |
| `fig_threshold_tradeoff.png` | E1, then `python scripts/make_threshold_tradeoff_fig.py --out ../Thesis/figures/fig_threshold_tradeoff.png` | **Verified clone**; highlights accuracy-first `tau=0.002305` |
| `fig_direct_decompose_2x2_counts.png` | `python scripts/make_direct_decompose_2x2_fig.py --out ../Thesis/figures/fig_direct_decompose_2x2_counts.png` | **Verified clone** |
| `fig_routing_gap_oracle.png` | `python scripts/make_routing_gap_fig.py --out ../Thesis/figures/fig_routing_gap_oracle.png` | **Verified clone** |
| `fig_quadrant_confidence.png` | `python scripts/analyze_quadrant_confidence.py --out ../Thesis/figures/fig_quadrant_confidence.png` | **Verified clone** |

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
excluded from the review artifact.
