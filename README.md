# SEEK: Self-certainty Evaluated Evidence-gated Knowledge-retrieval

SEEK is the research code for a thesis on adaptive retrieval-augmented
generation. It decomposes a question into dependent sub-questions, uses
**Tail Leakage** to decide which steps need retrieval, and conservatively
selects between a direct answer and the decomposition-path answer.

This repository contains the active code, four 1,000-question evaluation
manifests, a canonical trace bundle for CPU-only analysis, the complete thesis
LaTeX project and PDF, and an anonymous ICLR 2027 working draft. The paper is a
**draft only**: it has not been submitted or published.

The quickest verified check is:

```bash
git clone https://github.com/HY-Chien/Thesis_Research.git
cd Thesis_Research/main_methodology
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/analyze_tlsum_fixA_step_gate_final.py
```

The accuracy-first `tau=0.002305` row should report **41.88 EM / 48.90
F1**. The remainder of this README is the complete code-review entry point;
the reviewer does not need to find another reproduction guide.

## 1. Hardware environment

The reported experiments were developed and checked on this workstation:

| Component | Author workstation |
|---|---|
| Operating system | Ubuntu 24.04.2 LTS |
| CPU | 2 × Intel Xeon Gold 6430, 32 physical cores/socket, 128 logical CPUs total |
| System memory | 251 GiB RAM |
| GPU | 2 × NVIDIA L40S |
| VRAM | 49,140 MiB per GPU (approximately 48 GiB) |
| NVIDIA driver | 575.57.08 |
| PyTorch CUDA runtime | CUDA 12.8 |

The bundled analyses and plots are CPU-only and need approximately 8 GiB RAM.
Llama-3-8B bf16 inference needs one CUDA GPU with at least 24 GiB VRAM.
Full-corpus experiments additionally need an Elasticsearch `wiki` index.
The two L40S cards describe the author machine, not a requirement for the
offline reproduction.

## 2. Software environment

These versions were verified on 2026-08-24:

| Software | Tested version |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.10.0+cu128 |
| Transformers | 5.6.0 |
| Accelerate | 1.11.0 |
| PEFT | 0.18.1 |
| Datasets | 4.0.0 |
| NumPy / SciPy | 2.4.6 / 1.17.1 |
| Matplotlib | 3.10.9 |
| SentencePiece | 0.2.1 |
| rank-bm25 | 0.2.2 |
| Elasticsearch Python client / server | 7.17.13 / 7.10.2 |
| TeX toolchain | TeX Live 2026, latexmk 4.88, Biber 2.21, pdfLaTeX |

Use [main_methodology/requirements.txt](main_methodology/requirements.txt) for
installation. The files `requirements_thesis.txt` and
`environment_thesis.yml` are exact workstation snapshots, while the root
`environment.yml` and `t.txt` are legacy records and are not installation
sources.

## 3. Installation

Run all Python commands below from `Thesis_Research/main_methodology/`.
Neither the CPU path nor the virtual environment requires `sudo`.

### CPU/offline review

```bash
git clone https://github.com/HY-Chien/Thesis_Research.git
cd Thesis_Research/main_methodology

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python scripts/run_seek_dual.py --help
python scripts/analyze_tlsum_fixA_step_gate_final.py
```

### GPU environment

Install the PyTorch wheel matching the host driver before the other
requirements. The command used for the tested CUDA 12.8 environment is:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
hf auth login
```

Use the corresponding PyTorch build instead of `cu128` on a machine with a
different CUDA/driver combination.

### Open-domain retrieval

The thesis uses Elasticsearch 7.10.2 with a `wiki` index built from
`psgs_w100`, approximately 21 million non-overlapping 100-word Wikipedia
passages. The index is external and is not stored in Git.

```bash
curl -s http://localhost:9200/_cluster/health
curl -s http://localhost:9200/_cat/indices/wiki?v

export SEEK_ES_HOST=localhost
export SEEK_ES_PORT=9200
```

## 4. Repository, inputs, and outputs

### Repository map

| Path | Purpose | Reads | Writes / status |
|---|---|---|---|
| [main_methodology/seek](main_methodology/seek) | Active answerer, pipeline, retriever, and verifier | Model, adapter, evaluation JSONL, optional ES | Returns predictions to the selected CLI |
| [main_methodology/scripts](main_methodology/scripts) | Live runners, analysis, ablations, and plot producers | Evaluation data and `results/traces/` | `results/generated/`, `results/reproduced/`, and logs |
| [main_methodology/configs](main_methodology/configs) | LLaMA-Factory decomposer configurations | Registered 12K training set and base model | LoRA adapter under the configured `output_dir` |
| [main_methodology/datasets/evaluation](main_methodology/datasets/evaluation) | Four released 1,000-question evaluation manifests | Official dataset splits | Read-only input for full-corpus live runs |
| [main_methodology/results/traces](main_methodology/results/traces) | Tracked canonical experiment artifacts | Historical GPU runs | Input to deterministic CPU reproduction |
| `main_methodology/results/reproduced/` | Reviewer-generated reports and figures | Canonical traces/reports | Generated on demand; ignored by Git |
| [Thesis](Thesis) | Current thesis LaTeX, PDF, revision diff, and 13 figures | `.tex`, bibliography, figures | `main.pdf`, `main_diff.pdf` |
| [papers/seek](papers/seek) | Anonymous paper working draft | Paper LaTeX and bibliography | `iclr2027/main.pdf`; not a publication |
| `deeprag_compare/`, `experiment/` | Earlier baseline and exploratory work | Their own historical assets | Provenance only; not imported by active SEEK |

Optional detailed indexes remain in
[main_methodology/DATASETS.md](main_methodology/DATASETS.md),
[main_methodology/scripts/README.md](main_methodology/scripts/README.md),
[main_methodology/results/README.md](main_methodology/results/README.md), and
[Thesis/figure_sources/README.md](Thesis/figure_sources/README.md). All
commands needed for code review are already included in this root README.

### Four released evaluation sets

These files were generated independently from the official dataset releases.
They follow DeepRAG's deterministic first-1,000 convention: its pinned
[`run.sh`](https://github.com/gxy-gxy/DeepRAG/blob/1665780b7b6317375d56e71c3ddc52d6f706c380/scripts/inference/run.sh)
passes `--sample 1000` without `--shuffle`, and
[`main.py`](https://github.com/gxy-gxy/DeepRAG/blob/1665780b7b6317375d56e71c3ddc52d6f706c380/src/main.py#L176-L183)
therefore takes the source split's first 1,000 rows. They are SEEK release
artifacts, not evaluation files distributed by DeepRAG, and contain exactly
4,000 records in total.

| Dataset | Official split and selection | Released file | Rows |
|---|---|---|---:|
| [HotpotQA](https://github.com/hotpotqa/hotpot) | Distractor development split, first 1,000 | [hotpotqa.jsonl](main_methodology/datasets/evaluation/hotpotqa.jsonl) | 1,000 |
| [2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop) | Development split, first 1,000 | [2wikimultihopqa.jsonl](main_methodology/datasets/evaluation/2wikimultihopqa.jsonl) | 1,000 |
| [MuSiQue](https://github.com/StonyBrookNLP/musique) | Development split, first 1,000 | [musique.jsonl](main_methodology/datasets/evaluation/musique.jsonl) | 1,000 |
| [PopQA](https://github.com/AlexTMallen/adaptive-retrieval#popqa) | Test split, first 1,000 | [popqa.jsonl](main_methodology/datasets/evaluation/popqa.jsonl) | 1,000 |

Each compact row has this schema:

```json
{
  "qid": "question identifier",
  "question": "question text",
  "answer": "one answer or a list of accepted aliases",
  "answer_id": null
}
```

The tracked manifests are sufficient for `--corpus wiki` live runs because
retrieval comes from Elasticsearch. For `--corpus paragraphs`, add a
`paragraphs` array whose entries contain `title` and `paragraph_text`.
The live runners accept both `qid` and `id`, and both scalar and list-valued
`answer`.

Verify all four released files from `main_methodology/`:

```bash
(cd datasets/evaluation && sha256sum -c SHA256SUMS)
wc -l datasets/evaluation/*.jsonl
```

To reconstruct the same four manifests from official local source files:

```bash
python scripts/prepare_eval_subsets.py \
  --hotpotqa /data/hotpot_dev_distractor_v1.json \
  --twowiki /data/2wiki_dev.json \
  --musique /data/musique_ans_v1.0_dev.jsonl \
  --popqa /data/popQA.tsv \
  --output-dir /tmp/seek-evaluation \
  --force
```

### Other inputs and output locations

| Asset or output | Location/form | Availability |
|---|---|---|
| Base model | `meta-llama/Meta-Llama-3-8B-Instruct` or local equivalent | External gated model |
| Final SEEK decomposer | PEFT directory with adapter config and safetensors | External to Git; can be trained with E13 |
| 12K decomposer training set | 4K MuSiQue + 4K 2Wiki + 4K PopQA | External to Git |
| Full paragraph-rich/raw dataset copies | `datasets/{dataset}/raw/` | External to Git; used by E8, E9, and E12 |
| Wikipedia retrieval index | Elasticsearch `wiki` over `psgs_w100` | External to Git |
| Canonical offline artifacts | `results/traces/`: 4 final, 4 confidence, 8 step, 2 synthesis, 6 coverage files | Included |
| Offline reports and plots | `results/reproduced/` | Generated, ignored by Git |
| New GPU traces | `results/generated/` | Generated, ignored by Git |
| Run logs | `logs/script_runs/` | Generated, ignored by Git |
| Thesis figures and PDFs | `../Thesis/figures/`, `../Thesis/main*.pdf` | Source assets and release PDFs included |
| Paper draft PDF | `../papers/seek/iclr2027/main.pdf` | Included working draft |

## 5. Thesis experiments and figures

Baselines copied from earlier papers are intentionally excluded. Status labels
used below are:

- **verified clone** — runs on CPU from tracked artifacts;
- **external assets** — the exact producer is present, but a model, raw data,
  adapter, or ES index is external;
- **historical only** — a fresh producer exists, but the exact row-level
  historical inputs or final aggregation behind the thesis table are absent;
- **manual source gap** — the rendered asset exists but no generator/editable
  original was recovered.

The reported **41.88 EM / 48.90 F1** is a controlled fixed-chain replay:
thresholds select between stored no-retrieval and retrieved step answers, and
the selected chains use stored base-model synthesis. Later resolved
sub-questions remain fixed. E14 is the active sequential live implementation;
because an earlier answer can change a later query, a fresh E14 run is not
claimed to reproduce the fixed-chain number bit-for-bit.

### E1-E7: experiments reproducible after a normal clone

| ID | Thesis content | Input | Output | Status |
|---|---|---|---|---|
| E1 | Ch. 4 SEEK main row; Ch. 5 retrieval necessity and threshold sensitivity | 18 canonical final/confidence/step/synthesis traces | `tlsum_fixA_step_gate_final.{md,json}` | verified clone |
| E2 | Ch. 5 skipped-retrieval safety | E1 artifacts | Markdown, JSON, and case JSONL | verified clone |
| E3 | Appendix decomposition failure taxonomy | E1 artifacts and six coverage traces | `decomp_failure_taxonomy_tau0p002305.{md,json}` | verified clone |
| E4 | Ch. 5 representative confidence-signal AUROC battery | Four confidence traces | `confidence_battery_analysis.md` | verified clone |
| E5 | Ch. 5 entropy components and selective accuracy; Appendix confidently-wrong categories | Four confidence traces | `tlsum_confidence_story.md` | verified clone |
| E6 | Ch. 5 summed-vs-weakest-token result and paired bootstrap | Four confidence traces | `tail_leakage_variants.md` | verified clone |
| E7 | Ch. 5 path choice/overlap, routing/oracle; Appendix confidence quadrants | Accuracy-first replay artifacts | Three reports and three PNGs | verified clone |

```bash
# E1
python scripts/analyze_tlsum_fixA_step_gate_final.py

# E2
python scripts/analyze_skipped_retrieval_questions.py

# E3
python scripts/analyze_accuracy_first_failure_taxonomy.py

# E4
python scripts/analyze_confidence_battery.py

# E5
python scripts/analyze_tlsum_confidence_story.py

# E6
python scripts/analyze_tail_leakage_variants.py

# E7
python scripts/make_direct_decompose_2x2_fig.py
python scripts/make_routing_gap_fig.py
python scripts/analyze_quadrant_confidence.py
```

All outputs go to `results/reproduced/`. For E1, use the
`tau=0.002305` row in the threshold table; the report also retains the older
`tau=0.008461` source-chain operating point for provenance.

```text
HotpotQA  37.80 EM / 49.03 F1
2Wiki     49.30 EM / 56.22 F1
MuSiQue   26.50 EM / 34.63 F1
PopQA     53.90 EM / 55.73 F1
Macro     41.88 EM / 48.90 F1
Pooled step retrieval rate: 93.84%
```

### E8-E14: experiments requiring external assets or historical provenance

| ID | Thesis content | Status |
|---|---|---|
| E8 | Four-dataset answer-containing Hit@5 | external raw inputs + ES; figure alone is verified from clone |
| E9 | MuSiQue original/SEEK/gold/base decomposition recall | external base model + raw input + ES |
| E10 | Title-off/title-on/path-aware comparison | external assets; final aggregation is historical |
| E11 | Maximum token entropy versus Tail Leakage | historical maximum-entropy row; fresh producer included |
| E12 | Repeated-sampling confidence errors | external assets; final appendix aggregation is historical |
| E13 | 12K decomposer training | external training data + base model |
| E14 | Live final SEEK system | external base model + adapter + ES |

<details>
<summary><strong>Show exact E8-E14 commands, inputs, and outputs</strong></summary>

Set common model paths once:

```bash
export BASE_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
export SEEK_ADAPTER=/absolute/path/to/seek-lora-adapter
mkdir -p results/generated
```

**E8 — four-dataset answer-containing Hit@5.** This recomputation reads the
four normalized files under `datasets/{dataset}/raw/`, the tracked final
traces, and the external ES `wiki` index. It writes the report and plot under
`results/reproduced/`. The compact released manifests document the exact
question set, but the current E8 loader still uses the normalized raw copies.
Status: **external assets**.

```bash
python scripts/recompute_answer_containing_recall_4ds.py \
  --ks 1 3 5 8 10 \
  --corpus wiki \
  --out results/reproduced/answer_containing_recall_4ds_open_domain.md

python scripts/make_answer_hit_fig.py \
  --input results/reproduced/answer_containing_recall_4ds_open_domain.md \
  --out results/reproduced/fig_answer_hit_retrieval.png
```

The figure alone can be rebuilt without ES from the tracked canonical report:

```bash
python scripts/make_answer_hit_fig.py \
  --input results/reports/answer_containing_recall_4ds_open_domain.md \
  --out results/reproduced/fig_answer_hit_retrieval.png
```

**E9 — MuSiQue original/SEEK/gold/base decomposition recall.** Inputs are the
normalized MuSiQue development slice, base model, tracked SEEK trace, and ES
`wiki`. Outputs are a base-decomposition JSONL, row-level detail JSONL, and
the printed recall table. Status: **external assets**.

```bash
python scripts/decompose_base_musique.py \
  --base "$BASE_MODEL" \
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

**E10 — title-off/title-on/path-aware comparison.** The launcher reads the
base model, adapter, tracked source traces, and ES `wiki`; it writes per-arm
JSONL and logs under `results/generated/title_policy/`. The exact thesis
table aggregation is historical and is not labelled a clean-clone
reproduction. Status: **external assets + historical aggregation**.

```bash
bash scripts/run_tlsum_title_ablation_tau002305.sh off 0
bash scripts/run_tlsum_title_ablation_tau002305.sh global 0
bash scripts/run_tlsum_title_ablation_tau002305.sh pathaware 0
```

**E11 — maximum token entropy versus Tail Leakage.** The Tail Leakage row is
E1. The following exact command produces a fresh maximum-entropy live arm for
all four released datasets. The row-level historical maximum-entropy traces
used by the thesis are not bundled, so this is a new run rather than a
bit-for-bit reconstruction of that row. Status: **historical only for the
thesis maximum-entropy row**.

```bash
for DATASET in hotpotqa 2wikimultihopqa musique popqa; do
  CUDA_VISIBLE_DEVICES=0 python scripts/run_seek_dual.py \
    --base "$BASE_MODEL" \
    --adapter "$SEEK_ADAPTER" \
    --dev "datasets/evaluation/${DATASET}.jsonl" \
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
    --out "results/generated/maxent_${DATASET}.jsonl"
done
```

**E12 — repeated-sampling confidence errors.** This probe reads the model,
adapter, and paragraph-rich normalized evaluation files and writes 15 samples
per question under `results/generated/multisample/`. The appendix's final
identical-answer/zero-mean-entropy aggregation has no dedicated current writer.
Status: **external assets + historical aggregation**.

```bash
bash scripts/run_m2b_multisample.sh \
  0 \
  hotpotqa:datasets/hotpotqa/raw/dev_first1000.jsonl \
  2wiki:datasets/2wiki/raw/dev_first1000.jsonl \
  musique:datasets/musique/raw/dev_first1000.jsonl \
  popqa:datasets/popqa/raw/dev.jsonl
```

**E13 — decomposer training.** Register the external
`musique_decomp`, `twowiki_decomp`, and `popqa_decomp` datasets in
LLaMA-Factory's `data/dataset_info.json`. The config reads those 12,000
examples and the base model, and writes the LoRA adapter to
`saves/seek/llama3-8b-decomposer-musique-2wiki-popqa`. Status:
**external assets**.

```bash
cd ../..
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
git checkout 7e20db5735443861fad91766614b3415ff8966a4
python -m pip install -e .
llamafactory-cli train \
  ../Thesis_Research/main_methodology/configs/llama3_8b_decomposer_lora_3ds.yaml
cd ../Thesis_Research/main_methodology
```

**E14 — live final SEEK system.** This is the active sequential producer. It
reads the base model, adapter, released 1,000-question manifest, and ES
`wiki`, then writes one prediction trace per dataset. Use
`--corpus paragraphs` only with paragraph-rich JSONL. Status:
**external assets**.

```bash
for DATASET in hotpotqa 2wikimultihopqa musique popqa; do
  CUDA_VISIBLE_DEVICES=0 python scripts/run_seek_dual.py \
    --base "$BASE_MODEL" \
    --adapter "$SEEK_ADAPTER" \
    --dev "datasets/evaluation/${DATASET}.jsonl" \
    --mode C \
    --signal tail_leakage \
    --tau 0.002305 \
    --corpus wiki \
    --es-host "$SEEK_ES_HOST" \
    --es-port "$SEEK_ES_PORT" \
    --top-k 8 \
    --title-mode pathaware \
    --direct-context-union \
    --no-evidence \
    --direct-confidence-tau 0.0005 \
    --decomp-post-confidence-tau 0.0158 \
    --typeaware-fallback \
    --fallback auto \
    --out "results/generated/seek_${DATASET}.jsonl"
done
```

</details>

### All 13 thesis figures

| Figure | Producer or editable source | Status |
|---|---|---|
| `Thesis_Overview.pdf` | Preserve the tracked PDF; no producer was found | manual source gap |
| `prompt_Decomposer.pdf` | `../Thesis/figure_sources/prompt_Decomposer.drawio` | editable diagrams.net source recovered |
| `prompt_Answer.pdf` | `../Thesis/figure_sources/prompt_Answer.drawio` | editable diagrams.net source recovered |
| `fig_renyi_reweight_auroc.pdf` | `../Thesis/make_ch3_signal_figs.py` | script-rebuildable from documented constants |
| `fig_same_entropy_risk.pdf` | Same Chapter 3 producer | script-rebuildable illustrative figure |
| `fig_leakage_monotonicity.pdf` | Same Chapter 3 producer | script-rebuildable illustrative figure |
| `fig_tail_leakage_error_bins.pdf` | Same Chapter 3 producer | script-rebuildable from documented values |
| `fig_answer_hit_retrieval.png` | `make_answer_hit_fig.py` and the tracked report, or E8 | plot verified from clone; metric recomputation needs ES |
| `fig_3way_ablation.png` | E1, then `make_3way_fig.py` | verified clone |
| `fig_threshold_tradeoff.png` | E1, then `make_threshold_tradeoff_fig.py` | verified clone |
| `fig_direct_decompose_2x2_counts.png` | `make_direct_decompose_2x2_fig.py` | verified clone |
| `fig_routing_gap_oracle.png` | `make_routing_gap_fig.py` | verified clone |
| `fig_quadrant_confidence.png` | `analyze_quadrant_confidence.py` | verified clone |

The four Chapter 3 files are written directly into `Thesis/figures/`.
The other plotters normally write to `results/reproduced/`; the explicit
`--out` paths below update the LaTeX assets.

<details>
<summary><strong>Show exact commands to regenerate the script-built figures</strong></summary>

```bash
(cd ../Thesis && python make_ch3_signal_figs.py)

python scripts/make_answer_hit_fig.py \
  --input results/reports/answer_containing_recall_4ds_open_domain.md \
  --out ../Thesis/figures/fig_answer_hit_retrieval.png

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

</details>

The two Draw.io files are manual editable sources and are exported with
diagrams.net. The four Chapter 3 figures contain documented illustrative or
reported constants; rebuilding them verifies the source-to-figure path, not a
new raw-data experiment.

### Thesis and paper builds

The thesis uses pdfLaTeX:

```bash
(cd ../Thesis && \
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex && \
  latexmk -pdf -interaction=nonstopmode -halt-on-error main_diff.tex)
```

Both current thesis PDFs have 63 pages and all 13 referenced figure files are
included. `main_diff.tex` is buildable, although the exact historical source
used by the original latexdiff run is no longer available.

Build the anonymous paper draft separately:

```bash
(cd ../papers/seek/iclr2027 && \
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)
```

The current paper output has 13 pages. It remains a working draft, not evidence
of submission or publication.

### Known external and provenance limits

- Gated Llama weights, the final adapter, 12K training data, and the 21M
  Wikipedia index are not in Git.
- E8, E9, and E12 require normalized or paragraph-rich raw inputs in addition
  to the released compact evaluation manifests.
- E10's final table aggregation, E11's historical maximum-entropy row, and
  E12's final appendix aggregation cannot be reproduced bit-for-bit from the
  minimal clone; their available producers and limits are stated above.
- `Thesis_Overview.pdf` has no recovered editable source. The two prompt
  diagrams do have recovered Draw.io sources.
- Newly generated traces, reports, figures, and logs are ignored by Git unless
  deliberately promoted as release artifacts.
