# SEEK: Tail-Leakage-Guided Adaptive Retrieval

SEEK is the research code for a thesis on adaptive retrieval-augmented
generation. It decomposes a question into dependent sub-questions, uses
**Tail Leakage** to decide which steps need retrieval, and conservatively
selects between a direct answer and the decomposition-path answer.

This code-review snapshot contains:

- the active implementation and portable experiment scripts in
  `main_methodology/`;
- bundled canonical traces for CPU-only reproduction of the reported
  **41.88 EM / 48.90 F1** result and revised-thesis diagnostics;
- the current thesis LaTeX project, `main.pdf`, revision diff, and all 13
  figures used by the thesis in `Thesis/`;
- an anonymous ICLR 2027 **working draft** in `papers/seek/`. It is
  reproducibly buildable, but it has **not been submitted or published**.

Large licensed models, the LoRA adapter, raw datasets, and the 21M-passage
Wikipedia index are external assets and are not stored in Git.

## 1. Hardware environment

The reported SEEK experiments were developed and checked on this workstation:

| Component | Author workstation |
|---|---|
| Operating system | Ubuntu 24.04.2 LTS |
| CPU | 2 x Intel Xeon Gold 6430, 32 physical cores/socket, 128 logical CPUs total |
| System memory | 251 GiB RAM, 8 GiB swap |
| GPU | 2 x NVIDIA L40S |
| VRAM | 49,140 MiB per GPU (approximately 48 GiB) |
| NVIDIA driver | 575.57.08 |
| PyTorch CUDA runtime | CUDA 12.8 |

Resource requirements depend on the task:

| Task | Practical requirement |
|---|---|
| Bundled offline analyses | CPU only; approximately 8 GiB RAM |
| Thesis/paper LaTeX build | CPU only; a full TeX Live installation |
| Llama-3-8B bf16 inference | One CUDA GPU with at least 24 GiB VRAM recommended |
| Two-shard synthesis replay | One or two CUDA GPUs; the full guide shows GPU 0/1 sharding |
| Full-corpus retrieval | External Elasticsearch 7.10.2 service plus the `wiki` index |

The two L40S cards describe the author machine, not a requirement for the
CPU/offline reproduction path.

## 2. Software environment

The exact environment verified on 2026-08-24 is:

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
| Elasticsearch Python client | 7.17.13 |
| Elasticsearch server used by the thesis | 7.10.2 |
| TeX toolchain | TeX Live 2026, latexmk 4.88, Biber 2.21, pdfLaTeX |

`main_methodology/requirements.txt` is the portable dependency list.
`main_methodology/requirements_thesis.txt` and
`main_methodology/environment_thesis.yml` are exact
workstation provenance snapshots and may contain platform-specific transitive
packages. The root `environment.yml` and `t.txt` are legacy snapshots and
are not the installation source for SEEK.

## 3. Installation

### CPU/offline reviewer environment

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

No `sudo` access is needed for this path.

### GPU environment matching the author workstation

Install the PyTorch wheel matching the machine's CUDA/driver, then install the
remaining requirements. The tested CUDA 12.8 environment can be recreated
with:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

For another CUDA version, select the corresponding PyTorch build instead of
copying the `cu128` command unchanged.

### External retrieval service

Open-domain experiments use an Elasticsearch 7.10.2 `wiki` index built from
`psgs_w100` (approximately 21 million passages). The server and index are
not distributed in Git and the repository does not pretend to rebuild them
from the small trace bundle. Point SEEK at an existing lab service and verify:

```bash
curl -s http://localhost:9200/_cluster/health
curl -s http://localhost:9200/_cat/indices/wiki?v
```

Set `SEEK_ES_HOST` and `SEEK_ES_PORT`, or pass `--es-host` and
`--es-port`, when the service is not on localhost.

## 4. Data, model assets, and outputs

### Included in a normal clone

- Four canonical final traces, four direct confidence traces, eight Tail
  Leakage step shards, two fixed-chain synthesis shards, and six compact
  coverage traces under `main_methodology/results/traces/`.
- These tracked artifacts are sufficient for the verified CPU/offline rows in
  Section 6.
- Thesis and paper LaTeX sources plus their compiled PDFs.

### Required only for live inference, training, or retrieval reruns

| Asset | Expected location/form | Read by |
|---|---|---|
| Base model | `meta-llama/Meta-Llama-3-8B-Instruct` or local path | live inference, synthesis, probes |
| SEEK LoRA adapter | PEFT directory with config and safetensors | decomposition and step answering |
| Evaluation data | JSONL; schema in `main_methodology/README.md` | live inference and retrieval analyses |
| Training data | generated JSON/JSONL registered with LLaMA-Factory | decomposer training |
| Wikipedia retrieval | Elasticsearch `wiki` index | open-domain runs |

Output locations are listed below. New run outputs under `results/generated/`,
`results/reproduced/`, and `logs/` are ignored by Git. The canonical trace
bundle and release figures/PDFs are intentionally tracked exceptions.

| Output path | Contents |
|---|---|
| `main_methodology/results/reproduced/` | deterministic CPU reports and regenerated figures |
| `main_methodology/results/generated/` | new GPU/live traces |
| `main_methodology/logs/script_runs/` | run logs |
| `Thesis/figures/` | thesis-ready figures generated by plotting scripts |
| `Thesis/main.pdf`, `Thesis/main_diff.pdf` | compiled thesis PDFs; these two are tracked |
| `papers/seek/iclr2027/main.pdf` | compiled paper-draft PDF; tracked |

## 5. Repository and documentation map

| Path | Purpose | Reads | Writes / status |
|---|---|---|---|
| `README.md` | Code-review entry point, environment, I/O, command matrix | Repository metadata | Documentation only |
| `main_methodology/README.md` | Full SEEK live-inference, data, training, and fixed-chain guide | Code, trace layout | Documentation only |
| `main_methodology/THESIS_REPRODUCTION.md` | One-to-one thesis experiment and figure provenance | Thesis tables/figures, scripts | Documents exact outputs and external-asset limits |
| `main_methodology/DATASETS.md` | Dataset preparation and split provenance | Raw external datasets | Prepared data under `datasets/` |
| `main_methodology/scripts/README.md` | Active, ablation, diagnostic, and historical script index | `scripts/` | Documentation only |
| `main_methodology/results/README.md` | Canonical trace manifest and result layout | `results/traces/` | Explains `reports/`, `tables/`, `reproduced/` |
| `main_methodology/RESULTS_MASTER.md` | Chronological research ledger | Historical runs | Historical reference; newest canonical values are identified in the reproduction guide |
| `main_methodology/seek/` | Active answerer, pipeline, retriever, and verifier code | Models, JSONL, optional ES | Prediction traces selected by CLI `--out` |
| `main_methodology/configs/` | Portable LLaMA-Factory training configs | Registered training datasets | LoRA checkpoints in configured output directory |
| `Thesis/README.md` | Thesis source map and pdfLaTeX build instructions | `.tex`, `.bib`, `figures/` | `main.pdf`, `main_diff.pdf` |
| `Thesis/figure_sources/` | Figure-to-script/editable-source manifest | Canonical reports/traces | Thesis figures |
| `Thesis/revision_submission/revision_response.md` | Committee revision response | Current/diff thesis | Documentation only |
| `papers/seek/README.md` | Paper-draft status and structure | `papers/seek/iclr2027/` | Documentation only |
| `papers/seek/SUBMISSION_CHECKLIST.md` | Remaining paper-draft work | Draft and experiment gaps | Not evidence of publication |
| `deeprag_compare/` | Earlier DeepRAG comparison/reproduction work | Its own data/results | Historical baseline area; not the active SEEK entry point |
| `experiment/` | Earlier self-certainty, query-rewrite, and diagnostic experiments | Experiment-specific assets | Historical provenance |
| `EXPERIMENTS_SUMMARY.md`, `weekly_2026-04-28.md` | Earlier research summaries | Historical results | Do not use as the current SEEK headline |
| root `environment.yml`, `t.txt` | Legacy environment dumps | N/A | Not used by the active installation |

The active code-review path is `main_methodology/`. Historical directories
remain in Git for provenance, but they are not imported by SEEK.

## 6. Thesis experiment and figure commands

Run these from `main_methodology/` unless a command explicitly changes
directory. Baseline methods reported from prior work are intentionally not
listed. The detailed one-to-one mapping, required inputs, outputs, and expected
values is in
[`main_methodology/THESIS_REPRODUCTION.md`](main_methodology/THESIS_REPRODUCTION.md).

Status legend:

- **verified clone**: runs from the tracked canonical bundle on CPU;
- **external assets**: command is provided, but a model, raw dataset, or ES
  index is intentionally not in Git;
- **provenance only**: the thesis reports a historical result whose full
  row-level assets or editable original are unavailable.

| Thesis item | Command | Status |
|---|---|---|
| Ch. 4 SEEK main row; Ch. 5 retrieval necessity and threshold sensitivity | `python scripts/analyze_tlsum_fixA_step_gate_final.py` | verified clone |
| Ch. 5 skipped-retrieval safety | `python scripts/analyze_skipped_retrieval_questions.py` | verified clone |
| Appendix decomposition failure taxonomy | `python scripts/analyze_accuracy_first_failure_taxonomy.py` | verified clone |
| Ch. 5 confidence-signal AUROC battery | `python scripts/analyze_confidence_battery.py` | verified clone |
| Ch. 5 entropy components, selective accuracy, confidently-wrong categories | `python scripts/analyze_tlsum_confidence_story.py` | verified clone |
| Supporting Tail Leakage variant analysis | `python scripts/analyze_tail_leakage_variants.py` | verified clone |
| Ch. 5 three-way retrieval and threshold figures | `python scripts/analyze_tlsum_fixA_step_gate_final.py && python scripts/make_3way_fig.py && python scripts/make_threshold_tradeoff_fig.py` | verified clone |
| Ch. 5 path overlap, routing/oracle, and Appendix confidence-quadrant figures | `python scripts/make_direct_decompose_2x2_fig.py && python scripts/make_routing_gap_fig.py && python scripts/analyze_quadrant_confidence.py` | verified clone |
| Ch. 3 four Tail Leakage/Renyi explanatory figures | `cd ../Thesis && python make_ch3_signal_figs.py` | verified script; illustrative inputs are documented constants |
| Ch. 5 four-dataset answer-containing Hit@5 and figure | `python scripts/recompute_answer_containing_recall_4ds.py && python scripts/make_answer_hit_fig.py` | external raw dev sets + ES |
| Ch. 5 MuSiQue gold/base decomposition recall | `python scripts/decompose_base_musique.py ...` then `python scripts/recall_gold_decomp_musique.py ...` | external base model + raw MuSiQue + ES |
| Ch. 5 three-arm title policy | `bash scripts/run_tlsum_title_ablation_tau002305.sh off 0`, `global 0`, and `pathaware 0` | external base model + adapter + ES |
| Appendix repeated-sampling probe | `bash scripts/run_m2b_multisample.sh 0 dataset:/path/to/dev.jsonl` | external base model + adapter + data; final appendix aggregation is provenance only |
| Ch. 5 maximum-entropy-vs-Tail-Leakage end-to-end comparison | Tail row uses the headline analyzer; maximum-entropy row uses historical unbundled live traces | provenance only for the maximum-entropy arm |
| Pipeline overview | No generator was found for `Thesis_Overview.pdf` | provenance only; editable original is a known gap |
| Decomposer/answer prompt diagrams | Open `Thesis/figure_sources/*.drawio` with diagrams.net | editable sources recovered from embedded PDF metadata |

### Fast expected result

```bash
python scripts/analyze_tlsum_fixA_step_gate_final.py
```

The `tau=0.002305` accuracy-first row should report:

```text
HotpotQA 37.80/49.03
2Wiki    49.30/56.22
MuSiQue  26.50/34.63
PopQA    53.90/55.73
Macro    41.88/48.90
```

Reports are written under `results/reproduced/`.

## 7. Live GPU inference

After accepting the Meta Llama license and obtaining the SEEK adapter:

```bash
hf auth login

python scripts/run_seek_dual.py \
  --base meta-llama/Meta-Llama-3-8B-Instruct \
  --adapter /path/to/seek-lora-adapter \
  --dev /path/to/dev.jsonl \
  --mode C \
  --signal tail_leakage \
  --tau 0.002305 \
  --corpus wiki \
  --es-host localhost \
  --es-port 9200 \
  --top-k 8 \
  --title-mode pathaware \
  --direct-context-union \
  --no-evidence \
  --direct-confidence-tau 0.0005 \
  --decomp-post-confidence-tau 0.0158 \
  --typeaware-fallback \
  --fallback auto \
  --limit 10 \
  --out results/generated/gpu_smoke.jsonl
```

Use `--corpus paragraphs` with per-question passages to run without
Elasticsearch. Remove `--limit 10` for a full dataset. The detailed JSONL
schema and the distinction between live inference and controlled fixed-chain
replay are in `main_methodology/README.md`.

## 8. Thesis and paper builds

The thesis uses pdfLaTeX, not XeLaTeX:

```bash
cd Thesis
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error main_diff.tex
```

The current outputs are 63 pages and include all referenced figures.

Build the anonymous paper working draft separately:

```bash
cd papers/seek/iclr2027
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The result is a 13-page working draft. It is not evidence of submission or
publication; consult `papers/seek/SUBMISSION_CHECKLIST.md` for open work.

## 9. What is not in Git

The normal clone deliberately excludes:

- gated Meta Llama weights;
- the final SEEK LoRA adapter;
- raw evaluation and training datasets;
- the full Wikipedia Elasticsearch data directory;
- newly generated traces and run logs, caches, full checkpoint weights, and
  intermediate optimizer state;
- unfinished Qwen cross-backbone sweep scripts/results.

Those exclusions do not affect the verified clone/offline commands above.
They do affect live, training, open-domain retrieval, and historical
provenance-only rows, which are labelled explicitly.
