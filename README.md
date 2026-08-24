# SEEK: Self-certainty Evaluated Evidence-gated Knowledge-retrieval

SEEK is the research code for a thesis on adaptive retrieval-augmented
generation. It decomposes a question into dependent sub-questions, uses
**Tail Leakage** to decide which steps need retrieval, and conservatively
selects between a direct answer and the decomposition-path answer.

This repository contains the active code, four 1,000-question evaluation
manifests, a canonical trace bundle for CPU-only analysis, the complete thesis
LaTeX project and PDF, and an anonymous ICLR 2027 working draft. The paper is a
**draft only**: it has not been submitted or published.

## 1. Hardware environment

| Component | Author workstation |
|---|---|
| Operating system | Ubuntu 24.04.2 LTS |
| CPU | 2 × Intel Xeon Gold 6430, 32 physical cores/socket, 128 logical CPUs total |
| System memory | 251 GiB RAM |
| GPU | 2 × NVIDIA L40S |
| VRAM | 49,140 MiB per GPU (approximately 48 GiB) |
| NVIDIA driver | 575.57.08 |
| PyTorch CUDA runtime | CUDA 12.8 |

Offline analyses need only a CPU and about 8 GiB RAM. Live Llama-3-8B bf16
inference needs at least 24 GiB VRAM; full-corpus runs also need the
Elasticsearch `wiki` index.

## 2. Software environment

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

Install from [main_methodology/requirements.txt](main_methodology/requirements.txt).

## 3. Installation

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

For the tested CUDA 12.8 environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
hf auth login
```

### Open-domain retrieval

The thesis uses Elasticsearch 7.10.2 with a `wiki` index built from the
approximately 21M-passage `psgs_w100` corpus.

```bash
curl -s http://localhost:9200/_cluster/health
curl -s http://localhost:9200/_cat/indices/wiki?v

export SEEK_ES_HOST=localhost
export SEEK_ES_PORT=9200
```

## 4. Repository, inputs, and outputs

### Repository structure

| Path | Purpose |
|---|---|
| [main_methodology/seek](main_methodology/seek) | SEEK pipeline, retriever, answerer, and verifier |
| [main_methodology/scripts](main_methodology/scripts) | Experiment, analysis, and figure commands |
| [main_methodology/configs](main_methodology/configs) | Decomposer training configuration |
| [main_methodology/datasets/evaluation](main_methodology/datasets/evaluation) | Four 1,000-question evaluation files |
| [main_methodology/results/traces](main_methodology/results/traces) | Bundled inputs for offline reproduction |
| [Thesis](Thesis) | Thesis LaTeX, PDFs, and 13 figures |
| [papers/seek](papers/seek) | Anonymous paper draft LaTeX and PDF |
| `deeprag_compare/`, `experiment/` | Historical baseline and exploratory work |

### Four released evaluation sets

SEEK uses the first 1,000 rows without shuffling, matching
[DeepRAG's selection rule](https://github.com/gxy-gxy/DeepRAG/blob/1665780b7b6317375d56e71c3ddc52d6f706c380/src/main.py#L176-L183).
HotpotQA, 2Wiki, and MuSiQue use `dev`; PopQA uses `test`.

| Dataset | File | Rows |
|---|---|---:|
| [HotpotQA](https://github.com/hotpotqa/hotpot) | [hotpotqa.jsonl](main_methodology/datasets/evaluation/hotpotqa.jsonl) | 1,000 |
| [2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop) | [2wikimultihopqa.jsonl](main_methodology/datasets/evaluation/2wikimultihopqa.jsonl) | 1,000 |
| [MuSiQue](https://github.com/StonyBrookNLP/musique) | [musique.jsonl](main_methodology/datasets/evaluation/musique.jsonl) | 1,000 |
| [PopQA](https://github.com/AlexTMallen/adaptive-retrieval#popqa) | [popqa.jsonl](main_methodology/datasets/evaluation/popqa.jsonl) | 1,000 |

Rows contain `qid`, `question`, `answer`, and `answer_id`. They are direct
inputs for `--corpus wiki`; `--corpus paragraphs` requires paragraph-bearing
JSONL instead.

Verify from `main_methodology/`:

```bash
(cd datasets/evaluation && sha256sum -c SHA256SUMS)
wc -l datasets/evaluation/*.jsonl
```

<details>
<summary><strong>Rebuild the four manifests from official downloads</strong></summary>

```bash
python scripts/prepare_eval_subsets.py \
  --hotpotqa /data/hotpot_dev_distractor_v1.json \
  --twowiki /data/2wiki_dev.json \
  --musique /data/musique_ans_v1.0_dev.jsonl \
  --popqa /data/popQA.tsv \
  --output-dir /tmp/seek-evaluation \
  --force
```

</details>

### Runtime assets and outputs

| Item | Location | Use |
|---|---|---|
| Base model | `meta-llama/Meta-Llama-3-8B-Instruct` | External; live inference and training |
| SEEK adapter | PEFT adapter directory | External; live inference or produced by E13 |
| 12K training set | 4K MuSiQue + 4K 2Wiki + 4K PopQA | External; E13 |
| Raw/paragraph data | `datasets/{dataset}/raw/` | External; E8, E9, E12 |
| Wikipedia index | Elasticsearch `wiki` | External; open-domain retrieval |
| Bundled traces | `results/traces/` | Included; E1–E7 inputs |
| Generated results | `results/reproduced/`, `results/generated/` | Reports, figures, and new traces |
| PDFs | `../Thesis/main*.pdf`, `../papers/seek/iclr2027/main.pdf` | Thesis and paper draft outputs |

## 5. Thesis experiments and figures

Baselines from earlier papers are excluded. E1–E7 run on CPU from the bundled
`results/traces/`; E8–E14 list their additional requirements.

The **41.88 EM / 48.90 F1** result is a fixed-chain replay. E14 is live
sequential inference, so a fresh E14 run is not expected to be bit-identical.

### E1–E7: offline reproduction

| ID | Thesis content | Output |
|---|---|---|
| E1 | Ch. 4 main row; Ch. 5 retrieval necessity and threshold sweep | `tlsum_fixA_step_gate_final.{md,json}` |
| E2 | Ch. 5 skipped-retrieval safety | Markdown, JSON, and case JSONL |
| E3 | Appendix failure taxonomy | `decomp_failure_taxonomy_tau0p002305.{md,json}` |
| E4 | Ch. 5 confidence-signal AUROC | `confidence_battery_analysis.md` |
| E5 | Ch. 5 entropy/selective accuracy; Appendix confidently-wrong cases | `tlsum_confidence_story.md` |
| E6 | Ch. 5 Tail Leakage variants and bootstrap | `tail_leakage_variants.md` |
| E7 | Ch. 5 routing/oracle and Appendix confidence quadrants | Three reports and three PNGs |

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

Outputs go to `results/reproduced/`. E1 at `tau=0.002305` gives:

```text
HotpotQA  37.80 EM / 49.03 F1
2Wiki     49.30 EM / 56.22 F1
MuSiQue   26.50 EM / 34.63 F1
PopQA     53.90 EM / 55.73 F1
Macro     41.88 EM / 48.90 F1
Pooled step retrieval rate: 93.84%
```

### E8–E14: external runs

| ID | Thesis content | Needs |
|---|---|---|
| E8 | Answer-containing Hit@5 | Raw inputs + ES; the figure alone uses a bundled report |
| E9 | MuSiQue decomposition recall | Base model + raw input + ES |
| E10 | Title-policy comparison | Base model + adapter + ES |
| E11 | Maximum entropy vs. Tail Leakage | Historical max-entropy traces are not bundled |
| E12 | Repeated-sampling errors | Base model + adapter + paragraph data |
| E13 | Decomposer training | Base model + 12K training set |
| E14 | Live SEEK | Base model + adapter + ES |

<details>
<summary><strong>Show exact E8-E14 commands, inputs, and outputs</strong></summary>

Set common model paths once:

```bash
export BASE_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
export SEEK_ADAPTER=/absolute/path/to/seek-lora-adapter
mkdir -p results/generated
```

**E8 — answer-containing Hit@5.** Uses normalized raw files, bundled traces,
and ES; writes the report and plot to `results/reproduced/`.

```bash
python scripts/recompute_answer_containing_recall_4ds.py \
  --ks 1 3 5 8 10 \
  --corpus wiki \
  --out results/reproduced/answer_containing_recall_4ds_open_domain.md

python scripts/make_answer_hit_fig.py \
  --input results/reproduced/answer_containing_recall_4ds_open_domain.md \
  --out results/reproduced/fig_answer_hit_retrieval.png
```

The figure alone can use the bundled report:

```bash
python scripts/make_answer_hit_fig.py \
  --input results/reports/answer_containing_recall_4ds_open_domain.md \
  --out results/reproduced/fig_answer_hit_retrieval.png
```

**E9 — MuSiQue decomposition recall.** Produces a base-decomposition JSONL,
detail JSONL, and recall table.

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

**E10 — title-policy comparison.** Writes per-arm traces under
`results/generated/title_policy/`; the thesis table aggregation is historical.

```bash
bash scripts/run_tlsum_title_ablation_tau002305.sh off 0
bash scripts/run_tlsum_title_ablation_tau002305.sh global 0
bash scripts/run_tlsum_title_ablation_tau002305.sh pathaware 0
```

**E11 — maximum entropy vs. Tail Leakage.** E1 supplies the Tail Leakage row;
this command creates a new max-entropy run because the historical traces are
not bundled.

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

**E12 — repeated-sampling errors.** Writes 15 samples per question; the final
appendix aggregation is historical.

```bash
bash scripts/run_m2b_multisample.sh \
  0 \
  hotpotqa:datasets/hotpotqa/raw/dev_first1000.jsonl \
  2wiki:datasets/2wiki/raw/dev_first1000.jsonl \
  musique:datasets/musique/raw/dev_first1000.jsonl \
  popqa:datasets/popqa/raw/dev.jsonl
```

**E13 — decomposer training.** Register `musique_decomp`, `twowiki_decomp`,
and `popqa_decomp` in LLaMA-Factory's `data/dataset_info.json`.

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

**E14 — live SEEK.** Writes one prediction trace per dataset.

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

| Figure | Producer or editable source |
|---|---|
| `Thesis_Overview.pdf` | Tracked PDF; editable source unavailable |
| `prompt_Decomposer.pdf` | `../Thesis/figure_sources/prompt_Decomposer.drawio` |
| `prompt_Answer.pdf` | `../Thesis/figure_sources/prompt_Answer.drawio` |
| `fig_renyi_reweight_auroc.pdf` | `../Thesis/make_ch3_signal_figs.py` |
| `fig_same_entropy_risk.pdf` | `../Thesis/make_ch3_signal_figs.py` |
| `fig_leakage_monotonicity.pdf` | `../Thesis/make_ch3_signal_figs.py` |
| `fig_tail_leakage_error_bins.pdf` | `../Thesis/make_ch3_signal_figs.py` |
| `fig_answer_hit_retrieval.png` | `make_answer_hit_fig.py` |
| `fig_3way_ablation.png` | E1 + `make_3way_fig.py` |
| `fig_threshold_tradeoff.png` | E1 + `make_threshold_tradeoff_fig.py` |
| `fig_direct_decompose_2x2_counts.png` | `make_direct_decompose_2x2_fig.py` |
| `fig_routing_gap_oracle.png` | `make_routing_gap_fig.py` |
| `fig_quadrant_confidence.png` | `analyze_quadrant_confidence.py` |

Plotters default to `results/reproduced/`; the commands below write the files
used by LaTeX.

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

The prompt diagrams are exported with diagrams.net. Chapter 3 figure values
are stored in `make_ch3_signal_figs.py`.

### Thesis and paper builds

```bash
(cd ../Thesis && \
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex && \
  latexmk -pdf -interaction=nonstopmode -halt-on-error main_diff.tex)
```

The two thesis PDFs have 63 pages. Build the paper draft with:

```bash
(cd ../papers/seek/iclr2027 && \
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)
```

The paper draft has 13 pages and is not published.

### Limits

- Gated Llama weights, the final adapter, 12K training data, and the 21M
  Wikipedia index are not in Git.
- E8, E9, and E12 need additional raw data. Historical aggregation artifacts
  for E10–E12 are not bundled.
- `Thesis_Overview.pdf` has no editable source; the prompt diagrams do.
