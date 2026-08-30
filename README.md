# SEEK: Self-certainty Evaluated Evidence-gated Knowledge-retrieval

SEEK is the research code for a thesis on adaptive retrieval-augmented
generation. It decomposes a question into dependent sub-questions, uses
**Tail Leakage** to decide which steps need retrieval, and conservatively
selects between a direct answer and the decomposition-path answer.

## Hardware Environment

- OS: Ubuntu 24.04.2 LTS
- CPU: 2 × Intel Xeon Gold 6430, 32 physical cores per socket
- System memory: 251 GiB
- GPU: 2 × NVIDIA L40S
- VRAM: approximately 48 GiB per GPU
- NVIDIA driver: 575.57.08
- CUDA runtime used by PyTorch: 12.8

The bundled offline analyses require only a CPU and approximately 8 GiB RAM.
Live Llama-3-8B BF16 inference requires at least 24 GiB VRAM. Open-domain
retrieval additionally requires the Elasticsearch `wiki` index.

## Software Environment

- Python: 3.11.15
- PyTorch: 2.10.0+cu128
- Transformers: 5.6.0
- Accelerate: 1.11.0
- PEFT: 0.18.1
- Datasets: 4.0.0
- NumPy: 2.4.6
- SciPy: 1.17.1
- Matplotlib: 3.10.9
- SentencePiece: 0.2.1
- rank-bm25: 0.2.2
- Elasticsearch client/server: 7.17.13/7.10.2
- TeX: TeX Live 2026, latexmk 4.88, Biber 2.21, pdfLaTeX

Python dependencies are listed in
[`main_methodology/requirements.txt`](main_methodology/requirements.txt).

## Environment Installation

### 1. Clone the repository

```bash
git clone https://github.com/HY-Chien/Thesis_Research.git
cd Thesis_Research/main_methodology
```

Run all remaining experiment commands from `main_methodology/` unless a
section says otherwise.

### 2. Check the CUDA environment

The GPU workflow was tested with CUDA 12.8:

```bash
nvidia-smi
```

The E1–E7 offline reproduction does not require CUDA.

### 3. Create the Python environment

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Verify the installation:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m compileall -q seek scripts
```

### 4. Request access to the base model

The training and live-inference experiments use the gated
[`meta-llama/Meta-Llama-3-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
weights. Accept Meta's license on Hugging Face, then authenticate:

```bash
hf auth login
```

The offline E1–E7 reproduction and the public SEEK training dataset do not
require access to the gated model.

### 5. Check Elasticsearch for open-domain retrieval

The thesis uses Elasticsearch 7.10.2 with a `wiki` index built from the
approximately 21M-passage `psgs_w100` corpus:

```bash
curl -s http://localhost:9200/_cluster/health
curl -s http://localhost:9200/_cat/indices/wiki?v

export SEEK_ES_HOST=localhost
export SEEK_ES_PORT=9200
```

Elasticsearch is required only for the open-domain E8–E11 and E14 runs.

## Dataset

### Decomposer training dataset

The exact 12K training set is public at
[`Chien1011/SEEK-decomposer-12k`](https://huggingface.co/datasets/Chien1011/SEEK-decomposer-12k).
It contains 4,000 MuSiQue, 4,000 2Wiki, and 4,000 PopQA examples.

Download the verified revision:

```bash
SEEK_DATA_DIR=/tmp/SEEK-decomposer-12k
hf download Chien1011/SEEK-decomposer-12k \
  --repo-type dataset \
  --revision 3dcb5c057a82b6610b7b303855e273bb93f9bab7 \
  --local-dir "$SEEK_DATA_DIR"
(cd "$SEEK_DATA_DIR" && sha256sum -c SHA256SUMS)
```

The Hub configurations are `musique`, `twowiki`, and `popqa`.

### Evaluation datasets

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

## Repository Structure

```text
Thesis_Research/
├── main_methodology/
│   ├── seek/                  # SEEK pipeline and retrieval components
│   ├── scripts/               # Experiments, analyses, and figure generators
│   ├── configs/               # Decomposer training configurations
│   ├── datasets/evaluation/   # Four deterministic evaluation manifests
│   └── results/traces/        # Inputs for the offline reproduction
├── Thesis/                    # Thesis LaTeX, PDFs, and figures
├── papers/seek/               # Paper draft source and PDF
├── deeprag_compare/           # Historical DeepRAG comparison
├── experiment/                # Exploratory experiments
├── .gitignore
└── README.md
```

## Models, Inputs, and Outputs

| Input | Location |
|---|---|
| Base model | `meta-llama/Meta-Llama-3-8B-Instruct` |
| SEEK adapter | PEFT adapter directory |
| 12K training set | [Hugging Face: 4K MuSiQue + 4K 2Wiki + 4K PopQA](https://huggingface.co/datasets/Chien1011/SEEK-decomposer-12k) |
| Raw/paragraph data | `datasets/{dataset}/raw/` |
| Wikipedia index | Elasticsearch `wiki` |
| Bundled traces | `results/traces/` |

| Output | Location |
|---|---|
| Generated results | `results/reproduced/`, `results/generated/` |
| PDFs | `../Thesis/main*.pdf`, `../papers/seek/iclr2027/main.pdf` |

## Training

The decomposer is a LoRA fine-tune of
`meta-llama/Meta-Llama-3-8B-Instruct` on the public 12K dataset. The exact
configuration is
[`main_methodology/configs/llama3_8b_decomposer_lora_3ds.yaml`](main_methodology/configs/llama3_8b_decomposer_lora_3ds.yaml).
It trains for three epochs with LoRA rank 16, LoRA alpha 32, learning rate
`1e-4`, BF16, per-device batch size 4, and gradient accumulation 4.

The trained adapter is written to:

```text
LLaMA-Factory/saves/seek/llama3-8b-decomposer-musique-2wiki-popqa/
```

The complete LLaMA-Factory installation and training command appears under
E13 below.

## Benchmark Evaluation

Baselines from earlier papers are excluded. E1–E7 run on CPU from the bundled
`results/traces/`; E8–E14 list their additional requirements.

The **41.88 EM / 48.90 F1** result is a fixed-chain replay. E14 is live
sequential inference, so a fresh E14 run is not expected to be bit-identical.

### E1–E7: offline reproduction

This is the recommended starting point for a reviewer checking whether the
code runs. Every input below is committed under `results/traces/`; these checks
need only a CPU and do not download the Hugging Face training set, load Llama,
or contact Elasticsearch. Run all commands from `main_methodology/`.

| ID | Thesis content | Input | Output |
|---|---|---|---|
| E1 | Ch. 4 main row; Ch. 5 retrieval necessity and threshold sweep | 18 bundled fixed-chain step, direct-confidence, and synthesis traces under `results/traces/` | `results/reproduced/tlsum_fixA_step_gate_final.{md,json}` |
| E2 | Ch. 5 skipped-retrieval safety | Same fixed-chain traces as E1 | Markdown, JSON, and case JSONL under `results/reproduced/` |
| E3 | Appendix failure taxonomy | E1 inputs plus six bundled original/decomposed-query coverage traces | `results/reproduced/decomp_failure_taxonomy_tau0p002305.{md,json}` |
| E4 | Ch. 5 confidence-signal AUROC | Four `results/traces/{dataset}/conf_battery_{dataset}.jsonl` files | `results/reproduced/confidence_battery_analysis.md` |
| E5 | Ch. 5 entropy/selective accuracy; Appendix confidently-wrong cases | Same four `conf_battery` traces as E4 | `results/reproduced/tlsum_confidence_story.md` |
| E6 | Ch. 5 Tail Leakage variants and bootstrap | Same four `conf_battery` traces as E4 | `results/reproduced/tail_leakage_variants.md` |
| E7 | Ch. 5 routing/oracle and Appendix confidence quadrants | E1 fixed-chain traces plus the four direct-confidence traces | Three Markdown reports and three PNGs under `results/reproduced/` |

For a minimal pass/fail check, run E1. For the complete offline smoke test, run
E1 through E7:

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

| ID | Thesis content | Input | Additional resources/status |
|---|---|---|---|
| E8 | Answer-containing Hit@5 | `datasets/{hotpotqa,2wiki,musique}/raw/dev_first1000.jsonl`, `datasets/popqa/raw/dev.jsonl`, and bundled final traces | Elasticsearch index `wiki`; CPU. Figure-only reproduction can use the bundled report without raw data or ES. |
| E9 | MuSiQue decomposition recall | `datasets/musique/raw/dev_first1000.jsonl`, the bundled MuSiQue final trace, and generated base-decomposition JSONL | Base model, GPU, and Elasticsearch index `wiki` |
| E10 | Title-policy comparison | Bundled fixed-chain source and confidence traces under `results/traces/` | Base model, LoRA adapter, GPU, and Elasticsearch index `wiki`; final table aggregation is historical |
| E11 | Maximum entropy vs. Tail Leakage | `datasets/evaluation/{hotpotqa,2wikimultihopqa,musique,popqa}.jsonl` | Base model, adapter, GPU, and Elasticsearch index `wiki`; historical max-entropy traces are not bundled |
| E12 | Repeated-sampling errors | Paragraph-bearing raw files: `datasets/{hotpotqa,2wiki,musique}/raw/dev_first1000.jsonl` and `datasets/popqa/raw/dev.jsonl` | Base model, adapter, and GPU; final appendix aggregation is provenance-only |
| E13 | Decomposer training | [`Chien1011/SEEK-decomposer-12k`](https://huggingface.co/datasets/Chien1011/SEEK-decomposer-12k) at revision `3dcb5c057a82b6610b7b303855e273bb93f9bab7`, plus `configs/llama3_8b_decomposer_lora_3ds.yaml` | Gated base model, pinned LLaMA-Factory, and GPU |
| E14 | Live SEEK | `datasets/evaluation/{hotpotqa,2wikimultihopqa,musique,popqa}.jsonl` | Base model, adapter, GPU, and Elasticsearch index `wiki` (or paragraph-bearing JSONL input) |

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

**E13 — decomposer training.** The exact 12K data is hosted at
[`Chien1011/SEEK-decomposer-12k`](https://huggingface.co/datasets/Chien1011/SEEK-decomposer-12k).
Use the pinned LLaMA-Factory checkout:

```bash
cd ../..
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
git checkout 7e20db5735443861fad91766614b3415ff8966a4
python -m pip install -e .

SEEK_DATA_DIR="${SEEK_DATA_DIR:-/tmp/SEEK-decomposer-12k}"
mkdir -p data/seek_decomposer_12k
cp -R "$SEEK_DATA_DIR/data/." data/seek_decomposer_12k/
```

Merge these entries into `data/dataset_info.json`:

```json
{
  "musique_decomp": {
    "file_name": "seek_decomposer_12k/musique/train.json"
  },
  "twowiki_decomp": {
    "file_name": "seek_decomposer_12k/twowiki/train.json"
  },
  "popqa_decomp": {
    "file_name": "seek_decomposer_12k/popqa/train.json"
  }
}
```

Then train from the LLaMA-Factory checkout:

```bash
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

## Analysis and Thesis Figures

All figures used by the thesis are in `Thesis/figures/`; captions are defined
in the corresponding LaTeX chapters. The commands below rebuild the
script-generated figures.

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

## Thesis and Paper

- Thesis source: [`Thesis/main.tex`](Thesis/main.tex)
- Thesis PDF: [`Thesis/main.pdf`](Thesis/main.pdf)
- Paper source: [`papers/seek/iclr2027/main.tex`](papers/seek/iclr2027/main.tex)
- Paper PDF: [`papers/seek/iclr2027/main.pdf`](papers/seek/iclr2027/main.pdf)

Build the thesis and marked-difference thesis:

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

## Reproducibility Limits

- Gated Llama weights, the final adapter, and the 21M Wikipedia index are not
  in Git. The exact 12K training set is released on Hugging Face.
- E8, E9, and E12 need additional raw data. Historical aggregation artifacts
  for E10–E12 are not bundled.
