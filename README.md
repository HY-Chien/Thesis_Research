# SEEK: Self-certainty Evaluated Evidence-gated Knowledge-retrieval

SEEK is an adaptive retrieval-augmented generation system for
knowledge-intensive question answering. It generates direct and decomposed
answer paths, uses **Tail Leakage** to decide whether each sub-question needs
retrieval, and applies a deterministic verifier to select the final answer.

This README is the main entry point for reproducing the thesis. For detailed
analysis commands, inputs, outputs, expected values, and figure-generation
steps, follow
[`main_methodology/THESIS_REPRODUCTION.md`](main_methodology/THESIS_REPRODUCTION.md).

## Repository Structure

```text
Thesis_Research/
├── main_methodology/                         # Main SEEK implementation and reproduction package
│   ├── seek/                                 # Core answerer, retrieval, pipeline, and verifier code
│   ├── scripts/                              # Evaluation, reproduction, analysis, and plotting programs
│   ├── tests/                                # Unit tests for data and training helpers
│   ├── configs/                              # Decomposer training configurations
│   ├── datasets/evaluation/                  # Four 1,000-question evaluation JSONL files
│   ├── results/
│   │   ├── traces/                           # Saved predictions and confidence scores used for analysis
│   │   └── reproduced/                       # Reports and figures created by reproduction scripts
│   ├── checkpoints/                          # Training metadata; model weights are not included
│   ├── requirements.txt                      # Python dependencies
│   ├── DATASETS.md                           # Dataset sources and preparation
│   └── THESIS_REPRODUCTION.md                # Detailed analysis and figure commands
├── experiment/                               # Supporting and historical thesis experiments
├── deeprag_compare/                          # DeepRAG comparison code and results
├── Thesis/                                   # Thesis source, PDF, chapters, and figures
│   ├── contents/                             # Thesis chapters
│   ├── figures/                              # Thesis figures
│   ├── front/                                # Abstracts and front matter
│   ├── back/                                 # References and appendices
│   ├── main.tex                              # Main LaTeX source
│   └── main.pdf                              # Compiled thesis
├── papers/seek/                              # SEEK paper drafts
├── .gitignore
└── README.md
```

## Hardware and Software

The author workstation used Ubuntu 24.04.2 LTS, two Intel Xeon Gold 6430 CPUs,
251 GiB RAM, and two NVIDIA L40S GPUs with approximately 48 GiB VRAM each.

| Software | Tested version |
|---|---|
| Python | 3.11.15 |
| CUDA / NVIDIA driver | 12.8 / 575.57.08 |
| PyTorch | 2.10.0+cu128 |
| Transformers / Accelerate / PEFT | 5.6.0 / 1.11.0 / 0.18.1 |
| Datasets | 4.0.0 |
| NumPy / SciPy / Matplotlib | 2.4.6 / 1.17.1 / 3.10.9 |
| Elasticsearch client / server | 7.17.13 / 7.10.2 |

The offline analyses use result files already included in this repository and
need only a CPU and approximately 8 GiB RAM. Live Llama-3-8B BF16 inference
needs at least 24 GiB VRAM. Python 3.11 is required because the pinned
LLaMA-Factory revision requires Python 3.11 or newer; Python 3.8 cannot install
that training stack.

## Quick Start

Run these commands from `main_methodology/`.

| Goal | Command | Input | Output |
|---|---|---|---|
| Fastest reported-result check | `python scripts/analyze_tlsum_fixA_step_gate_final.py` | `results/traces/{hotpotqa,2wiki,musique,popqa,other}/` | `results/reproduced/tlsum_fixA_step_gate_final.{md,json}` |
| Complete offline analysis | `bash scripts/run_offline_reproduction.sh` | `results/traces/` | `results/reproduced/` |
| Train the decomposer | `bash scripts/train_decomposer.sh` | `$SEEK_DATA_DIR/data/{musique,twowiki,popqa}/train.json`<br>`configs/llama3_8b_decomposer_lora_3ds.yaml` | `LLaMA-Factory/saves/seek/llama3-8b-decomposer-musique-2wiki-popqa/` |
| Run the thesis main method | [Use the four-dataset command below](#main-evaluation) | `datasets/evaluation/{hotpotqa,2wikimultihopqa,musique,popqa}.jsonl`<br>Base model, adapter, and Elasticsearch paths are configured below | `results/generated/seek_{dataset}.jsonl` |

The fastest check reproduces **41.88 EM / 48.90 F1** from the saved model
outputs included in this repository. The main evaluation performs fresh
sequential generation and is not expected to be bit-identical.

## Environment Installation

```bash
git clone https://github.com/HY-Chien/Thesis_Research.git
cd Thesis_Research/main_methodology

python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
python -m unittest discover -s tests -v
```

CUDA and Hugging Face login are unnecessary when analyzing the saved result
files. Training and live evaluation use the gated
[`meta-llama/Meta-Llama-3-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
model; accept its license and run `hf auth login` before downloading it.

## Data and Model Files

| Item | Location | Availability |
|---|---|---|
| Saved model outputs used to reproduce thesis tables | `main_methodology/results/traces/` | Included in this repository |
| Four 1,000-question evaluation files | `main_methodology/datasets/evaluation/` | Included in this repository |
| Exact 12K decomposer training set | [Hugging Face](https://huggingface.co/datasets/Chien1011/SEEK-decomposer-12k) | Public |
| Base model | `meta-llama/Meta-Llama-3-8B-Instruct` | Gated license |
| SEEK LoRA adapter | Directory created by decomposer training | Not included; train it below or obtain it from the author |
| Wikipedia search data | Elasticsearch 7.10.2 index named `wiki` | Not included |

Download and verify the exact training data:

```bash
SEEK_DATA_DIR=/tmp/SEEK-decomposer-12k
hf download Chien1011/SEEK-decomposer-12k \
  --repo-type dataset \
  --revision 3dcb5c057a82b6610b7b303855e273bb93f9bab7 \
  --local-dir "$SEEK_DATA_DIR"
(cd "$SEEK_DATA_DIR" && sha256sum -c SHA256SUMS)
```

The dataset contains 4,000 MuSiQue, 4,000 2Wiki, and 4,000 PopQA examples.
Sources, schemas, licenses, and evaluation-subset construction are documented
in [`main_methodology/DATASETS.md`](main_methodology/DATASETS.md).

## Reproduce the Reported Thesis Result

Run from `main_methodology/`:

```bash
python scripts/analyze_tlsum_fixA_step_gate_final.py
```

Expected accuracy-first result at `tau=0.002305`:

| Dataset | EM / F1 |
|---|---:|
| HotpotQA | 37.80 / 49.03 |
| 2Wiki | 49.30 / 56.22 |
| MuSiQue | 26.50 / 34.63 |
| PopQA | 53.90 / 55.73 |
| **Macro** | **41.88 / 48.90** |

The pooled step-retrieval rate must be **93.84%**. Outputs are written to
`results/reproduced/tlsum_fixA_step_gate_final.{md,json}`.

For the complete CPU-only analysis:

```bash
bash scripts/run_offline_reproduction.sh
```

This runs the headline result, skip-safety analysis, failure taxonomy,
confidence analyses, routing analysis, and associated figures. All outputs go
to `results/reproduced/` and do not modify tracked thesis figures.

## Main Evaluation

The thesis main method uses:

- Llama-3-8B-Instruct with the three-source LoRA decomposer;
- Tail Leakage step gate at `tau=0.002305`;
- BM25 top-8 retrieval over the Elasticsearch `wiki` index;
- path-aware title retrieval and original-question evidence union;
- direct/decomposition confidence thresholds `0.0005` and `0.0158`;
- the deterministic type-aware fallback.

Check Elasticsearch:

```bash
curl -s http://localhost:9200/_cluster/health
curl -s http://localhost:9200/_cat/indices/wiki?v
```

Run all four thesis datasets:

```bash
export BASE_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
export SEEK_ADAPTER=/absolute/path/to/seek-lora-adapter
export SEEK_ES_HOST=localhost
export SEEK_ES_PORT=9200

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

For a small no-Elasticsearch smoke test, provide a paragraph-bearing JSONL,
change `--corpus wiki` to `--corpus paragraphs`, and add `--limit 10`.
The required JSONL schema is in
[`main_methodology/README.md`](main_methodology/README.md#32-input-jsonl).

## Train the Decomposer

Training uses the public 12K dataset and the pinned LLaMA-Factory revision
`7e20db5735443861fad91766614b3415ff8966a4`.

The YAML file is a configuration input, not an executable program. Do **not**
run `./configs/llama3_8b_decomposer_lora_3ds.yaml`.

From `main_methodology/`, install the pinned training tool:

```bash
export LLAMA_FACTORY_DIR="$(cd ../.. && pwd)/LLaMA-Factory"

if [ ! -d "$LLAMA_FACTORY_DIR/.git" ]; then
  git clone https://github.com/hiyouga/LLaMA-Factory.git "$LLAMA_FACTORY_DIR"
fi

git -C "$LLAMA_FACTORY_DIR" checkout 7e20db5735443861fad91766614b3415ff8966a4
python -m pip install -e "$LLAMA_FACTORY_DIR"
python -m pip check
```

After downloading the 12K dataset as described above, validate every path and
dataset registration without starting training:

```bash
export SEEK_DATA_DIR=/tmp/SEEK-decomposer-12k
bash scripts/train_decomposer.sh --check-only
```

Optionally run one real training step and write its temporary output under
`/tmp/seek-decomposer-smoke.*`:

```bash
bash scripts/train_decomposer.sh --smoke-test
```

Start training only after the preflight passes:

```bash
bash scripts/train_decomposer.sh
```

The wrapper defaults to `CUDA_VISIBLE_DEVICES=0` so that the effective batch
size remains 16, matching the thesis run. To use another GPU, provide one GPU
ID, for example `CUDA_VISIBLE_DEVICES=1 bash scripts/train_decomposer.sh`.

The wrapper copies and registers all three datasets, then passes this
configuration to `llamafactory-cli train`:

```text
main_methodology/configs/llama3_8b_decomposer_lora_3ds.yaml
```

It specifies LoRA rank 16, alpha 32, dropout 0.05, three epochs, effective
batch size 16, BF16, 1,024-token context, and learning rate `1e-4`. The
adapter is written to:

```text
LLaMA-Factory/saves/seek/llama3-8b-decomposer-musique-2wiki-popqa/
```

The complete dataset-registration and training commands are in the
[detailed reproduction guide](main_methodology/THESIS_REPRODUCTION.md#decomposer-training).

## Analysis and Ablations

The complete offline script covers the main result, threshold sweep,
skipped-retrieval safety, failure taxonomy, confidence analyses, Tail Leakage
variants, routing behavior, oracle headroom, and confidence-quadrant figures.

Additional thesis analyses cover answer-containing retrieval recall, MuSiQue
decomposition recall, title-policy comparison, maximum entropy versus Tail
Leakage, and repeated-sampling errors. These runs need raw development data,
the base model, the SEEK adapter, a GPU, or the Elasticsearch `wiki` index.
Their exact commands, inputs, outputs, and availability notes are in the
[detailed reproduction guide](main_methodology/THESIS_REPRODUCTION.md).

## Thesis and Paper

- Thesis source/PDF: [`Thesis/main.tex`](Thesis/main.tex) /
  [`Thesis/main.pdf`](Thesis/main.pdf)
- Paper source/PDF: [`papers/seek/iclr2027/main.tex`](papers/seek/iclr2027/main.tex) /
  [`papers/seek/iclr2027/main.pdf`](papers/seek/iclr2027/main.pdf)
