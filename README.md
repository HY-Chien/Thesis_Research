# SEEK: Self-certainty Evaluated Evidence-gated Knowledge-retrieval

SEEK is an adaptive retrieval-augmented generation system for
knowledge-intensive question answering. It generates direct and decomposed
answer paths, uses **Tail Leakage** to decide whether each sub-question needs
retrieval, and applies a deterministic verifier to select the final answer.

This README is the reviewer entry point. Exact commands for every thesis
experiment and figure are in
[`main_methodology/THESIS_REPRODUCTION.md`](main_methodology/THESIS_REPRODUCTION.md).

## Reviewer Quick Start

| Goal | Command | Requirement |
|---|---|---|
| Fastest code/result check | `python scripts/analyze_tlsum_fixA_step_gate_final.py` | CPU, bundled traces |
| Complete offline check (E1–E7) | `bash scripts/run_offline_reproduction.sh` | CPU, bundled traces |
| Run the thesis main method live (E14) | `python scripts/run_seek_dual.py ...` | GPU, base model, adapter, ES `wiki` |
| Train the decomposer (E13) | `llamafactory-cli train ...` | GPU, base model, public 12K dataset |

The reported **41.88 EM / 48.90 F1** is reproduced by E1 from canonical
fixed-chain traces. E14 is fresh sequential generation and is not expected to
be bit-identical.

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

E1–E7 need only a CPU and approximately 8 GiB RAM. Live Llama-3-8B BF16
inference needs at least 24 GiB VRAM. The released stack has been verified on
Python 3.11.15; Python 3.8 compatibility has not been verified.

## Environment Installation

```bash
git clone https://github.com/HY-Chien/Thesis_Research.git
cd Thesis_Research/main_methodology

python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
```

For E1–E7, CUDA and Hugging Face login are unnecessary. E13 and E14 use the
gated
[`meta-llama/Meta-Llama-3-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
model; accept its license and run `hf auth login` before downloading it.

## Data and Model Assets

| Asset | Location | Availability |
|---|---|---|
| Offline reproduction traces | `main_methodology/results/traces/` | Bundled |
| Four 1,000-question manifests | `main_methodology/datasets/evaluation/` | Bundled |
| Exact 12K decomposer training set | [Hugging Face](https://huggingface.co/datasets/Chien1011/SEEK-decomposer-12k) | Public |
| Base model | `meta-llama/Meta-Llama-3-8B-Instruct` | Gated license |
| SEEK LoRA adapter | PEFT adapter directory | External; train with E13 or obtain from the author |
| Wikipedia corpus | Elasticsearch 7.10.2 index `wiki` | External |

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

## Fast Reproduction of the Thesis Result (E1)

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

For the complete CPU-only E1–E7 check:

```bash
bash scripts/run_offline_reproduction.sh
```

This runs the headline result, skip-safety analysis, failure taxonomy,
confidence analyses, routing analysis, and associated figures. All outputs go
to `results/reproduced/` and do not modify tracked thesis figures.

## Main Method: Live SEEK (E14)

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

## Decomposer Training (E13)

Training uses the public 12K dataset and the pinned LLaMA-Factory revision
`7e20db5735443861fad91766614b3415ff8966a4`. The thesis configuration is:

```text
main_methodology/configs/llama3_8b_decomposer_lora_3ds.yaml
```

It specifies LoRA rank 16, alpha 32, dropout 0.05, three epochs, effective
batch size 16, BF16, 1,024-token context, and learning rate `1e-4`. The
adapter is written to:

```text
LLaMA-Factory/saves/seek/llama3-8b-decomposer-musique-2wiki-popqa/
```

The complete dataset-registration and training commands are in
[`E13 of the reproduction matrix`](main_methodology/THESIS_REPRODUCTION.md#e13-decomposer-training).

## Thesis Experiment Map

| ID | Thesis content | Entry point | Status |
|---|---|---|---|
| E1 | Main result, retrieval necessity, threshold sweep | `analyze_tlsum_fixA_step_gate_final.py` | Verified clone |
| E2 | Skipped-retrieval safety | `analyze_skipped_retrieval_questions.py` | Verified clone |
| E3 | Failure taxonomy | `analyze_accuracy_first_failure_taxonomy.py` | Verified clone |
| E4 | Confidence-signal AUROC | `analyze_confidence_battery.py` | Verified clone |
| E5 | Selective accuracy and confidently-wrong cases | `analyze_tlsum_confidence_story.py` | Verified clone |
| E6 | Tail Leakage variants and bootstrap | `analyze_tail_leakage_variants.py` | Verified clone |
| E7 | Routing, oracle, and confidence quadrants | Three analysis/plot scripts | Verified clone |
| E8 | Answer-containing Hit@5 | `recompute_answer_containing_recall_4ds.py` | Raw data + ES |
| E9 | MuSiQue decomposition recall | Two decomposition/recall scripts | Base model + raw data + ES |
| E10 | Title-policy comparison | `run_tlsum_title_ablation_tau002305.sh` | Base model + adapter + ES |
| E11 | Maximum entropy vs. Tail Leakage | `run_seek_dual.py --signal max_ent` | Historical traces incomplete |
| E12 | Repeated-sampling errors | `run_m2b_multisample.sh` | Historical aggregation incomplete |
| E13 | Decomposer training | LLaMA-Factory configuration | External model + GPU |
| E14 | Live final SEEK | `run_seek_dual.py` | External adapter + ES + GPU |

Inputs, outputs, exact commands, expected values, and thesis figure producers
for every ID are recorded in
[`main_methodology/THESIS_REPRODUCTION.md`](main_methodology/THESIS_REPRODUCTION.md).

## Thesis and Paper

- Thesis source/PDF: [`Thesis/main.tex`](Thesis/main.tex) /
  [`Thesis/main.pdf`](Thesis/main.pdf)
- Paper source/PDF: [`papers/seek/iclr2027/main.tex`](papers/seek/iclr2027/main.tex) /
  [`papers/seek/iclr2027/main.pdf`](papers/seek/iclr2027/main.pdf)

Build the thesis:

```bash
cd ../Thesis
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Detailed Documentation and Limits

- [Implementation and live-input schema](main_methodology/README.md)
- [All experiment and figure commands](main_methodology/THESIS_REPRODUCTION.md)
- [Dataset sources and preparation](main_methodology/DATASETS.md)
- [Script index](main_methodology/scripts/README.md)
- [Result artifact index](main_methodology/results/README.md)

The gated base model, final LoRA adapter, raw paragraph datasets, and 21M
Wikipedia index are not in Git. E10–E12 retain some historical/provenance-only
results; the reproduction matrix labels these cases explicitly rather than
claiming clean-clone reproduction.
