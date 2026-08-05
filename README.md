# SEEK: Tail-Leakage-Guided Adaptive Retrieval

SEEK is the research code for a thesis on adaptive retrieval-augmented
generation. It decomposes a question into dependent sub-questions, uses
**Tail Leakage** to decide which steps need retrieval, and conservatively
selects between a direct answer and the decomposed answer.

The repository contains:

- the active implementation in `main_methodology/seek/`;
- executable inference and reproduction scripts in
  `main_methodology/scripts/`;
- a small canonical artifact bundle that reproduces the reported
  **41.88 EM / 48.90 F1** result without a GPU;
- the thesis source in `Thesis/`.

## Quick start

The current review snapshot is on `seek-code-review`:

```bash
git clone --branch seek-code-review --single-branch \
  https://github.com/HY-Chien/Thesis_Research.git
cd Thesis_Research/main_methodology

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the command-line entry point:

```bash
python scripts/run_seek_dual.py --help
```

Reproduce the thesis headline from the bundled traces and fixed-chain
syntheses:

```bash
python scripts/analyze_tlsum_fixA_step_gate_final.py
```

The `tau=0.002305` row should report:

```text
HotpotQA 37.80/49.03
2Wiki   49.30/56.22
MuSiQue 26.50/34.63
PopQA   53.90/55.73
Macro   41.88/48.90
```

The command above is a fast CPU/offline result-integrity check. To execute the
SEEK model itself, follow the GPU path below.

## GPU live inference

The current implementation is tested with Python 3.11 and a CUDA GPU. A GPU
with at least 24 GB of VRAM is recommended for Llama-3-8B in bf16. Install the
PyTorch build matching the host CUDA driver before installing
`requirements.txt`, then confirm that CUDA is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

A live run needs these assets:

| Asset | What to provide |
|---|---|
| Base model | `meta-llama/Meta-Llama-3-8B-Instruct` or a local copy |
| SEEK decomposer | The author-provided PEFT/LoRA adapter directory |
| Evaluation data | JSONL with `id`, `question`, `answer`, and optional aliases |
| Retrieval source | Per-question `paragraphs`, or the external Elasticsearch `wiki` index |

Accept the Meta Llama license, authenticate with Hugging Face, and place the
LoRA adapter on the reviewer machine:

```bash
hf auth login
```

For the simplest GPU smoke run, use a JSONL file containing per-question
paragraphs and execute:

```bash
python scripts/run_seek_dual.py \
  --base meta-llama/Meta-Llama-3-8B-Instruct \
  --adapter /path/to/seek-lora-adapter \
  --dev /path/to/dev.jsonl \
  --mode C \
  --signal tail_leakage \
  --tau 0.002305 \
  --corpus paragraphs \
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

Remove `--limit 10` for a full run. Full-Wikipedia evaluation uses the same
command with `--corpus wiki --es-host localhost --es-port 9200`.

See [the full reproduction guide](main_methodology/README.md) for the exact
JSONL schema, Wikipedia index setup, dataset preparation, decomposer training,
and the distinction between live inference and the paper's controlled
fixed-chain evaluation. The relative link follows whichever Git branch is
currently being viewed.

## Repository map

| Path | Purpose |
|---|---|
| `main_methodology/seek/` | Tail Leakage, decomposition, retrieval, synthesis, and verifier logic |
| `main_methodology/scripts/run_seek_dual.py` | Main live dual-path CLI |
| `main_methodology/scripts/analyze_tlsum_fixA_step_gate_final.py` | Offline headline reproduction |
| `main_methodology/results/` | Bundled canonical artifacts and result reports |
| `main_methodology/configs/` | Portable LLaMA-Factory training configurations |
| `Thesis/` | LaTeX thesis source and compiled PDF |

The root `environment.yml` is an older environment and is not used by SEEK.
Install from `main_methodology/requirements.txt` instead.
