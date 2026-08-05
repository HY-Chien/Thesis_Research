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

This reproduction is CPU/offline. A live end-to-end run additionally requires
the Llama-3 base model, the SEEK LoRA decomposer, evaluation JSONL files, and
either per-question paragraphs or a compatible Elasticsearch Wikipedia index.

See [the full reproduction guide](main_methodology/README.md) for the exact
input schema, live inference command, external assets, dataset preparation,
and the distinction between live inference and the paper's controlled
fixed-chain evaluation.

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
