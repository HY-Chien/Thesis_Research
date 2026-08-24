# SEEK Reproduction Guide

SEEK is a dual-path adaptive retrieval system for knowledge-intensive question
answering. It keeps both a direct answer and a decomposed reasoning path,
retrieves evidence only for uncertain sub-questions, and uses a deterministic
verifier to select the final answer.

The confidence signal is **Tail Leakage**. For every generated answer token,
SEEK measures the probability mass outside the five leading next-token
candidates. The answer-level score is

```text
Tail Leakage = sum_t -log(sum of the top-5 next-token probabilities at t)
```

A larger score indicates greater uncertainty. The active implementation accepts
`tail_leakage` as the public CLI name; `tl_sum` remains an equivalent legacy
alias for old experiment scripts.

## Reproduction status

| Task | Works after a normal clone? | Additional resources |
|---|---:|---|
| Import the code and inspect every CLI | Yes | None |
| Reproduce the reported 41.88 EM / 48.90 F1 | Yes | Bundled canonical artifacts; CPU only |
| Re-run fixed-chain synthesis | No | Llama-3-8B-Instruct and a CUDA GPU |
| Run SEEK live on a JSONL dataset | No | Base model, LoRA adapter, dataset, and optionally an ES index |
| Train the decomposer | No | Training data and the pinned LLaMA-Factory checkout |

Large raw datasets, the base model, the LoRA weights, and the 21M-passage
Wikipedia index are deliberately not committed to Git. The approximately
52 MB artifact bundle needed for the official offline evaluation and revised
thesis diagnostics **is** committed.

## 1. Clone and install

Clone the default `main` branch:

```bash
git clone https://github.com/HY-Chien/Thesis_Research.git
cd Thesis_Research/main_methodology

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` contains the portable direct dependencies. It does not
contain author-specific paths or CUDA packages. For GPU inference, install the
PyTorch build matching the host CUDA driver before installing the requirements;
the compatible existing PyTorch installation will be retained.

Confirm that the active entry points load:

```bash
python scripts/run_seek.py --help
python scripts/run_seek_dual.py --help
python scripts/analyze_tlsum_fixA_step_gate_final.py --help
```

The files `requirements_thesis.txt` and `environment_thesis.yml` are
provenance snapshots of the original workstation. They are not the recommended
installation path because they include machine-specific transitive packages.

The root [`README.md`](../README.md) records the exact author-workstation
hardware and verified package versions. The one-to-one mapping from every
current thesis experiment and figure to its command, inputs, outputs, and
resource status is in [`THESIS_REPRODUCTION.md`](THESIS_REPRODUCTION.md).

The root [`README.md`](../README.md) records the exact author-workstation
hardware and verified package versions. The one-to-one mapping from every
current thesis experiment and figure to its command, inputs, outputs, and
resource status is in [`THESIS_REPRODUCTION.md`](THESIS_REPRODUCTION.md).

## 2. Reproduce the thesis headline without a GPU

Run:

```bash
python scripts/analyze_tlsum_fixA_step_gate_final.py
```

The command reads the bundled trace and synthesis shards under
`results/traces/`, applies the all-Tail-Leakage verifier, and writes a report
and machine-readable JSON under `results/reproduced/`.

The accuracy-first row `tau=0.002305` should be:

| Dataset | EM / F1 |
|---|---:|
| HotpotQA | 37.80 / 49.03 |
| 2WikiMultihopQA | 49.30 / 56.22 |
| MuSiQue | 26.50 / 34.63 |
| PopQA | 53.90 / 55.73 |
| **Macro average** | **41.88 / 48.90** |

Its pooled step-retrieval rate is 93.84%. This is a controlled fixed-chain
evaluation: each threshold chooses between already stored no-retrieval and
retrieved step answers, then uses the stored base-model synthesis. Later
resolved sub-questions remain fixed. It is not presented as an independent
live rerun at every threshold.

### Bundled inputs

For each of `hotpotqa`, `2wiki`, `musique`, and `popqa`, Git contains:

- the canonical direct/decomposition trace;
- the direct-answer confidence battery;
- two `tlsum_fixA` step-answer shards.

Two deduplicated fixed-chain synthesis shards are stored in
`results/traces/other/`. Six compact coverage traces for HotpotQA, 2Wiki, and
MuSiQue support the revised failure-taxonomy appendix. Other historical traces
remain excluded because the complete local experiment directory is
substantially larger.

### Revised-thesis diagnostic analyses

The skipped-retrieval comparison uses the same fixed-chain replay and final
verifier as the headline result:

```bash
python scripts/analyze_skipped_retrieval_questions.py
```

Its pooled check finds 478 skipped steps out of 7,761 (6.16%). Among the 418
affected questions, the paired final-EM quadrants are 211 both correct, 16
SEEK-only, 16 always-retrieve-only, and 175 both wrong.

The decomposition-path failure taxonomy additionally joins the bundled
coverage@10 traces:

```bash
python scripts/analyze_accuracy_first_failure_taxonomy.py
```

Both commands write regenerated Markdown/JSON artifacts under
`results/reproduced/`.

The bundled direct-confidence traces also support the revised confidence
tables without a GPU:

```bash
python scripts/analyze_confidence_battery.py
python scripts/analyze_tail_leakage_variants.py
python scripts/analyze_tlsum_confidence_story.py
```

The path-overlap, routing/oracle, and quadrant-confidence analyses are rebuilt
at the same accuracy-first operating point with:

```bash
python scripts/make_direct_decompose_2x2_fig.py
python scripts/make_routing_gap_fig.py
python scripts/analyze_quadrant_confidence.py
```

These commands default to `results/reproduced/`; see
`THESIS_REPRODUCTION.md` for explicit commands that update the thesis figures.

The bundled direct-confidence traces also support the revised confidence
tables without a GPU:

```bash
python scripts/analyze_confidence_battery.py
python scripts/analyze_tail_leakage_variants.py
python scripts/analyze_tlsum_confidence_story.py
```

The path-overlap, routing/oracle, and quadrant-confidence analyses are rebuilt
at the same accuracy-first operating point with:

```bash
python scripts/make_direct_decompose_2x2_fig.py
python scripts/make_routing_gap_fig.py
python scripts/analyze_quadrant_confidence.py
```

These commands default to `results/reproduced/`; see
`THESIS_REPRODUCTION.md` for explicit commands that update the thesis figures.

## 3. Live end-to-end inference

### 3.1 Required assets

| Asset | Expected form |
|---|---|
| Base model | Hugging Face ID `meta-llama/Meta-Llama-3-8B-Instruct` or a local path |
| SEEK decomposer | PEFT/LoRA adapter directory containing `adapter_config.json` and `adapter_model.safetensors` |
| Evaluation data | One JSON object per line; schema below |
| Retrieval source | Per-question `paragraphs`, or an Elasticsearch index such as `wiki` |

The Meta Llama model is gated. Accept its license and authenticate before the
first download:

```bash
hf auth login
```

The LoRA weights are larger than GitHub's normal file limit and are not part of
the clone. Place the author-provided adapter at any path and pass that path to
`--adapter`, or train it from the supplied configuration in Section 5.

### 3.2 Input JSONL

For per-question retrieval, each row has this minimal form:

```json
{
  "id": "example-1",
  "question": "Which city is the birthplace of the author of Example Book?",
  "answer": "Example City",
  "answer_aliases": [],
  "paragraphs": [
    {
      "title": "Example Book",
      "paragraph_text": "Example Book was written by Example Author."
    },
    {
      "title": "Example Author",
      "paragraph_text": "Example Author was born in Example City."
    }
  ]
}
```

When `--corpus wiki` is used, retrieval comes from Elasticsearch and the
`paragraphs` field may be omitted. `question` and `answer` are required for
evaluation; `answer_aliases` is optional.

### 3.3 Run with per-question paragraphs

This path does not require Elasticsearch:

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
  --out results/generated/example.jsonl
```

Remove `--limit 10` for the full dataset.

### 3.4 Run with the full Wikipedia corpus

The thesis uses Elasticsearch 7.10.2 with the `wiki` index built from
`psgs_w100` (approximately 21 million passages). SEEK now contains its own
small Elasticsearch query client; another Python source checkout is not
required at runtime.

Start or obtain a compatible index, then verify it:

```bash
curl -s http://localhost:9200/_cluster/health
curl -s http://localhost:9200/_cat/indices/wiki?v
```

Run the same command with:

```bash
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
  --out results/generated/seek_predictions.jsonl
```

The model can also be configured with `SEEK_BASE_MODEL`; Elasticsearch uses
`SEEK_ES_HOST` and `SEEK_ES_PORT` when the corresponding CLI options are
omitted.

### Live run versus reported fixed-chain score

`run_seek_dual.py` is the active live implementation: it generates and
resolves each chain in sequence. The reported 41.88/48.90 threshold comparison
uses the controlled fixed-chain protocol described in Section 2. Because an
earlier answer can change a later resolved query in a live run, the two
protocols should not be described as identical reruns.

## 4. Re-run fixed-chain synthesis

The clone already contains the two synthesis shards, so this section is only
needed to regenerate them. It requires the base model and GPU but not the LoRA
adapter:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/replay_tlsum_fixA_step_gate_synthesis.py \
  --base meta-llama/Meta-Llama-3-8B-Instruct \
  --shard 0 --num-shards 2 \
  --out results/generated/tlsum_fixA_step_gate_synth_shard0.jsonl

CUDA_VISIBLE_DEVICES=1 python scripts/replay_tlsum_fixA_step_gate_synthesis.py \
  --base meta-llama/Meta-Llama-3-8B-Instruct \
  --shard 1 --num-shards 2 \
  --out results/generated/tlsum_fixA_step_gate_synth_shard1.jsonl

python scripts/analyze_tlsum_fixA_step_gate_final.py \
  --synth \
    results/generated/tlsum_fixA_step_gate_synth_shard0.jsonl \
    results/generated/tlsum_fixA_step_gate_synth_shard1.jsonl
```

## 5. Dataset preparation and decomposer training

The raw datasets and generated SFT files are excluded from Git. The preparation
scripts always write relative to this checkout:

```bash
python scripts/prepare_musique.py
python scripts/prepare_2wiki.py
python scripts/prepare_hotpotqa.py
```

PopQA evaluation data follows the DeepRAG-aligned 1,000-question protocol and
must be supplied to `scripts/prep_popqa.py --src ...`. After the held-out
evaluation IDs exist, build the 4,000-example PopQA decomposer set with:

```bash
python scripts/build_popqa_sft.py
```

The final decomposer uses 12,000 examples: 4,000 each from MuSiQue,
2WikiMultihopQA, and PopQA. See `DATASETS.md` for data construction details.

Training uses the LLaMA-Factory revision recorded for the experiment:

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
git checkout 7e20db5735443861fad91766614b3415ff8966a4
python -m pip install -e .

# Register the three generated datasets in LLaMA-Factory's data/dataset_info.json,
# then run from the LLaMA-Factory checkout:
llamafactory-cli train \
  /path/to/Thesis_Research/main_methodology/configs/llama3_8b_decomposer_lora_3ds.yaml
```

The YAML uses a public Hugging Face model ID and a relative output directory.
If the base model is already downloaded locally, replace
`model_name_or_path` with that path.

## 6. Active code and research artifacts

| Path | Role |
|---|---|
| `seek/answerer.py` | Greedy answer generation and confidence signals, including Tail Leakage |
| `seek/pipeline.py` | Decomposition, step gate, retrieval, answer propagation, and synthesis |
| `seek/retriever.py` | Per-question BM25 and open-domain Elasticsearch retrieval |
| `seek/verifier.py` | Deterministic direct-versus-decomposition verifier |
| `scripts/run_seek_dual.py` | Main live dual-path entry point |
| `scripts/rerun_tlsum_fixA.py` | Generate stored no-retrieval/retrieved branches for Tail Leakage replay |
| `scripts/replay_tlsum_fixA_step_gate_synthesis.py` | Regenerate fixed-chain synthesis outputs |
| `scripts/analyze_tlsum_fixA_step_gate_final.py` | Reproduce final metrics and threshold sensitivity |
| `scripts/analyze_skipped_retrieval_questions.py` | Compare SEEK with always retrieve on affected questions |
| `scripts/analyze_accuracy_first_failure_taxonomy.py` | Recompute the appendix decomposition failure taxonomy |
| `scripts/accuracy_first_replay.py` | Shared row-level reconstruction for current path/routing figures |
| `THESIS_REPRODUCTION.md` | Complete thesis experiment/figure command and asset matrix |
| `scripts/accuracy_first_replay.py` | Shared row-level reconstruction for current path/routing figures |
| `THESIS_REPRODUCTION.md` | Complete thesis experiment/figure command and asset matrix |
| `scripts/README.md` | Index of ablation, probe, replay, and plotting scripts |
| `RESULTS_MASTER.md` | Chronological experiment record; older sections retain historical operating points |

Scripts outside the active path are retained because they generated thesis
ablations and diagnostic figures. They are research artifacts rather than
required steps in the quick-start path.

## 7. Resource expectations

- Offline headline reproduction: CPU and approximately 8 GB RAM are sufficient.
- Live Llama-3-8B inference: one CUDA GPU with at least 24 GB VRAM is
  recommended for the current bf16 implementation.
- Decomposer training: the original run used LoRA with batch size 4 and
  gradient accumulation 4.
- Full-corpus retrieval: Elasticsearch plus the external 21M-passage index;
  the index is tens of gigabytes and is not distributed through Git.

When reporting results, distinguish the full `wiki` protocol from
per-question distractor paragraphs. They are different retrieval settings and
their numbers should not be compared as if they used the same corpus.
