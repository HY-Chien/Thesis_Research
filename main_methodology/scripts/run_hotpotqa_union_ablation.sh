#!/usr/bin/env bash
set -euo pipefail

# HotpotQA direct-context union ablation.
# Each decomposed sub-question sees its own top-k evidence plus the original
# question top-k evidence. This targets over-decomposition / retrieval drift in
# HotpotQA without overwriting the canonical path-aware result.

cd "$(dirname "$0")/.."

PY=${PY:-python}
GPU=${GPU:-0}
LIMIT_ARG=()
if [[ "${LIMIT:-}" != "" ]]; then
  LIMIT_ARG=(--limit "$LIMIT")
fi

ADAPTER=${ADAPTER:-checkpoints/llama3-8b-decomposer-musique-2wiki-popqa}
DEV=${DEV:-datasets/hotpotqa/raw/dev_first1000.jsonl}
OUT=${OUT:-results/hotpotqa_1000_verify_v2_pathaware_directctx.jsonl}
LOG=${LOG:-scripts/hotpotqa_directctx.log}

CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/run_seek_dual.py \
  --adapter "$ADAPTER" \
  --dev "$DEV" \
  --mode C \
  --signal max_ent \
  --tau 0.3 \
  --top-k 8 \
  --corpus wiki \
  --title-mode pathaware \
  --no-evidence \
  --fallback auto \
  --direct-entropy-tau 0.4 \
  --decomp-post-entropy-tau 0.8 \
  --direct-context-union \
  "${LIMIT_ARG[@]}" \
  --out "$OUT" | tee "$LOG"
