#!/usr/bin/env bash
# M2b: multi-sample direct-answer probe (routing signal only; EM stays M1's greedy answer).
# Same 12K adapter + top-k=5 retrieval as M1/always-skip. N=15 samples, T=0.7.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
ADAPTER="${SEEK_ADAPTER:-checkpoints/llama3-8b-decomposer-musique-2wiki-popqa}"
GPU=$1; shift
for spec in "$@"; do
  ds="${spec%%:*}"; dev="${spec##*:}"
  echo "[GPU$GPU] START $ds  $(date +%T)"
  mkdir -p results/generated/multisample logs/script_runs
  CUDA_VISIBLE_DEVICES=$GPU "${PYTHON_BIN}" scripts/probe_multisample.py \
      --base "${BASE_MODEL}" --adapter "${ADAPTER}" \
      --top-k 5 --n-sample 15 --temperature 0.7 \
      --dev "$dev" --out "results/generated/multisample/${ds}_1000_msample.jsonl" \
      > "logs/script_runs/m2b_${ds}_msample.log" 2>&1
  echo "[GPU$GPU] DONE  $ds  $(date +%T)  -> $(tail -1 "logs/script_runs/m2b_${ds}_msample.log")"
done
