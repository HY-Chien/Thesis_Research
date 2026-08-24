#!/usr/bin/env bash
set -euo pipefail

TITLE_MODE="${1:?usage: $0 off|global|pathaware GPU_ID}"
GPU_ID="${2:?usage: $0 off|global|pathaware GPU_ID}"

if [[ "${TITLE_MODE}" != "off" && "${TITLE_MODE}" != "global" && "${TITLE_MODE}" != "pathaware" ]]; then
    echo "TITLE_MODE must be off, global, or pathaware" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUNNER="${PROJECT_DIR}/scripts/rerun_tlsum_fixA.py"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/results/generated/title_policy}"
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/tlsum_tau002305_title${TITLE_MODE}.log"

for DATASET in hotpotqa 2wiki musique popqa; do
    OUTPUT_FILE="${OUTPUT_DIR}/tlsum_tau002305_title${TITLE_MODE}_${DATASET}.jsonl"
    {
        date
        echo "starting title_mode=${TITLE_MODE} dataset=${DATASET} gpu=${GPU_ID}"
    } >> "${LOG_FILE}"
    TRANSFORMERS_VERBOSITY=error CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        "${PYTHON_BIN}" "${RUNNER}" \
        --ds "${DATASET}" \
        --out "${OUTPUT_FILE}" \
        --base "${BASE_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}" \
        --adapter "${SEEK_ADAPTER:-${PROJECT_DIR}/checkpoints/llama3-8b-decomposer-musique-2wiki-popqa}" \
        --tau 0.002305 \
        --title-mode "${TITLE_MODE}" \
        --resume >> "${LOG_FILE}" 2>&1
done

date >> "${LOG_FILE}"
echo "completed title_mode=${TITLE_MODE}" >> "${LOG_FILE}"
