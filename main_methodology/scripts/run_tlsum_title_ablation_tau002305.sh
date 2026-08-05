#!/usr/bin/env bash
set -euo pipefail

TITLE_MODE="${1:?usage: $0 off|global GPU_ID}"
GPU_ID="${2:?usage: $0 off|global GPU_ID}"

if [[ "${TITLE_MODE}" != "off" && "${TITLE_MODE}" != "global" ]]; then
    echo "TITLE_MODE must be off or global" >&2
    exit 2
fi

PROJECT_DIR="/home/S113065528/Research/main_methodology"
PYTHON_BIN="/home/S113065528/miniconda3/envs/thesis/bin/python"
RUNNER="${PROJECT_DIR}/scripts/rerun_tlsum_fixA.py"
LOG_FILE="${PROJECT_DIR}/results/tlsum_tau002305_title${TITLE_MODE}.log"

for DATASET in hotpotqa 2wiki musique popqa; do
    OUTPUT_FILE="${PROJECT_DIR}/results/tlsum_tau002305_title${TITLE_MODE}_${DATASET}.jsonl"
    {
        date
        echo "starting title_mode=${TITLE_MODE} dataset=${DATASET} gpu=${GPU_ID}"
    } >> "${LOG_FILE}"
    TRANSFORMERS_VERBOSITY=error CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        "${PYTHON_BIN}" "${RUNNER}" \
        --ds "${DATASET}" \
        --out "${OUTPUT_FILE}" \
        --tau 0.002305 \
        --title-mode "${TITLE_MODE}" \
        --resume >> "${LOG_FILE}" 2>&1
done

date >> "${LOG_FILE}"
echo "completed title_mode=${TITLE_MODE}" >> "${LOG_FILE}"
