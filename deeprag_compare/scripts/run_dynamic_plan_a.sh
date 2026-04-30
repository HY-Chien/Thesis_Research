#!/usr/bin/env bash
# Plan A: FLARE-style improvements to dynamic SC-RAG.
#   - query_mode = masked  (anchor tokens of triggering sentence, low-SC tokens dropped)
#   - on_trigger = regenerate_sentence  (drop only triggering sentence, keep earlier CoT)
#
# Outputs (per dataset):
#   <dataset>_<model-tag>_dyn_planA.{jsonl,_summary.json}
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2 bash run_dynamic_plan_a.sh        # full 1000Q
#   CUDA_VISIBLE_DEVICES=2 bash run_dynamic_plan_a.sh 30     # smoke
#   DATASETS="hotpotqa" CUDA_VISIBLE_DEVICES=2 bash run_dynamic_plan_a.sh 30
set -euo pipefail

N=${1:-1000}
MODEL=${MODEL_ID:-"/workspace/models/Meta-Llama-3-8B-Instruct"}
DATASETS=${DATASETS:-"hotpotqa 2wikimultihopqa musique popqa"}
SUFFIX=${OUTPUT_SUFFIX:-"dyn_planA"}

SRC=/home/S113065528/Research/deeprag_compare/src
PY=/dataspace/S113065528/envs/research/bin/python
export HF_HUB_OFFLINE=1

# Same trigger conservativeness as run_dynamic_c3.sh (drop=3 was the calibrated value)
DROP_TH=3.0
MAX_RET=2
MIN_SENT=6
DYN_MAX_NEW=512
MASK_PCT=30.0

echo "=== Plan A: masked query + regenerate_sentence ==="
echo "Datasets: $DATASETS  | N=$N  | suffix=$SUFFIX"

for DS in $DATASETS; do
    echo "========== $DS =========="
    $PY "$SRC/run_experiment.py" \
        --dataset "$DS" \
        --max_questions "$N" \
        --model_id "$MODEL" \
        --run_dynamic \
        --skip_baselines \
        --dyn_on_trigger regenerate_sentence \
        --dyn_query_mode masked \
        --dyn_mask_pct "$MASK_PCT" \
        --dyn_drop_threshold "$DROP_TH" \
        --dyn_max_retrievals "$MAX_RET" \
        --dyn_min_sent_tokens "$MIN_SENT" \
        --dyn_max_new_tokens "$DYN_MAX_NEW" \
        --output_suffix "$SUFFIX"
done

echo "========== Done =========="
