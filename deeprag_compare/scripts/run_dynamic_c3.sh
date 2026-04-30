#!/usr/bin/env bash
# C3: Dynamic SC-triggered RAG across all 4 datasets × 2 trigger modes.
# Loads baselines from existing JSONL (--skip_baselines), only runs dynamic gen.
#
# Outputs (per dataset, per mode):
#   <dataset>_<model-tag>_dyn_continue.{jsonl,_summary.json}
#   <dataset>_<model-tag>_dyn_restart.{jsonl,_summary.json}
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2 bash run_dynamic_c3.sh        # full 1000Q
#   CUDA_VISIBLE_DEVICES=2 bash run_dynamic_c3.sh 30     # smoke
set -euo pipefail

N=${1:-1000}
MODEL=${MODEL_ID:-"/workspace/models/Meta-Llama-3-8B-Instruct"}
SRC=/home/S113065528/Research/deeprag_compare/src
PY=/dataspace/S113065528/envs/research/bin/python
export HF_HUB_OFFLINE=1

# Conservative params (1.5 was too sensitive, fired mid-sentence too early)
DROP_TH=3.0
MAX_RET=2
MIN_SENT=6
DYN_MAX_NEW=512

DATASETS=(hotpotqa 2wikimultihopqa musique popqa)
MODES=(continue restart)

for MODE in "${MODES[@]}"; do
    for DS in "${DATASETS[@]}"; do
        echo "========== $DS  /  on_trigger=$MODE =========="
        $PY $SRC/run_experiment.py \
            --dataset "$DS" \
            --max_questions "$N" \
            --model_id "$MODEL" \
            --run_dynamic \
            --skip_baselines \
            --dyn_on_trigger "$MODE" \
            --dyn_drop_threshold "$DROP_TH" \
            --dyn_max_retrievals "$MAX_RET" \
            --dyn_min_sent_tokens "$MIN_SENT" \
            --dyn_max_new_tokens "$DYN_MAX_NEW" \
            --output_suffix "dyn_${MODE}"
    done
done

echo "========== Summary =========="
$PY /home/S113065528/Research/deeprag_compare/scripts/summarize_vs_deeprag.py
