#!/usr/bin/env bash
# Run the full 3-dataset DeepRAG-comparison experiment.
#
# Requires:
#   * Elasticsearch running on :9200 with `wiki` index (21M psgs_w100 docs)
#   * Meta-Llama-3-8B-Instruct in /home/S113065528/Adaptive-RAG/hf_cache/
#   * conda env `adaptive_rag` (torch / transformers / elasticsearch)
#
# Usage:
#   bash run_all.sh           # full 1000 Q per dataset (takes ~hours)
#   bash run_all.sh 50        # smoke test on first 50 Q per dataset
#
set -euo pipefail

N=${1:-1000}
MODEL=${MODEL_ID:-"meta-llama/Meta-Llama-3-8B-Instruct"}
SRC=/home/S113065528/Research/deeprag_compare/src
PY=/home/S113065528/miniconda3/envs/adaptive_rag/bin/python

for DS in hotpotqa 2wikimultihopqa musique; do
    echo "========== $DS =========="
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    $PY $SRC/run_experiment.py \
        --dataset "$DS" \
        --max_questions "$N" \
        --model_id "$MODEL"
done

echo "========== Summary =========="
$PY /home/S113065528/Research/deeprag_compare/scripts/summarize_vs_deeprag.py
