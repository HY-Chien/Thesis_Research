#!/usr/bin/env bash
# Sweep DeepRAG-SC over multiple thresholds.
# Output filename includes threshold so runs don't overwrite each other:
#   <ds>_deeprag-8b_sc_th<thr>.{jsonl,_summary.json}
#
# Usage:
#   CUDA_VISIBLE_DEVICES=3 bash run_deeprag_sc_sweep.sh                     # default 1000Q × th={17,19,21} × 3ds
#   CUDA_VISIBLE_DEVICES=3 bash run_deeprag_sc_sweep.sh 30                  # smoke
#   THRESHOLDS="15 17 19 21 23" CUDA_VISIBLE_DEVICES=3 bash run_deeprag_sc_sweep.sh 1000
#   DATASETS="hotpotqa" CUDA_VISIBLE_DEVICES=3 bash run_deeprag_sc_sweep.sh 30
set -euo pipefail

N=${1:-1000}
DATASETS=${DATASETS:-"hotpotqa 2wikimultihopqa musique"}
THRESHOLDS=${THRESHOLDS:-"17 19 21"}

PY=/dataspace/S113065528/envs/research/bin/python
SRC=/home/S113065528/Research/deeprag_compare/src
MODEL_ID="xinyan233333/DeepRAG-8b"
MODEL_TAG="deeprag-8b"
HF_CACHE=/home/S113065528/Adaptive-RAG/hf_cache
export HF_HUB_OFFLINE=1

echo "=== DeepRAG-SC threshold sweep ==="
echo "Datasets: $DATASETS"
echo "Thresholds: $THRESHOLDS"
echo "N=$N"

for THR in $THRESHOLDS; do
    for DS in $DATASETS; do
        echo ""
        echo "--- Dataset: $DS  | Threshold: $THR ---"
        $PY "$SRC/run_deeprag_sc.py" \
            --dataset "$DS" \
            --gate sc \
            --sc_threshold "$THR" \
            --model_id "$MODEL_ID" \
            --model_tag "$MODEL_TAG" \
            --hf_cache "$HF_CACHE" \
            --n_samples "$N" \
            --topk 3 \
            --es_index wiki \
            --max_depth 6
    done
done

echo "=== Sweep done ==="
