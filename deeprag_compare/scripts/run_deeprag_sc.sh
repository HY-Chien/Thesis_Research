#!/usr/bin/env bash
# Run DeepRAG-SC experiments: learned gate vs SC gate on 4 datasets.
# Uses downloaded DeepRAG-8b checkpoint (LoRA fine-tune of LLaMA-3-8B).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=3 bash run_deeprag_sc.sh              # full 1000Q
#   CUDA_VISIBLE_DEVICES=3 bash run_deeprag_sc.sh 30           # smoke
#   CUDA_VISIBLE_DEVICES=3 bash run_deeprag_sc.sh 1000 sc 12.0 # full, SC gate
#   DATASETS="hotpotqa" CUDA_VISIBLE_DEVICES=3 bash run_deeprag_sc.sh 30
set -euo pipefail

N=${1:-1000}
GATE=${2:-sc}
SC_THRESH=${3:-12.0}
DATASETS=${DATASETS:-"hotpotqa 2wikimultihopqa musique"}

PY=/dataspace/S113065528/envs/research/bin/python
SRC=/home/S113065528/Research/deeprag_compare/src
MODEL_ID=${MODEL_ID:-"xinyan233333/DeepRAG-8b"}
MODEL_TAG=${MODEL_TAG:-"deeprag-8b"}
HF_CACHE=${HF_CACHE:-"/home/S113065528/Adaptive-RAG/hf_cache"}
DEVICE=${DEVICE:-"cuda"}            # "cuda" single-GPU, "auto" for tensor parallel >1 GPU
PROMPT_STYLE=${PROMPT_STYLE:-"auto"}  # auto/deeprag/vanilla
USE_FEWSHOT=${USE_FEWSHOT:-"auto"}    # auto/yes/no
export HF_HUB_OFFLINE=1

echo "=== DeepRAG-SC Experiment ==="
echo "Gate=$GATE  SC_threshold=$SC_THRESH  N=$N"
echo "Datasets: $DATASETS"

for DS in $DATASETS; do
    echo ""
    echo "--- Dataset: $DS ---"
    $PY "$SRC/run_deeprag_sc.py" \
        --dataset "$DS" \
        --gate "$GATE" \
        --sc_threshold "$SC_THRESH" \
        --model_id "$MODEL_ID" \
        --model_tag "$MODEL_TAG" \
        --hf_cache "$HF_CACHE" \
        --device "$DEVICE" \
        --prompt_style "$PROMPT_STYLE" \
        --use_fewshot "$USE_FEWSHOT" \
        --n_samples "$N" \
        --topk 3 \
        --es_index wiki \
        --max_depth 6
    echo ""
done

echo "=== Done ==="
