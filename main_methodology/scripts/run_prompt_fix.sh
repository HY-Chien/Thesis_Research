#!/bin/bash
# Prompt-fix rerun, one shard per GPU: bash scripts/run_prompt_fix.sh {0|1}
set -u
SHARD=$1
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate thesis
cd /home/S113065528/Research/main_methodology

for DS in hotpotqa 2wiki musique popqa; do
    CUDA_VISIBLE_DEVICES=$SHARD python scripts/rerun_prompt_fix.py \
        --ds $DS --shard $SHARD --num-shards 2 \
        --out results/prompt_fix_${DS}_shard${SHARD}.jsonl --resume \
        >> scripts/prompt_fix_gpu${SHARD}.log 2>&1
done
touch scripts/prompt_fix_gpu${SHARD}.done
