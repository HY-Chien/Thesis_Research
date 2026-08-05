#!/bin/bash
# tl_sum-gated prompt-fix-A rerun, shard 1 on GPU 1: bash scripts/run_tlsum_fixA_gpu1.sh
set -u
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate thesis
cd /home/S113065528/Research/main_methodology

for DS in hotpotqa 2wiki musique popqa; do
    CUDA_VISIBLE_DEVICES=1 python scripts/rerun_tlsum_fixA.py \
        --ds $DS --shard 1 --num-shards 2 \
        --out results/tlsum_fixA_${DS}_shard1.jsonl --resume \
        >> scripts/tlsum_fixA_gpu1.log 2>&1
done
touch scripts/tlsum_fixA_gpu1.done
