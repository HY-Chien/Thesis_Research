#!/bin/bash
# tl_sum-gated prompt-fix-A rerun, shard 0 on GPU 0: bash scripts/run_tlsum_fixA_gpu0.sh
set -u
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate thesis
cd /home/S113065528/Research/main_methodology

for DS in hotpotqa 2wiki musique popqa; do
    CUDA_VISIBLE_DEVICES=0 python scripts/rerun_tlsum_fixA.py \
        --ds $DS --shard 0 --num-shards 2 \
        --out results/tlsum_fixA_${DS}_shard0.jsonl --resume \
        >> scripts/tlsum_fixA_gpu0.log 2>&1
done
touch scripts/tlsum_fixA_gpu0.done
