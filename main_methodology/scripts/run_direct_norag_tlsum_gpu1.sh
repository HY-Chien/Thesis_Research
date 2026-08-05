#!/usr/bin/env bash
set -euo pipefail
cd /home/S113065528/Research/main_methodology
for ds in hotpotqa 2wiki musique popqa; do
  CUDA_VISIBLE_DEVICES=1 /home/S113065528/miniconda3/envs/thesis/bin/python \
    scripts/probe_direct_norag_tlsum.py --ds "$ds" --shard 1 --num-shards 2 \
    --out "results/direct_norag_tlsum_${ds}_shard1.jsonl" --resume
done
