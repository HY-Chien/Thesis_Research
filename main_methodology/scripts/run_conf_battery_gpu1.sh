#!/usr/bin/env bash
set -uo pipefail
source /home/S113065528/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd "$(dirname "$0")/.."
CUDA_VISIBLE_DEVICES=1 python scripts/probe_confidence_battery.py --dev datasets/hotpotqa/raw/dev_first1000.jsonl --trace results/hotpotqa_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl --out results/conf_battery_hotpotqa.jsonl --resume >> scripts/conf_battery_hotpotqa.log 2>&1
CUDA_VISIBLE_DEVICES=1 python scripts/probe_confidence_battery.py --dev datasets/popqa/raw/dev.jsonl --trace results/popqa_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl --out results/conf_battery_popqa.jsonl --resume >> scripts/conf_battery_popqa.log 2>&1
CUDA_VISIBLE_DEVICES=1 python scripts/probe_confidence_battery_decomp.py --trace results/hotpotqa_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl --dev datasets/hotpotqa/raw/dev_first1000.jsonl --out results/conf_battery_decomp_hotpotqa.jsonl --resume >> scripts/conf_battery_decomp_hotpotqa.log 2>&1
CUDA_VISIBLE_DEVICES=1 python scripts/probe_confidence_battery_decomp.py --trace results/popqa_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl --dev datasets/popqa/raw/dev.jsonl --out results/conf_battery_decomp_popqa.jsonl --resume >> scripts/conf_battery_decomp_popqa.log 2>&1
echo "GPU1 ALL DONE" > scripts/conf_battery_gpu1.done
