#!/usr/bin/env bash
set -uo pipefail
source /home/S113065528/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd "$(dirname "$0")/.."
CUDA_VISIBLE_DEVICES=0 python scripts/probe_confidence_battery.py --dev datasets/2wiki/raw/dev_first1000.jsonl --trace results/2wiki_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl --out results/conf_battery_2wiki.jsonl --resume >> scripts/conf_battery_2wiki.log 2>&1
CUDA_VISIBLE_DEVICES=0 python scripts/probe_confidence_battery.py --dev datasets/musique/raw/dev_first1000.jsonl --trace results/musique_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl --out results/conf_battery_musique.jsonl --resume >> scripts/conf_battery_musique.log 2>&1
CUDA_VISIBLE_DEVICES=0 python scripts/probe_confidence_battery_decomp.py --trace results/2wiki_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl --dev datasets/2wiki/raw/dev_first1000.jsonl --out results/conf_battery_decomp_2wiki.jsonl --resume >> scripts/conf_battery_decomp_2wiki.log 2>&1
CUDA_VISIBLE_DEVICES=0 python scripts/probe_confidence_battery_decomp.py --trace results/musique_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl --dev datasets/musique/raw/dev_first1000.jsonl --out results/conf_battery_decomp_musique.jsonl --resume >> scripts/conf_battery_decomp_musique.log 2>&1
echo "GPU0 ALL DONE" > scripts/conf_battery_gpu0.done
