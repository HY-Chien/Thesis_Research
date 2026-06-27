#!/usr/bin/env bash
# 8K decomposer (musique-2wiki) on pool B (global wiki, BM25), identical flags to 12K wiki runs.
# Usage: run_8k_poolB.sh <GPU> <ds:dev> [more...]
set -uo pipefail
source /home/S113065528/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd /home/S113065528/Research/main_methodology

ADAPTER="checkpoints/llama3-8b-decomposer-musique-2wiki"
COMMON="--adapter $ADAPTER --mode C --signal max_ent --tau 0.3 --top-k 3 --corpus wiki"
GPU=$1; shift
for spec in "$@"; do
  ds="${spec%%:*}"; dev="${spec#*:}"
  echo "[GPU$GPU] START $ds $(date +%T)"
  CUDA_VISIBLE_DEVICES=$GPU python scripts/run_seek.py $COMMON \
      --dev "$dev" --out "results/${ds}_1000_wiki_top3_8k.jsonl" \
      > "scripts/wiki_top3_8k_${ds}.log" 2>&1
  echo "[GPU$GPU] DONE $ds $(date +%T) -> $(grep -E '^EM:|^F1:' scripts/wiki_top3_8k_${ds}.log | tr '\n' ' ')"
done
echo "[GPU$GPU] ALL DONE $(date +%T)"
