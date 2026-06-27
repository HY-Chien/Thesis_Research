#!/usr/bin/env bash
# Re-run SEEK under DeepRAG's retrieval setting: open-domain BM25 over the shared
# ES `wiki` index (psgs_w100, 21M), top-k=3. Everything else = canonical 12K run
# (Mode C, max_ent gate tau=0.3, 3-dataset decomposer).
set -euo pipefail
source /home/S113065528/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd /home/S113065528/Research/main_methodology

ADAPTER="${ADAPTER:-checkpoints/llama3-8b-decomposer-musique-2wiki-popqa}"
SUFFIX="${SUFFIX:-wiki_top3}"          # output -> results/{ds}_1000_${SUFFIX}.jsonl
COMMON="--adapter $ADAPTER --mode C --signal max_ent --tau 0.3 --top-k 3 --corpus wiki"
GPU=$1; shift
for spec in "$@"; do
  ds="${spec%%:*}"; dev="${spec##*:}"
  echo "[GPU$GPU] START $ds  $(date +%T)"
  CUDA_VISIBLE_DEVICES=$GPU python scripts/run_seek.py $COMMON \
      --dev "$dev" --out "results/${ds}_1000_${SUFFIX}.jsonl" \
      > "scripts/${SUFFIX}_${ds}.log" 2>&1
  echo "[GPU$GPU] DONE  $ds  $(date +%T)  ->  $(grep -E '^EM:|^F1:' scripts/${SUFFIX}_${ds}.log | tr '\n' ' ')"
done
