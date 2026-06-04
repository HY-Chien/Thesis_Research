#!/usr/bin/env bash
# Experiment matrix for options (1) native corpus and (3) post-retrieval fallback.
# Everything else = canonical 12K run (Mode C, max_ent gate tau=0.3, top-k=3).
#   Lane A: native corpus, no fallback           -> results/{ds}_1000_native_top3.jsonl
#   Lane B: native corpus + post-retrieval tau   -> results/{ds}_1000_native_fb15_top3.jsonl
# Usage: run_native_experiments.sh <GPU> <lane:A|B> <ds:corpus:dev> [more specs...]
set -uo pipefail
source /home/S113065528/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd /home/S113065528/Research/main_methodology

ADAPTER="checkpoints/llama3-8b-decomposer-musique-2wiki-popqa"
GPU=$1; shift
LANE=$1; shift
if [ "$LANE" = "B" ]; then
  SUFFIX="native_fb15_top3"; FB="--post-retrieval-tau 1.5"
else
  SUFFIX="native_top3";      FB=""
fi
COMMON="--adapter $ADAPTER --mode C --signal max_ent --tau 0.3 --top-k 3"

for spec in "$@"; do
  ds="${spec%%:*}"; rest="${spec#*:}"; corpus="${rest%%:*}"; dev="${rest#*:}"
  echo "[GPU$GPU lane$LANE] START $ds corpus=$corpus  $(date +%T)"
  CUDA_VISIBLE_DEVICES=$GPU python scripts/run_seer.py $COMMON $FB \
      --corpus "$corpus" --dev "$dev" \
      --out "results/${ds}_1000_${SUFFIX}.jsonl" \
      > "scripts/${SUFFIX}_${ds}.log" 2>&1
  echo "[GPU$GPU lane$LANE] DONE  $ds  $(date +%T)  ->  $(grep -E '^EM:|^F1:|^fallback:' scripts/${SUFFIX}_${ds}.log | tr '\n' ' ')"
done
echo "[GPU$GPU lane$LANE] ALL DONE $(date +%T)"
