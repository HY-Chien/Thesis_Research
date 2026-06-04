#!/usr/bin/env bash
# M1: clean always-skip (no-decompose) baseline = single-step RAG with the 12K adapter.
# Same model/params as the 3ds_terse decompose results, only difference: never decompose.
# skip_decompose_tau=1e9 forces skip on every question.
set -euo pipefail
source /home/S113065528/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd /home/S113065528/Research/main_methodology

ADAPTER=checkpoints/llama3-8b-decomposer-musique-2wiki-popqa   # 12K, matches decompose results
COMMON="--adapter $ADAPTER --mode C --signal max_ent --tau 0.3 --top-k 5 --skip-decompose-tau 1e9"
GPU=$1; shift
for spec in "$@"; do
  ds="${spec%%:*}"; dev="${spec##*:}"
  echo "[GPU$GPU] START $ds  $(date +%T)"
  CUDA_VISIBLE_DEVICES=$GPU python scripts/run_seer.py $COMMON \
      --dev "$dev" --out "results/${ds}_1000_alwaysskip.jsonl" \
      > "scripts/m1_${ds}_alwaysskip.log" 2>&1
  echo "[GPU$GPU] DONE  $ds  $(date +%T)  -> $(grep '^EM:' scripts/m1_${ds}_alwaysskip.log)"
done
