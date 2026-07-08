#!/usr/bin/env bash
# M2b: multi-sample direct-answer probe (routing signal only; EM stays M1's greedy answer).
# Same 12K adapter + top-k=5 retrieval as M1/always-skip. N=15 samples, T=0.7.
set -euo pipefail
source /home/S113065528/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd /home/S113065528/Research/main_methodology

ADAPTER=checkpoints/llama3-8b-decomposer-musique-2wiki-popqa
COMMON="--adapter $ADAPTER --top-k 5 --n-sample 15 --temperature 0.7"
GPU=$1; shift
for spec in "$@"; do
  ds="${spec%%:*}"; dev="${spec##*:}"
  echo "[GPU$GPU] START $ds  $(date +%T)"
  CUDA_VISIBLE_DEVICES=$GPU python scripts/probe_multisample.py $COMMON \
      --dev "$dev" --out "results/${ds}_1000_msample.jsonl" \
      > "scripts/m2b_${ds}_msample.log" 2>&1
  echo "[GPU$GPU] DONE  $ds  $(date +%T)  -> $(tail -1 scripts/m2b_${ds}_msample.log)"
done
