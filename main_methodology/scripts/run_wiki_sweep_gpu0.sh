#!/usr/bin/env bash
# Overnight hyperparameter sweep on the HONEST DeepRAG pool: shared ES `wiki`
# (psgs_w100, 21M passages). Everything else = canonical 12K run (Mode C,
# max_ent gate tau=0.3). Two levers, pure hyperparameter (no code change):
#   (1) top_k  3->5->8->10   — recall is the bottleneck on 21M
#   (2) single-step          — skip_decompose_tau=1e9: never decompose, retrieve
#                              with the ORIGINAL question (tests "decompose hurts
#                              HotpotQA on open domain", recall_ab_m3 finding #3)
# Runs sequentially on ONE GPU. Each config reloads the model (~1.5min) so a
# crash in one config does not kill the rest. Results -> results/{ds}_1000_<suffix>.jsonl
# Live summary  -> scripts/wiki_sweep_summary.txt
set -uo pipefail
source /home/S113065528/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd /home/S113065528/Research/main_methodology
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPU="${1:-0}"
ADAPTER="checkpoints/llama3-8b-decomposer-musique-2wiki-popqa"
BASE="--adapter $ADAPTER --mode C --signal max_ent --tau 0.3 --corpus wiki"
SUMMARY="scripts/wiki_sweep_summary.txt"

declare -A DEV=(
  [hotpotqa]="datasets/hotpotqa/raw/dev_first1000.jsonl"
  [2wiki]="datasets/2wiki/raw/dev_first1000.jsonl"
  [musique]="datasets/musique/raw/dev_first1000.jsonl"
)

# config rows: "ds  top_k  skip_tau(-=decompose)  suffix"
# Ordered worst-dataset + biggest-lever first, so early results are the useful ones.
CONFIGS=(
  "hotpotqa 8  -    wiki_k8"
  "hotpotqa 8  1e9  wiki_single_k8"
  "2wiki    8  -    wiki_k8"
  "hotpotqa 5  -    wiki_k5"
  "2wiki    5  -    wiki_k5"
  "musique  8  -    wiki_k8"
  "musique  5  -    wiki_k5"
  "hotpotqa 10 -    wiki_k10"
)

echo "=== wiki 21M sweep START $(date) on GPU$GPU ===" | tee -a "$SUMMARY"
echo "baseline ref (k=3): 2wiki 38.7 | hotpotqa 25.1 | musique 20.3  EM" | tee -a "$SUMMARY"

for row in "${CONFIGS[@]}"; do
  read -r ds k skip suffix <<< "$row"
  dev="${DEV[$ds]}"
  flags="$BASE --top-k $k --dev $dev"
  [ "$skip" != "-" ] && flags="$flags --skip-decompose-tau $skip"
  out="results/${ds}_1000_${suffix}.jsonl"
  log="scripts/${suffix}_${ds}.log"
  echo "[GPU$GPU] START $ds  $suffix  k=$k skip=$skip  $(date +%T)" | tee -a "$SUMMARY"
  CUDA_VISIBLE_DEVICES=$GPU python scripts/run_seer.py $flags --out "$out" > "$log" 2>&1 || \
    echo "  !! FAILED $ds $suffix (see $log)" | tee -a "$SUMMARY"
  res=$(grep -E '^EM:|^F1:|^skipped:' "$log" 2>/dev/null | tr '\n' ' ')
  echo "[GPU$GPU] DONE  $ds  $suffix  $(date +%T)  ->  $res" | tee -a "$SUMMARY"
done
echo "=== wiki 21M sweep ALL DONE $(date) ===" | tee -a "$SUMMARY"
