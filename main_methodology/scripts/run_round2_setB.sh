#!/usr/bin/env bash
# Round-2 set B, FORCED onto GPU0 (GPU1 is held by another user's vLLM, ~46GB).
# Waits for set A (run_round2.sh 0) to finish — detected via the summary file —
# then runs set B sequentially on GPU0 so the two batches never contend.
set -uo pipefail
source /home/S113065528/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd /home/S113065528/Research/main_methodology
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SUMMARY="scripts/round2_summary.txt"
echo "[setB] $(date +%T) waiting for set A (GPU0 ALL DONE) ..." | tee -a "$SUMMARY"
until grep -q "round2 GPU0 ALL DONE" "$SUMMARY" 2>/dev/null; do sleep 60; done
echo "[setB] $(date +%T) set A done -> starting set B on GPU0" | tee -a "$SUMMARY"

ADAPTER="checkpoints/llama3-8b-decomposer-musique-2wiki-popqa"
BASE="--adapter $ADAPTER --mode C --signal max_ent --tau 0.3 --corpus wiki --top-k 8"
declare -A DEV=(
  [2wiki]="datasets/2wiki/raw/dev_first1000.jsonl"
  [musique]="datasets/musique/raw/dev_first1000.jsonl"
  [popqa]="datasets/popqa/raw/dev.jsonl"
)
# rows: "ds  skip_tau(1e9=single-step)  suffix"
CONFIGS=(
  "2wiki   0.5 router05"
  "2wiki   0.8 router08"
  "2wiki   1.2 router12"
  "musique 1e9 single_k8"
  "popqa   0.5 router05"
  "popqa   0.8 router08"
  "popqa   1.2 router12"
)
echo "=== round2 setB(GPU0) START $(date) ===" | tee -a "$SUMMARY"
for row in "${CONFIGS[@]}"; do
  read -r ds skip suffix <<< "$row"
  dev="${DEV[$ds]}"
  out="results/${ds}_1000_${suffix}.jsonl"
  log="scripts/r2_${suffix}_${ds}.log"
  echo "[GPU0-B] START $ds $suffix skip=$skip $(date +%T)" | tee -a "$SUMMARY"
  CUDA_VISIBLE_DEVICES=0 python scripts/run_seer.py $BASE \
      --skip-decompose-tau "$skip" --dev "$dev" --out "$out" > "$log" 2>&1 || \
    echo "  !! FAILED $ds $suffix (see $log)" | tee -a "$SUMMARY"
  res=$(grep -E '^EM:|^F1:|^skipped:' "$log" 2>/dev/null | tr '\n' ' ')
  echo "[GPU0-B] DONE  $ds $suffix $(date +%T) -> $res" | tee -a "$SUMMARY"
done
echo "=== round2 setB(GPU0) ALL DONE $(date) ===" | tee -a "$SUMMARY"
