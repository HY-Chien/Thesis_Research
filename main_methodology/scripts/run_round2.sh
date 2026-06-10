#!/usr/bin/env bash
# Round 2 on the honest wiki-21M pool. k=8, 12K decomposer, mode C, gate tau=0.3.
# Two questions:
#   (A) single-step confirm: 2wiki/musique with skip_decompose_tau=1e9 (always
#       single-step). Expect EM to DROP vs decompose -> proves "real multi-hop
#       needs decomposition" (contrast to HotpotQA where single-step WON).
#   (B) adaptive router: --skip-decompose-tau {0.5,0.8,1.2}. Probe the ORIGINAL
#       question; if max_ent < tau (confident) skip decomposition (single-step),
#       else decompose. Goal: one auto tau approaching the per-dataset oracle
#       upper bound (HotpotQA->single, 2wiki/musique/popqa->decompose).
# Usage: run_round2.sh <GPU 0|1>
set -uo pipefail
source /home/S113065528/miniconda3/etc/profile.d/conda.sh
conda activate thesis
cd /home/S113065528/Research/main_methodology
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPU="${1:?usage: run_round2.sh <GPU>}"
ADAPTER="checkpoints/llama3-8b-decomposer-musique-2wiki-popqa"
BASE="--adapter $ADAPTER --mode C --signal max_ent --tau 0.3 --corpus wiki --top-k 8"
SUMMARY="scripts/round2_summary.txt"

declare -A DEV=(
  [hotpotqa]="datasets/hotpotqa/raw/dev_first1000.jsonl"
  [2wiki]="datasets/2wiki/raw/dev_first1000.jsonl"
  [musique]="datasets/musique/raw/dev_first1000.jsonl"
  [popqa]="datasets/popqa/raw/dev.jsonl"
)

# rows: "ds  skip_tau(1e9=single-step)  suffix"
if [ "$GPU" = "0" ]; then
  CONFIGS=(
    "hotpotqa 0.5 router05"
    "hotpotqa 0.8 router08"
    "hotpotqa 1.2 router12"
    "2wiki    1e9 single_k8"
    "musique  0.5 router05"
    "musique  0.8 router08"
    "musique  1.2 router12"
  )
else
  CONFIGS=(
    "2wiki   0.5 router05"
    "2wiki   0.8 router08"
    "2wiki   1.2 router12"
    "musique 1e9 single_k8"
    "popqa   0.5 router05"
    "popqa   0.8 router08"
    "popqa   1.2 router12"
  )
fi

echo "=== round2 GPU$GPU START $(date) ===" | tee -a "$SUMMARY"
for row in "${CONFIGS[@]}"; do
  read -r ds skip suffix <<< "$row"
  dev="${DEV[$ds]}"
  out="results/${ds}_1000_${suffix}.jsonl"
  log="scripts/r2_${suffix}_${ds}.log"
  echo "[GPU$GPU] START $ds $suffix skip=$skip $(date +%T)" | tee -a "$SUMMARY"
  CUDA_VISIBLE_DEVICES=$GPU python scripts/run_seer.py $BASE \
      --skip-decompose-tau "$skip" --dev "$dev" --out "$out" > "$log" 2>&1 || \
    echo "  !! FAILED $ds $suffix (see $log)" | tee -a "$SUMMARY"
  res=$(grep -E '^EM:|^F1:|^skipped:' "$log" 2>/dev/null | tr '\n' ' ')
  echo "[GPU$GPU] DONE  $ds $suffix $(date +%T) -> $res" | tee -a "$SUMMARY"
done
echo "=== round2 GPU$GPU ALL DONE $(date) ===" | tee -a "$SUMMARY"
