#!/usr/bin/env bash
# Dual-path SEEK margin-signal sweep.
# Runs TARG margin uncertainty plus prefix probability mean/max/first-token (MAQA) margins.
# Designed for long overnight runs; check scripts/dual_margin_sweep_master.log hourly.
set -u
cd /home/S113065528/Research/main_methodology
source /home/S113065528/miniconda3/etc/profile.d/conda.sh
conda activate thesis

ADAPTER=checkpoints/llama3-8b-decomposer-musique-2wiki-popqa
MASTER=scripts/dual_margin_sweep_master.log
COMMON=(--adapter "$ADAPTER" --mode C --top-k 8 --corpus wiki --title-mode pathaware \
        --no-evidence --fallback auto --direct-context-union --targ-prefix-k 20 --targ-beta 3.0)

# Coarse grids (subsets of the original dense grids, so completed dense points
# are reused via the skip check and the grid can be densified later without
# re-running anything). Dense originals:
#   TARG        0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.60 0.70
#   MEAN_MARGIN 0.50 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.93 0.95 0.97 0.99
#   MAX_MARGIN  0.50 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.93 0.95 0.97 0.98 0.985 0.99 0.995 0.998
#   MAQA_MARGIN 0.50 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.93 0.95 0.97 0.98 0.985 0.99 0.995 0.998
TARG_TAUS=(0.10 0.20 0.30 0.45 0.70)
MEAN_MARGIN_TAUS=(0.50 0.70 0.85 0.93 0.97 0.99)
MAX_MARGIN_TAUS=(0.50 0.75 0.90 0.97 0.99 0.998)
# MAQA first-token top1/top2 softmax margin. Pilot calibration (gate_margin_pilot100_summary.md)
# found matching entropy's retrieval rate needs tau ~= 0.972, so keep density near 0.97.
MAQA_MARGIN_TAUS=(0.50 0.75 0.90 0.95 0.97 0.985 0.998)

ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a "$MASTER"; }
tag_tau() { echo "$1" | sed 's/\./p/g'; }

# Verifier threshold defaults. For high-is-uncertain targ_unc, confidence is score < tau.
# For low-is-uncertain probability margins, confidence is score > tau.
verifier_tau() { # signal gate_tau
  local signal=$1 tau=$2
  case "$signal" in
    targ_unc) echo "0.45" ;;
    mean_margin|max_margin|maqa_margin) echo "$tau" ;;
    *) echo "$tau" ;;
  esac
}

run_one() { # gpu ds dev signal tau limit_extra
  local gpu=$1 ds=$2 dev=$3 signal=$4 tau=$5 limit_extra=$6
  local tag=$(tag_tau "$tau")
  local out="results/${ds}_dual_${signal}_tau${tag}.jsonl"
  local lg="scripts/${ds}_dual_${signal}_tau${tag}.log"
  local vt=$(verifier_tau "$signal" "$tau")
  if grep -q 'Verifier EM' "$lg" 2>/dev/null; then
    log "GPU$gpu SKIP  $ds signal=$signal tau=$tau ($(grep -m1 'Verifier EM' "$lg") $(grep -m1 'Verifier F1' "$lg"))"
    return
  fi
  log "GPU$gpu START $ds signal=$signal tau=$tau verifier_tau=$vt -> $out"
  CUDA_VISIBLE_DEVICES=$gpu TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false \
    python scripts/run_seek_dual.py "${COMMON[@]}" --dev "$dev" --signal "$signal" --tau "$tau" \
      --direct-entropy-tau "$vt" --decomp-post-entropy-tau "$vt" $limit_extra --out "$out" > "$lg" 2>&1
  local rc=$?
  log "GPU$gpu DONE  $ds signal=$signal tau=$tau rc=$rc $(grep -m1 'Verifier EM' "$lg") $(grep -m1 'Verifier F1' "$lg")"
}

run_dataset_grid() { # gpu ds dev limit_extra
  local gpu=$1 ds=$2 dev=$3 limit_extra=$4
  # Paper-style raw-logit-gap uncertainty. Broadened around the TARG/pilot range.
  for tau in "${TARG_TAUS[@]}"; do
    run_one "$gpu" "$ds" "$dev" targ_unc "$tau" "$limit_extra"
  done
  # Prefix softmax probability margins. Lower margin = uncertain, so retrieval uses score <= tau.
  for tau in "${MEAN_MARGIN_TAUS[@]}"; do
    run_one "$gpu" "$ds" "$dev" mean_margin "$tau" "$limit_extra"
  done
  # Prefix max margin is expected to be conservative; thresholds are close to 1 by design.
  for tau in "${MAX_MARGIN_TAUS[@]}"; do
    run_one "$gpu" "$ds" "$dev" max_margin "$tau" "$limit_extra"
  done
  # MAQA first-token top1/top2 probability margin. Same low-is-uncertain polarity and
  # near-1 threshold range as max_margin, but only the first generated answer token.
  for tau in "${MAQA_MARGIN_TAUS[@]}"; do
    run_one "$gpu" "$ds" "$dev" maqa_margin "$tau" "$limit_extra"
  done
}

gpu0() {
  run_dataset_grid 0 hotpotqa datasets/hotpotqa/raw/dev_first1000.jsonl ""
  run_dataset_grid 0 popqa    datasets/popqa/raw/dev.jsonl             "--limit 1000"
}

gpu1() {
  run_dataset_grid 1 2wiki    datasets/2wiki/raw/dev_first1000.jsonl   ""
  run_dataset_grid 1 musique  datasets/musique/raw/dev_first1000.jsonl ""
}

log "=== dual margin sweep START: signals={targ_unc,mean_margin,max_margin,maqa_margin}; 55 configs x 4 datasets ==="
gpu0 & P0=$!
gpu1 & P1=$!
wait $P0; RC0=$?
wait $P1; RC1=$?
log "=== dual margin sweep COMPLETE rc0=$RC0 rc1=$RC1 ==="
