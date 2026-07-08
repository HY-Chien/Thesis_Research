#!/usr/bin/env bash
# Dual-path SEEK min_lp sweep.
# Runs min_lp at three thresholds chosen to cover the high-retrieval regime.
set -u

cd /home/S113065528/Research/main_methodology
source /home/S113065528/miniconda3/etc/profile.d/conda.sh
conda activate thesis

ADAPTER=checkpoints/llama3-8b-decomposer-musique-2wiki-popqa
MASTER=scripts/dual_minlp_sweep_master.log
COMMON=(--adapter "$ADAPTER" --mode C --top-k 8 --corpus wiki --title-mode pathaware \
        --no-evidence --fallback auto --direct-context-union --targ-prefix-k 20 --targ-beta 3.0)

# min_lp is lower-is-uncertain: retrieve when min_lp <= tau.
TAUS=(-0.50 -0.20 -0.05)

ts() { date "+%F %T"; }
log() { echo "[$(ts)] $*" | tee -a "$MASTER"; }
tag_tau() { echo "$1" | sed "s/\./p/g"; }

run_one() { # gpu ds dev tau limit_extra
  local gpu=$1 ds=$2 dev=$3 tau=$4 limit_extra=$5
  local tag
  tag=$(tag_tau "$tau")
  local out="results/${ds}_dual_min_lp_tau${tag}.jsonl"
  local lg="scripts/${ds}_dual_min_lp_tau${tag}.log"

  if grep -q "Verifier EM" "$lg" 2>/dev/null; then
    log "GPU$gpu SKIP  $ds signal=min_lp tau=$tau ($(grep -m1 "Verifier EM" "$lg") $(grep -m1 "Verifier F1" "$lg"))"
    return
  fi

  log "GPU$gpu START $ds signal=min_lp tau=$tau verifier_tau=$tau -> $out"
  CUDA_VISIBLE_DEVICES=$gpu TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false \
    python scripts/run_seek_dual.py "${COMMON[@]}" --dev "$dev" --signal min_lp --tau "$tau" \
      --direct-entropy-tau "$tau" --decomp-post-entropy-tau "$tau" $limit_extra --out "$out" > "$lg" 2>&1
  local rc=$?
  log "GPU$gpu DONE  $ds signal=min_lp tau=$tau rc=$rc $(grep -m1 "Verifier EM" "$lg") $(grep -m1 "Verifier F1" "$lg")"
}

run_dataset_grid() { # gpu ds dev limit_extra
  local gpu=$1 ds=$2 dev=$3 limit_extra=$4
  for tau in "${TAUS[@]}"; do
    run_one "$gpu" "$ds" "$dev" "$tau" "$limit_extra"
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

log "=== dual min_lp sweep START: tau={-0.50,-0.20,-0.05}; 12 configs on 2 GPUs ==="
gpu0 & P0=$!
gpu1 & P1=$!
wait "$P0"; RC0=$?
wait "$P1"; RC1=$?
log "=== dual min_lp sweep COMPLETE rc0=$RC0 rc1=$RC1 ==="
