#!/usr/bin/env bash
# SEER full-system (dual + direct-context-union) reruns at tau=0.8, filling the
# gap between the existing 0.5 and 1.0 points. Same canonical config as
# run_seer_tau_dual.sh; only --tau differs. Re-runs the analysis afterwards.
set -u
cd /home/S113065528/Research/main_methodology
source /home/S113065528/miniconda3/etc/profile.d/conda.sh
conda activate thesis

ADAPTER=checkpoints/llama3-8b-decomposer-musique-2wiki-popqa
TAU=0.8
TAG=0p8
MASTER=scripts/seer_tau08_master.log
ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a "$MASTER"; }

run_one() { # gpu ds dev extra
  local gpu=$1 ds=$2 dev=$3 extra=$4
  local out="results/${ds}_directctx_tau${TAG}.jsonl"
  local lg="scripts/${ds}_directctx_tau${TAG}.log"
  if grep -q 'Verifier EM' "$lg" 2>/dev/null; then
    log "GPU$gpu SKIP  ${ds} tau=$TAU ($(grep -m1 'Verifier EM' "$lg"))"; return
  fi
  log "GPU$gpu START ${ds} tau=$TAU -> $out"
  CUDA_VISIBLE_DEVICES=$gpu python scripts/run_seek_dual.py \
    --adapter "$ADAPTER" --dev "$dev" --mode C --signal max_ent --tau "$TAU" \
    --top-k 8 --corpus wiki --title-mode pathaware --no-evidence --fallback auto \
    --direct-entropy-tau 0.4 --decomp-post-entropy-tau 0.8 --direct-context-union \
    $extra --out "$out" > "$lg" 2>&1
  log "GPU$gpu DONE  ${ds} tau=$TAU rc=$?  $(grep -m1 'Verifier EM' "$lg") $(grep -m1 'Verifier F1' "$lg")"
}

log "=== SEER tau=0.8 fill START: 4 runs on 2 GPUs ==="
{ run_one 0 hotpotqa datasets/hotpotqa/raw/dev_first1000.jsonl ""
  run_one 0 musique  datasets/musique/raw/dev_first1000.jsonl  ""; } & P0=$!
{ run_one 1 2wiki    datasets/2wiki/raw/dev_first1000.jsonl    ""
  run_one 1 popqa    datasets/popqa/raw/dev.jsonl              "--limit 1000"; } & P1=$!
wait $P0; wait $P1
log "=== tau=0.8 runs done — rebuilding seer_tau_table.md ==="
python scripts/seer_tau_table.py >> "$MASTER" 2>&1
log "=== SEER tau=0.8 fill COMPLETE — results/seer_tau_table.md ==="
