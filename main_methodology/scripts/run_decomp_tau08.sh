#!/usr/bin/env bash
# Decompose-only (run_seek.py) gate at tau=0.8 for the 4 datasets, to swap the
# §4.6 "Necessity of Retrieval" figure's gate bar from tau=0.3 to tau=0.8.
# Same config as the 3-way ablation (mode C, max_ent, path-aware, wiki, k8 /
# PopQA k3); never (tau=1e9) and always (tau=0) endpoints are reused.
set -u
cd /home/S113065528/Research/main_methodology
source /home/S113065528/miniconda3/etc/profile.d/conda.sh
conda activate thesis
ADAPTER=checkpoints/llama3-8b-decomposer-musique-2wiki-popqa
PY="python scripts/run_seek.py --adapter $ADAPTER --mode C --signal max_ent --corpus wiki --tau 0.8"
MASTER=scripts/decomp_tau08_master.log
ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a "$MASTER"; }

run_one() { # gpu ds topk dev extra
  local gpu=$1 ds=$2 topk=$3 dev=$4 extra=$5
  local out="results/${ds}_sweep_tau0p8.jsonl" lg="scripts/sweep_${ds}_tau0p8.log"
  if grep -q '^EM:' "$lg" 2>/dev/null; then log "GPU$gpu SKIP $ds ($(grep -m1 '^retrieval:' "$lg"))"; return; fi
  log "GPU$gpu START $ds tau=0.8 k=$topk -> $out"
  CUDA_VISIBLE_DEVICES=$gpu $PY --top-k "$topk" --dev "$dev" $extra --out "$out" > "$lg" 2>&1
  log "GPU$gpu DONE  $ds rc=$?  $(grep -m1 '^EM:' "$lg") $(grep -m1 '^retrieval:' "$lg")"
}

log "=== decompose-only tau=0.8: 4 runs on 2 GPUs ==="
{ run_one 0 hotpotqa 8 datasets/hotpotqa/raw/dev_first1000.jsonl ""
  run_one 0 musique  8 datasets/musique/raw/dev_first1000.jsonl  ""; } & P0=$!
{ run_one 1 2wiki    8 datasets/2wiki/raw/dev_first1000.jsonl    ""
  run_one 1 popqa    3 datasets/popqa/raw/dev.jsonl              "--limit 1000"; } & P1=$!
wait $P0; wait $P1
log "=== decompose-only tau=0.8 COMPLETE ==="
for ds in hotpotqa 2wiki musique popqa; do
  log "$ds: $(grep -m1 '^EM:' scripts/sweep_${ds}_tau0p8.log) $(grep -m1 '^retrieval:' scripts/sweep_${ds}_tau0p8.log)"
done