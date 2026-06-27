#!/usr/bin/env bash
# SEER full-system (dual + direct-context-union) reruns at tau in {0.5, 1.0}.
# Reproduces the canonical SEER (Ours) config exactly (run_seek_dual.py
# --direct-context-union, path-aware, no-evidence, direct_tau=0.4 decomp_tau=0.8)
# and only varies the per-step retrieval gate --tau, to update the main result
# table at lower retrieval rates. tau=0.3 column already exists (the *_directctx
# runs). Type-aware verifier replay is applied afterwards by the analysis step.
set -u
cd /home/S113065528/Research/main_methodology
source /home/S113065528/miniconda3/etc/profile.d/conda.sh
conda activate thesis

ADAPTER=checkpoints/llama3-8b-decomposer-musique-2wiki-popqa
MASTER=scripts/seer_tau_dual_master.log
ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a "$MASTER"; }

run_one() { # gpu ds dev tau extra
  local gpu=$1 ds=$2 dev=$3 tau=$4 extra=$5
  local tag=${tau/./p}
  local out="results/${ds}_directctx_tau${tag}.jsonl"
  local lg="scripts/${ds}_directctx_tau${tag}.log"
  if grep -q 'Verifier EM' "$lg" 2>/dev/null; then
    log "GPU$gpu SKIP  ${ds} tau=$tau ($(grep -m1 'Verifier EM' "$lg"))"; return
  fi
  log "GPU$gpu START ${ds} tau=$tau -> $out"
  CUDA_VISIBLE_DEVICES=$gpu python scripts/run_seek_dual.py \
    --adapter "$ADAPTER" --dev "$dev" --mode C --signal max_ent --tau "$tau" \
    --top-k 8 --corpus wiki --title-mode pathaware --no-evidence --fallback auto \
    --direct-entropy-tau 0.4 --decomp-post-entropy-tau 0.8 --direct-context-union \
    $extra --out "$out" > "$lg" 2>&1
  local rc=$?
  log "GPU$gpu DONE  ${ds} tau=$tau rc=$rc  $(grep -m1 'Verifier EM' "$lg") $(grep -m1 'Verifier F1' "$lg")"
}

# balanced split: each GPU ~160 min (hotpot/2wiki/musique ~45, popqa ~24)
gpu0() {
  run_one 0 hotpotqa datasets/hotpotqa/raw/dev_first1000.jsonl 0.5 ""
  run_one 0 popqa    datasets/popqa/raw/dev.jsonl              0.5 "--limit 1000"
  run_one 0 musique  datasets/musique/raw/dev_first1000.jsonl  1.0 ""
  run_one 0 2wiki    datasets/2wiki/raw/dev_first1000.jsonl    1.0 ""
}
gpu1() {
  run_one 1 2wiki    datasets/2wiki/raw/dev_first1000.jsonl    0.5 ""
  run_one 1 musique  datasets/musique/raw/dev_first1000.jsonl  0.5 ""
  run_one 1 hotpotqa datasets/hotpotqa/raw/dev_first1000.jsonl 1.0 ""
  run_one 1 popqa    datasets/popqa/raw/dev.jsonl              1.0 "--limit 1000"
}

log "=== SEER tau-dual sweep START: 8 runs (4 ds x {0.5,1.0}) on 2 GPUs ==="
gpu0 & P0=$!
gpu1 & P1=$!
wait $P0; wait $P1
log "=== all 8 dual runs done — running analysis (replay + table) ==="
python scripts/seer_tau_table.py >> "$MASTER" 2>&1
log "=== SEER tau-dual sweep COMPLETE — results/seer_tau_table.md ==="
