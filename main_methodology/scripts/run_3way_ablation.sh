#!/usr/bin/env bash
# 3-way per-step retrieval-gate ablation on honest wiki-21M, 1000Q x 4 datasets.
#   never-RAG  (tau=1e9)  : decompose, never retrieve  -> tests "拆解後 RAG 有用"
#   gate       (tau=0.3)  : entropy gate (canonical)
#   always-RAG (tau=0)    : retrieve every step        -> tests gate efficiency
# All three run through the SAME current run_seek.py (path-aware title default),
# so the comparison is config-clean. Waits for the user's vLLM job to free both
# GPUs before starting, then runs one model per GPU in parallel.
set -u
cd /home/S113065528/Research/main_methodology
source /home/S113065528/miniconda3/etc/profile.d/conda.sh
conda activate thesis

ADAPTER=checkpoints/llama3-8b-decomposer-musique-2wiki-popqa
MASTER=scripts/3way_master.log
NEED_FREE_MIB=30000          # both GPUs must have >= this free
MAX_WAIT_SEC=$((24*3600))    # give up after 24h
PY="python scripts/run_seek.py --adapter $ADAPTER --mode C --signal max_ent --corpus wiki"

ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a "$MASTER"; }

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n "$1p"; }

log "=== 3-way ablation driver START — waiting for both GPUs to free (>= ${NEED_FREE_MIB} MiB) ==="
waited=0
while :; do
  f0=$(free_mib 1); f1=$(free_mib 2)
  if [ "${f0:-0}" -ge "$NEED_FREE_MIB" ] && [ "${f1:-0}" -ge "$NEED_FREE_MIB" ]; then
    log "GPUs free: GPU0=${f0}MiB GPU1=${f1}MiB — launching."
    break
  fi
  if [ "$waited" -ge "$MAX_WAIT_SEC" ]; then
    log "GAVE UP after ${MAX_WAIT_SEC}s waiting for GPU (GPU0=${f0} GPU1=${f1})."; exit 1
  fi
  sleep 90; waited=$((waited+90))
done

# job spec: cond|tau|ds|topk|dev|extra
JOBS=(
  "never|1e9|hotpotqa|8|datasets/hotpotqa/raw/dev_first1000.jsonl|"
  "gate|0.3|hotpotqa|8|datasets/hotpotqa/raw/dev_first1000.jsonl|"
  "always|0|hotpotqa|8|datasets/hotpotqa/raw/dev_first1000.jsonl|"
  "never|1e9|2wiki|8|datasets/2wiki/raw/dev_first1000.jsonl|"
  "gate|0.3|2wiki|8|datasets/2wiki/raw/dev_first1000.jsonl|"
  "always|0|2wiki|8|datasets/2wiki/raw/dev_first1000.jsonl|"
  "never|1e9|musique|8|datasets/musique/raw/dev_first1000.jsonl|"
  "gate|0.3|musique|8|datasets/musique/raw/dev_first1000.jsonl|"
  "always|0|musique|8|datasets/musique/raw/dev_first1000.jsonl|"
  "never|1e9|popqa|3|datasets/popqa/raw/dev.jsonl|--limit 1000"
  "gate|0.3|popqa|3|datasets/popqa/raw/dev.jsonl|--limit 1000"
  "always|0|popqa|3|datasets/popqa/raw/dev.jsonl|--limit 1000"
)

run_one() { # gpu spec
  local gpu=$1 spec=$2
  IFS='|' read -r cond tau ds topk dev extra <<< "$spec"
  local out="results/${ds}_3way_${cond}.jsonl"
  local lg="scripts/3way_${cond}_${ds}.log"
  log "GPU$gpu START ${ds}/${cond} (tau=$tau k=$topk) -> $out"
  CUDA_VISIBLE_DEVICES=$gpu $PY --tau "$tau" --top-k "$topk" --dev "$dev" $extra --out "$out" > "$lg" 2>&1
  local rc=$?
  log "GPU$gpu DONE  ${ds}/${cond} rc=$rc  $(grep -m1 '^EM:' "$lg" 2>/dev/null) $(grep -m1 '^retrieval:' "$lg" 2>/dev/null)"
}

run_list() { # gpu idx...
  local gpu=$1; shift
  for idx in "$@"; do run_one "$gpu" "${JOBS[$idx]}"; done
}

# split 12 jobs across 2 GPUs (parity split keeps fast 'never' jobs balanced)
run_list 0 0 2 4 6 8 10 &
P0=$!
run_list 1 1 3 5 7 9 11 &
P1=$!
wait $P0; wait $P1
log "=== all 12 runs finished — computing 3-way tables + AUROC ==="

SUM=scripts/3way_summary.log
: > "$SUM"
for ds in hotpotqa 2wiki musique popqa; do
  python scripts/ablate_rag_usefulness.py \
    --never  "results/${ds}_3way_never.jsonl" \
    --gate   "results/${ds}_3way_gate.jsonl" \
    --always "results/${ds}_3way_always.jsonl" \
    --name "${ds} honest wiki-21M (1000Q, mode C, max_ent, path-aware)" >> "$SUM" 2>&1
done
log "=== SUMMARY written to $SUM ==="
cat "$SUM" | tee -a "$MASTER"
log "=== 3-way ablation driver COMPLETE ==="
