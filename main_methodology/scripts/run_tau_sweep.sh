#!/usr/bin/env bash
# Fine-grained per-step retrieval-gate (tau) sweep on honest wiki-21M, 1000Q x 4 ds.
# Motivation: at canonical tau=0.3 the retrieval rate is 91-99% (gate barely
# fires) because honest-pool max_ent medians are ~2.2-3.2. To draw a real
# cost-accuracy tradeoff we sweep tau across the entropy distribution and
# measure (retrieval rate, EM) at each point. Endpoints tau in {1e9, 0.3, 0}
# already exist as the 3-way never/gate/always runs and are reused for plotting.
# Same current run_seek.py (mode C, max_ent, path-aware title) as the canonical
# method — only --tau changes.
set -u
cd /home/S113065528/Research/main_methodology
source /home/S113065528/miniconda3/etc/profile.d/conda.sh
conda activate thesis

ADAPTER=checkpoints/llama3-8b-decomposer-musique-2wiki-popqa
MASTER=scripts/tau_sweep_master.log
PY="python scripts/run_seek.py --adapter $ADAPTER --mode C --signal max_ent --corpus wiki"

ts()  { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a "$MASTER"; }

# new tau points (endpoints 0 / 0.3 / 1e9 reused from 3way runs).
# low band 0.1-0.5 added per supervisor request: documents that fine-tuning tau
# near the canonical 0.3 barely moves retrieval rate (curve is flat there);
# high band 1.5-3.0 is where retrieval rate actually drops.
TAUS=(0.1 0.2 0.4 0.5 1.0 1.5 2.0 2.5 3.0)

# ds|topk|dev|extra
DSETS=(
  "hotpotqa|8|datasets/hotpotqa/raw/dev_first1000.jsonl|"
  "2wiki|8|datasets/2wiki/raw/dev_first1000.jsonl|"
  "musique|8|datasets/musique/raw/dev_first1000.jsonl|"
  "popqa|3|datasets/popqa/raw/dev.jsonl|--limit 1000"
)

# build the full job list: ds x tau
JOBS=()
for d in "${DSETS[@]}"; do
  for t in "${TAUS[@]}"; do JOBS+=("$t|$d"); done
done

run_one() { # gpu spec
  local gpu=$1 spec=$2
  IFS='|' read -r tau ds topk dev extra <<< "$spec"
  local tag=${tau/./p}
  local out="results/${ds}_sweep_tau${tag}.jsonl"
  local lg="scripts/sweep_${ds}_tau${tag}.log"
  if grep -q '^EM:' "$lg" 2>/dev/null; then
    log "GPU$gpu SKIP  ${ds} tau=$tau (already done: $(grep -m1 '^retrieval:' "$lg"))"; return
  fi
  log "GPU$gpu START ${ds} tau=$tau (k=$topk) -> $out"
  CUDA_VISIBLE_DEVICES=$gpu $PY --tau "$tau" --top-k "$topk" --dev "$dev" $extra --out "$out" > "$lg" 2>&1
  local rc=$?
  log "GPU$gpu DONE  ${ds} tau=$tau rc=$rc  $(grep -m1 '^EM:' "$lg") $(grep -m1 '^retrieval:' "$lg")"
}

run_list() { local gpu=$1; shift; for i in "$@"; do run_one "$gpu" "${JOBS[$i]}"; done; }

N=${#JOBS[@]}
log "=== tau sweep START: $N jobs (${#DSETS[@]} ds x ${#TAUS[@]} tau) across 2 GPUs ==="
EVEN=(); ODD=()
for ((i=0;i<N;i++)); do (( i % 2 == 0 )) && EVEN+=($i) || ODD+=($i); done
run_list 0 "${EVEN[@]}" &
P0=$!
run_list 1 "${ODD[@]}" &
P1=$!
wait $P0; wait $P1
log "=== tau sweep COMPLETE — run: python scripts/plot_cost_accuracy.py ==="
