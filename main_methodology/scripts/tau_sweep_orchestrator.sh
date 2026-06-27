#!/usr/bin/env bash
# Orchestrator: wait for the in-flight 5-tau sweep to finish, then run the
# low-band (0.1-0.5) pass (run_tau_sweep.sh SKIP logic re-uses done jobs),
# then build the cost-accuracy table + figure.
set -u
cd /home/S113065528/Research/main_methodology
source /home/S113065528/miniconda3/etc/profile.d/conda.sh
conda activate thesis
MASTER=scripts/tau_sweep_master.log
ORCH=scripts/tau_sweep_orchestrator.log
ts() { date '+%F %T'; }
log() { echo "[$(ts)] $*" | tee -a "$ORCH"; }

log "orchestrator START — waiting for first sweep COMPLETE"
# wait until the currently running sweep finishes its 20 jobs
while ! grep -q 'tau sweep COMPLETE' "$MASTER" 2>/dev/null; do sleep 60; done
log "first sweep done — launching low-band pass (SKIP re-uses finished jobs)"

# second pass: full TAUS grid now includes low band; SKIP skips the 20 done.
bash scripts/run_tau_sweep.sh
log "low-band pass done — building cost-accuracy table + figure"

python scripts/plot_cost_accuracy.py >> "$ORCH" 2>&1
log "orchestrator COMPLETE — results/cost_accuracy_tradeoff.{md,png}"
