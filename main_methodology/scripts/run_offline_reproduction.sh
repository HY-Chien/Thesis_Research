#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PROJECT_DIR"
export MPLBACKEND="${MPLBACKEND:-Agg}"

"$PYTHON_BIN" scripts/analyze_tlsum_fixA_step_gate_final.py
"$PYTHON_BIN" scripts/analyze_skipped_retrieval_questions.py
"$PYTHON_BIN" scripts/analyze_accuracy_first_failure_taxonomy.py
"$PYTHON_BIN" scripts/analyze_confidence_battery.py
"$PYTHON_BIN" scripts/analyze_tlsum_confidence_story.py
"$PYTHON_BIN" scripts/analyze_tail_leakage_variants.py
"$PYTHON_BIN" scripts/make_direct_decompose_2x2_fig.py
"$PYTHON_BIN" scripts/make_routing_gap_fig.py
"$PYTHON_BIN" scripts/analyze_quadrant_confidence.py

echo "Offline reproduction complete: $PROJECT_DIR/results/reproduced"
