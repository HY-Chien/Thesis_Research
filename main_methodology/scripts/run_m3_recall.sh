#!/usr/bin/env bash
# M3 — retrieval recall@k: original question (A) vs decomposed sub-questions (B).
# Runs the three multi-hop datasets that carry is_supporting gold labels.
# PopQA is single-hop with no is_supporting gold -> N/A (excluded by design).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/S113065528/miniconda3/envs/thesis/bin/python3   # has rank_bm25

TRACE_SUFFIX="C_maxent0.3_3ds_terse"   # canonical 12K three-source decomposer

$PY scripts/recall_ab.py --tag "2WikiMultiHopQA" --ks 3 5 10 \
  --dev   datasets/2wiki/raw/dev_first1000.jsonl \
  --trace results/2wiki_1000_${TRACE_SUFFIX}.jsonl

$PY scripts/recall_ab.py --tag "HotpotQA" --ks 3 5 10 \
  --dev   datasets/hotpotqa/raw/dev_first1000.jsonl \
  --trace results/hotpotqa_1000_${TRACE_SUFFIX}.jsonl

$PY scripts/recall_ab.py --tag "MuSiQue" --ks 3 5 10 \
  --dev   datasets/musique/raw/dev_first1000.jsonl \
  --trace results/musique_1000_${TRACE_SUFFIX}.jsonl
