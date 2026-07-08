"""Stratified sampling: 100 HotpotQA questions (25 per stratum) for scale-up.

Same strata as sample_20q.py (HI_correct / HI_wrong / LO_correct / LO_wrong on
nr_sc median × nr_em). Pool sizes from sampled_20q.json: LO_correct has 91
records — sufficient for 25 per stratum.

Seed 42 to match prior sampling so the original 20 questions are a *subset*
of the new 100 (since random.sample is deterministic given seed and pool order).
"""

import json
import random
import statistics
from pathlib import Path

ROOT = Path("/home/S113065528/Research/deeprag_compare")
SRC = ROOT / "results/baseline/hotpotqa_meta-llama-3-8b-instruct.jsonl"
OUT = ROOT / "check/sampled_100q.json"

SEED = 42
PER_STRATUM = 25

recs = [json.loads(l) for l in open(SRC)]
median_sc = statistics.median(r["nr_sc"] for r in recs)

strata = {"HI_correct": [], "HI_wrong": [], "LO_correct": [], "LO_wrong": []}
for r in recs:
    hi = r["nr_sc"] >= median_sc
    ok = r["nr_em"] == 1
    key = ("HI" if hi else "LO") + ("_correct" if ok else "_wrong")
    strata[key].append(r)

rng = random.Random(SEED)
sampled = []
for stratum, pool in strata.items():
    picks = rng.sample(pool, PER_STRATUM)
    for p in picks:
        sampled.append({
            "stratum": stratum,
            "qid": p["qid"],
            "question": p["question"],
            "golds": p["golds"],
            "nr_sc": p["nr_sc"],
            "nr_pred": p["nr_pred"],
            "nr_em": p["nr_em"],
            "nr_f1": p["nr_f1"],
            "rag_pred": p["rag_pred"],
            "rag_em": p["rag_em"],
            "rag_sc": p["rag_sc"],
        })

OUT.parent.mkdir(exist_ok=True)
with open(OUT, "w") as f:
    json.dump({
        "seed": SEED,
        "median_sc": median_sc,
        "n_per_stratum": PER_STRATUM,
        "stratum_pool_sizes": {k: len(v) for k, v in strata.items()},
        "samples": sampled,
    }, f, indent=2, ensure_ascii=False)

print(f"Median SC = {median_sc:.3f}")
print(f"Pool sizes: {[(k, len(v)) for k, v in strata.items()]}")
print(f"\nSampled {len(sampled)} questions → {OUT}")

# Verify subset relation: are the original 20 contained in the 100?
prev = json.load(open(ROOT / "check/sampled_20q.json"))
prev_qids = {s["qid"] for s in prev["samples"]}
new_qids = {s["qid"] for s in sampled}
overlap = prev_qids & new_qids
print(f"\nOverlap with sampled_20q.json: {len(overlap)}/{len(prev_qids)}")
