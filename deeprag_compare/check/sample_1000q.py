"""Build sampled_1000q.json — all baseline records (full HotpotQA dev pool).

Pool is exactly 1000, so this is the complete coverage; we keep stratum labels
for downstream analysis but no random subsampling is performed. The sample order
matches the baseline JSONL order so split-by-index parallelization is stable.
"""

import json
import statistics
from pathlib import Path

ROOT = Path("/home/S113065528/Research/deeprag_compare")
SRC = ROOT / "results/baseline/hotpotqa_meta-llama-3-8b-instruct.jsonl"
OUT = ROOT / "check/sampled_1000q.json"

recs = [json.loads(l) for l in open(SRC)]
median_sc = statistics.median(r["nr_sc"] for r in recs)

strata = {"HI_correct": [], "HI_wrong": [], "LO_correct": [], "LO_wrong": []}
sampled = []
for r in recs:
    hi = r["nr_sc"] >= median_sc
    ok = r["nr_em"] == 1
    key = ("HI" if hi else "LO") + ("_correct" if ok else "_wrong")
    strata[key].append(r)
    sampled.append({
        "stratum": key,
        "qid": r["qid"],
        "question": r["question"],
        "golds": r["golds"],
        "nr_sc": r["nr_sc"],
        "nr_pred": r["nr_pred"],
        "nr_em": r["nr_em"],
        "nr_f1": r["nr_f1"],
        "rag_pred": r["rag_pred"],
        "rag_em": r["rag_em"],
        "rag_sc": r["rag_sc"],
    })

OUT.parent.mkdir(exist_ok=True)
with open(OUT, "w") as f:
    json.dump({
        "seed": None,
        "median_sc": median_sc,
        "n_per_stratum": None,
        "stratum_pool_sizes": {k: len(v) for k, v in strata.items()},
        "samples": sampled,
    }, f, indent=2, ensure_ascii=False)

print(f"Median SC = {median_sc:.3f}")
print(f"Pool sizes: {[(k, len(v)) for k, v in strata.items()]}")
print(f"\nSampled {len(sampled)} questions → {OUT}")

prev = json.load(open(ROOT / "check/sampled_100q.json"))
prev_qids = {s["qid"] for s in prev["samples"]}
new_qids = {s["qid"] for s in sampled}
overlap = prev_qids & new_qids
print(f"Overlap with sampled_100q.json: {len(overlap)}/{len(prev_qids)}")
