"""
Stratified sampling: pick 20 HotpotQA questions for SC diagnostic study.

Strata (2x2 on nr_sc median x nr_em):
  HI_correct   — SC high, model correct       (ideal: SC trustworthy)
  HI_wrong     — SC high, model wrong         (Confident Hallucination)
  LO_correct   — SC low, model correct        (SC false-alarm / lucky)
  LO_wrong     — SC low, model wrong          (SC catches genuine "don't know")

5 per stratum, total 20. Seed fixed for reproducibility.
"""

import json
import random
import statistics
from pathlib import Path

ROOT = Path("/home/S113065528/Research/deeprag_compare")
SRC = ROOT / "results/baseline/hotpotqa_meta-llama-3-8b-instruct.jsonl"
OUT = ROOT / "check/sampled_20q.json"

SEED = 42
PER_STRATUM = 5

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
print(f"\nSampled {len(sampled)} questions → {OUT}\n")
for s in sampled:
    q = s["question"][:70] + ("..." if len(s["question"]) > 70 else "")
    print(f"  [{s['stratum']:11s}] sc={s['nr_sc']:5.2f} em={s['nr_em']} | {q}")
