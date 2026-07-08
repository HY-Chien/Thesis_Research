"""Sample a natural-distribution SFT set from MuSiQue GOLD decompositions.

Unlike sample_sft.py (which samples OSS-20B-generated, count-matched rows),
this samples straight from the raw MuSiQue train split — every gold
decomposition is correct by construction, so there is no quality filter.

Each sampled row carries a `gold_decomposition` field converted to the
decomposer training schema:
    {step_id, sub_question, depends_on, answer, paragraph_support_idx}
- step_id      : 1-indexed step position
- sub_question : gold step `question` text (keeps MuSiQue `#N` placeholders)
- depends_on   : step ids referenced by `#N` placeholders in sub_question
- answer       : gold step answer
- paragraph_support_idx : gold supporting paragraph index

The 4000-row set mirrors the hop-class distribution of raw train
(id prefix = hop class, e.g. `2hop`, `3hop1`, ...).

Usage:
    python sample_gold_sft.py --raw <raw/train.jsonl> \
        --n 4000 --out <path> [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


def hop_class(row_id: str) -> str:
    return row_id.split("__")[0]


def to_decomp_schema(question_decomposition: list[dict]) -> list[dict]:
    """Convert gold question_decomposition to the decomposer training schema."""
    steps = []
    for i, gs in enumerate(question_decomposition, start=1):
        q = gs.get("question", "")
        depends_on = sorted({int(x) for x in re.findall(r"#(\d+)", q)})
        steps.append({
            "step_id": i,
            "sub_question": q,
            "depends_on": depends_on,
            "answer": gs.get("answer"),
            "paragraph_support_idx": gs.get("paragraph_support_idx"),
        })
    return steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw train jsonl")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # raw rows bucketed by hop class
    pool: dict[str, list] = defaultdict(list)
    with open(args.raw) as f:
        for line in f:
            r = json.loads(line)
            pool[hop_class(r["id"])].append(r)
    raw_hop = {h: len(v) for h, v in pool.items()}
    raw_total = sum(raw_hop.values())

    # proportional quota (largest-remainder so the quotas sum to exactly --n)
    classes = sorted(raw_hop)
    exact = {h: args.n * raw_hop[h] / raw_total for h in classes}
    quota = {h: int(exact[h]) for h in classes}
    rem = args.n - sum(quota.values())
    for h in sorted(classes, key=lambda c: exact[c] - quota[c], reverse=True)[:rem]:
        quota[h] += 1

    # sample per class (every gold row is valid, so no deficit handling needed)
    picked: list = []
    for h in classes:
        avail = list(pool[h])
        rng.shuffle(avail)
        picked.extend(avail[:quota[h]])

    # attach converted gold decomposition
    for r in picked:
        r["gold_decomposition"] = to_decomp_schema(r["question_decomposition"])

    rng.shuffle(picked)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        for r in picked:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    final_hop = Counter(hop_class(r["id"]) for r in picked)
    print(f"output: {out_path}  ({len(picked)} rows)")
    print(f"{'hop':8s}{'raw%':>9s}{'quota':>8s}{'final':>8s}{'final%':>9s}")
    for h in classes:
        print(f"{h:8s}{raw_hop[h]/raw_total*100:8.1f}%{quota[h]:8d}"
              f"{final_hop[h]:8d}{final_hop[h]/max(len(picked),1)*100:8.1f}%")


if __name__ == "__main__":
    main()
