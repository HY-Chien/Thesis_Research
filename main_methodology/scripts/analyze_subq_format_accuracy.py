"""Does raw_sub_q format (">> relation" template vs natural-language) predict
per-step answering accuracy in SEEK traces?

Aligns SEEK's predicted steps to MuSiQue gold question_decomposition by
position (only for rows where step count == gold hop count), classifies each
step's raw_sub_q as "template" (contains '>>') or "natural", and compares
final_answer against the gold hop answer.

Usage:
    python analyze_subq_format_accuracy.py \
        --trace results/musique_FULL_C_maxent0.3.jsonl \
        --gold datasets/musique/raw/dev.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict


def norm(text: str | None) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def is_match(pred: str, gold: str) -> bool:
    p, g = norm(pred), norm(gold)
    if not p or not g:
        return False
    return p == g or p in g or g in p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--gold", required=True)
    args = ap.parse_args()

    gold_by_id = {}
    with open(args.gold, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            gold_by_id[d["id"]] = d["question_decomposition"]

    buckets = defaultdict(lambda: [0, 0])  # key -> [correct, total]
    hop_pos_buckets = defaultdict(lambda: [0, 0])  # (hop_idx, fmt) -> [correct, total]
    aligned_rows = 0
    total_rows = 0
    mismatched_len = 0
    examples = defaultdict(list)

    with open(args.trace, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            total_rows += 1
            gid = row["id"]
            gold_steps = gold_by_id.get(gid)
            pred_steps = row.get("steps") or []
            if gold_steps is None:
                continue
            if len(gold_steps) != len(pred_steps):
                mismatched_len += 1
                continue
            aligned_rows += 1
            for i, (gs, ps) in enumerate(zip(gold_steps, pred_steps)):
                fmt = "template" if ">>" in (ps.get("raw_sub_q") or "") else "natural"
                correct = is_match(ps.get("final_answer"), gs.get("answer"))
                buckets[fmt][0] += int(correct)
                buckets[fmt][1] += 1
                hop_pos_buckets[(i + 1, fmt)][0] += int(correct)
                hop_pos_buckets[(i + 1, fmt)][1] += 1
                if not correct and len(examples[fmt]) < 5:
                    examples[fmt].append({
                        "id": gid,
                        "raw_sub_q": ps.get("raw_sub_q"),
                        "sub_q": ps.get("sub_q"),
                        "pred": ps.get("final_answer"),
                        "gold": gs.get("answer"),
                    })

    print(f"total rows: {total_rows}, aligned (step count == gold hop count): {aligned_rows}, "
          f"skipped (length mismatch): {mismatched_len}")
    print()
    print("=== overall accuracy by raw_sub_q format ===")
    for fmt, (c, t) in sorted(buckets.items()):
        print(f"  {fmt:10s}: {c}/{t} = {c/t:.1%}")
    print()
    print("=== accuracy by hop position x format ===")
    for (pos, fmt), (c, t) in sorted(hop_pos_buckets.items()):
        if t == 0:
            continue
        print(f"  hop {pos}  {fmt:10s}: {c}/{t} = {c/t:.1%}")
    print()
    for fmt, exs in examples.items():
        print(f"=== sample wrong answers ({fmt}) ===")
        for e in exs:
            print(f"  [{e['id']}] raw_sub_q={e['raw_sub_q']!r} sub_q={e['sub_q']!r} "
                  f"pred={e['pred']!r} gold={e['gold']!r}")


if __name__ == "__main__":
    main()
