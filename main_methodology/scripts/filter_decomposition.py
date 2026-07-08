"""Filter an OSS decomposition file down to gold-verified-correct rows.

Inspired by DeepRAG's filter-stage1/extract.py: only decompositions whose
final answer matches the gold answer survive, yielding a clean SFT set.

A row is kept iff:
  - decomposition_error is None (parsed cleanly)
  - the last generated step's answer EM-matches gold `answer` (+ aliases)
  - with --require-count-match: generated step count == gold step count

Usage:
    python filter_decomposition.py <decomposition.jsonl> \
        [--out <path>] [--require-count-match]

Default --out is the input path with a `_gold_filtered` suffix.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_decomposition import em


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", default=None)
    ap.add_argument("--require-count-match", action="store_true",
                    help="also drop rows whose step count differs from gold")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out) if args.out else \
        in_path.with_name(in_path.stem + "_gold_filtered.jsonl")

    rows = [json.loads(l) for l in in_path.open()]
    n = len(rows)

    kept = []
    n_unparsed = n_wrong_answer = n_count_mismatch = 0
    for r in rows:
        if r["decomposition_error"] is not None:
            n_unparsed += 1
            continue
        d = r["generated_decomposition"]
        if not d:
            n_unparsed += 1
            continue

        gold = r["question_decomposition"]
        if args.require_count_match and len(d) != len(gold):
            n_count_mismatch += 1
            continue

        golds_final = [r["answer"]] + (r.get("answer_aliases") or [])
        last_ans = d[-1].get("answer") or ""
        if not em(last_ans, golds_final):
            n_wrong_answer += 1
            continue

        kept.append(r)

    with out_path.open("w", encoding="utf-8") as fout:
        for r in kept:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"input:  {in_path}  ({n} rows)")
    print(f"output: {out_path}  ({len(kept)} rows)")
    print(f"kept:   {len(kept)}/{n} = {len(kept)/max(n,1):.1%}")
    print(f"dropped — unparsed: {n_unparsed}  wrong_answer: {n_wrong_answer}"
          + (f"  count_mismatch: {n_count_mismatch}" if args.require_count_match else ""))


if __name__ == "__main__":
    main()
