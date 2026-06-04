"""Filter an OSS decomposition file to count-matched, structurally-valid rows.

Model-independent decomposition-quality gate — NO final-answer EM.

At the data-construction stage we only care whether OSS-20B *decomposed*
the question correctly, not whether it *answered* the sub-questions (the
latter is model-dependent and irrelevant to building a decomposer SFT set).
So this filter judges structure against the MuSiQue gold decomposition only.

A row is kept iff:
  - decomposition_error is None (parsed cleanly)
  - every depends_on edge references a STRICTLY EARLIER step
    (no forward refs, no self refs, no out-of-range refs)
  - generated step count == gold question_decomposition step count

Usage:
    python filter_count_match.py <decomposition.jsonl> [--out <path>]

Default --out is the input path with a `_count_matched` suffix.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def deps_valid(decomp: list[dict]) -> bool:
    """Every depends_on edge must point to a strictly earlier step (1-indexed)."""
    for step in decomp:
        sid = step["step_id"]
        for d in step.get("depends_on") or []:
            if d < 1 or d >= sid:
                return False
    return True


def hop_class(row_id: str) -> str:
    return row_id.split("__")[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out) if args.out else \
        in_path.with_name(in_path.stem + "_count_matched.jsonl")

    rows = [json.loads(l) for l in in_path.open()]
    n = len(rows)

    kept = []
    n_unparsed = n_bad_deps = n_count_mismatch = 0
    tot_hop: Counter = Counter()
    kept_hop: Counter = Counter()

    for r in rows:
        hop = hop_class(r["id"])
        tot_hop[hop] += 1

        if r["decomposition_error"] is not None:
            n_unparsed += 1
            continue
        d = r["generated_decomposition"]
        if not d:
            n_unparsed += 1
            continue
        if not deps_valid(d):
            n_bad_deps += 1
            continue
        if len(d) != len(r["question_decomposition"]):
            n_count_mismatch += 1
            continue

        kept.append(r)
        kept_hop[hop] += 1

    with out_path.open("w", encoding="utf-8") as fout:
        for r in kept:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"input:  {in_path}  ({n} rows)")
    print(f"output: {out_path}  ({len(kept)} rows)")
    print(f"kept:   {len(kept)}/{n} = {len(kept)/max(n,1):.1%}")
    print(f"dropped — unparsed: {n_unparsed}  bad_deps: {n_bad_deps}  "
          f"count_mismatch: {n_count_mismatch}")
    print("\nper hop class (kept / total):")
    for h in sorted(tot_hop):
        k, t = kept_hop[h], tot_hop[h]
        print(f"  {h:8s} {k:6d} / {t:6d} = {k/max(t,1):.1%}")


if __name__ == "__main__":
    main()
