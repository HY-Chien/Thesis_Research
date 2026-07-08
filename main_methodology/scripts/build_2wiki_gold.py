"""Build a 2WikiMultiHopQA gold decomposition SFT set from `evidences`.

2Wiki has no natural-language sub-question decomposition like MuSiQue, but
every row carries `evidences` — ordered (fact, relation, entity) triples,
one per hop. Each triple is an atomic single-hop fact and converts cleanly
to the same `>>` / `#N` schema MuSiQue gold uses:

    (Move (1970 film)) --director--> (Stuart Rosenberg)
        => sub_question "Move (1970 film) >> director", answer "Stuart Rosenberg"

Dependency edges are recovered type-independently: if step i's `fact`
exactly matches an earlier step j's `entity`, step i depends on step j and
its subject is rewritten to the `#j` placeholder (chain hop). Steps whose
fact matches nothing earlier are independent (comparison hop).

The 4000-row set mirrors the natural distribution of the 2Wiki `type` field
(bridge_comparison / compositional / inference / comparison).

Usage:
    python build_2wiki_gold.py --raw <2wiki/raw/train.jsonl> \
        --n 4000 --out <path> [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def to_decomp(evidences: list[dict]) -> list[dict] | None:
    """Convert evidence triples to the decomposer schema, or None if unusable."""
    if not evidences:
        return None
    entity_at = {}  # normalized entity text -> step_id of the step that produced it
    steps = []
    for i, ev in enumerate(evidences, start=1):
        fact = (ev.get("fact") or "").strip()
        rel = (ev.get("relation") or "").strip()
        ent = (ev.get("entity") or "").strip()
        if not fact or not rel:
            return None
        dep = entity_at.get(fact.lower())
        subject = f"#{dep}" if dep else fact
        steps.append({
            "step_id": i,
            "sub_question": f"{subject} >> {rel}",
            "depends_on": [dep] if dep else [],
            "answer": ent,
            "paragraph_support_idx": None,
        })
        if ent:
            entity_at[ent.lower()] = i
    return steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # bucket usable rows by type
    pool: dict[str, list] = defaultdict(list)
    n_raw = n_bad = 0
    dep_steps = tot_steps = 0
    for line in open(args.raw):
        r = json.loads(line)
        n_raw += 1
        decomp = to_decomp(r.get("evidences") or [])
        if decomp is None:
            n_bad += 1
            continue
        r["gold_decomposition"] = decomp
        pool[r.get("type") or "unknown"].append(r)
        tot_steps += len(decomp)
        dep_steps += sum(1 for s in decomp if s["depends_on"])

    raw_type = {t: len(v) for t, v in pool.items()}
    raw_total = sum(raw_type.values())

    # proportional quota (largest-remainder)
    types = sorted(raw_type)
    exact = {t: args.n * raw_type[t] / raw_total for t in types}
    quota = {t: int(exact[t]) for t in types}
    rem = args.n - sum(quota.values())
    for t in sorted(types, key=lambda c: exact[c] - quota[c], reverse=True)[:rem]:
        quota[t] += 1

    picked = []
    for t in types:
        avail = list(pool[t])
        rng.shuffle(avail)
        picked.extend(avail[:quota[t]])
    rng.shuffle(picked)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        for r in picked:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"raw rows: {n_raw}   unusable (no/bad evidences): {n_bad}")
    print(f"dependency edges: {dep_steps}/{tot_steps} steps = "
          f"{dep_steps/max(tot_steps,1):.1%} are chain hops")
    print(f"\noutput: {out_path}  ({len(picked)} rows)")
    final_type = Counter(r.get("type") for r in picked)
    print(f"{'type':20s}{'raw%':>9s}{'quota':>8s}{'final':>8s}")
    for t in types:
        print(f"{t:20s}{raw_type[t]/raw_total*100:8.1f}%{quota[t]:8d}{final_type[t]:8d}")


if __name__ == "__main__":
    main()
