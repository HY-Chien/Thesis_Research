"""Sample a natural-distribution SFT set from a filtered decomposition file.

The final training set mirrors the hop-class distribution of the raw
MuSiQue train split (id prefix = hop class, e.g. `2hop`, `3hop1`, ...).

Per-class quota is proportional to the raw distribution. If a class has
fewer survivors than its quota, the deficit is topped up from the most
abundant class so the total still hits --n.

Usage:
    python sample_sft.py <count_matched.jsonl> --raw <raw/train.jsonl> \
        --n 4000 --out <path> [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def hop_class(row_id: str) -> str:
    return row_id.split("__")[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="count-matched decomposition jsonl")
    ap.add_argument("--raw", required=True, help="raw train jsonl (for distribution)")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # raw hop-class distribution
    raw_hop: Counter = Counter()
    with open(args.raw) as f:
        for line in f:
            raw_hop[hop_class(json.loads(line)["id"])] += 1
    raw_total = sum(raw_hop.values())

    # survivors bucketed by hop class
    pool: dict[str, list] = defaultdict(list)
    with open(args.input) as f:
        for line in f:
            r = json.loads(line)
            pool[hop_class(r["id"])].append(r)

    # proportional quota (largest-remainder so the quotas sum to exactly --n)
    classes = sorted(raw_hop)
    exact = {h: args.n * raw_hop[h] / raw_total for h in classes}
    quota = {h: int(exact[h]) for h in classes}
    rem = args.n - sum(quota.values())
    for h in sorted(classes, key=lambda c: exact[c] - quota[c], reverse=True)[:rem]:
        quota[h] += 1

    # sample per class; track deficits where survivors < quota
    picked: list = []
    deficit = 0
    leftover: dict[str, list] = {}
    for h in classes:
        avail = pool.get(h, [])
        rng.shuffle(avail)
        take = min(quota[h], len(avail))
        picked.extend(avail[:take])
        leftover[h] = avail[take:]
        deficit += quota[h] - take

    # top up deficit from the most abundant class's leftovers
    if deficit:
        abundant = max(classes, key=lambda c: len(leftover[c]))
        top_up = leftover[abundant][:deficit]
        picked.extend(top_up)
        print(f"[deficit {deficit}] topped up from {abundant} leftovers "
              f"({len(top_up)} added)")

    rng.shuffle(picked)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        for r in picked:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    final_hop = Counter(hop_class(r["id"]) for r in picked)
    print(f"\noutput: {out_path}  ({len(picked)} rows)")
    print(f"{'hop':8s}{'raw%':>9s}{'quota':>8s}{'avail':>8s}{'final':>8s}{'final%':>9s}")
    for h in classes:
        print(f"{h:8s}{raw_hop[h]/raw_total*100:8.1f}%{quota[h]:8d}"
              f"{len(pool.get(h, [])):8d}{final_hop[h]:8d}"
              f"{final_hop[h]/max(len(picked),1)*100:8.1f}%")


if __name__ == "__main__":
    main()
