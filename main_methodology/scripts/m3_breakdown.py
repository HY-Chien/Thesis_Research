"""Per-type full-coverage breakdown from M3 honest-pool detail files.

For each dataset detail file (id, type, level, n_gold, cov[k][arm]) report
full-coverage@k for arm A (single query) vs B-union (decomposition), split by
question type. This is what tells us whether the honest-pool retrieval gap is
concentrated in specific question types (e.g. HotpotQA comparison vs bridge).
"""
import json, sys
from collections import defaultdict


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def musique_hop(rid):
    # ids like "2hop__...", "3hop1__...", "4hop2__..."
    return rid.split("__", 1)[0] if "__" in str(rid) else "?"


def main():
    for path, name, keyfn in [
        ("results/hotpotqa_m3_wiki_detail.jsonl", "HotpotQA", lambda r: r.get("type") or "?"),
        ("results/2wiki_m3_wiki_detail.jsonl", "2Wiki", lambda r: r.get("type") or "?"),
        ("results/musique_m3_wiki_detail.jsonl", "MuSiQue", lambda r: musique_hop(r["id"])),
    ]:
        try:
            rows = load(path)
        except FileNotFoundError:
            print(f"\n[skip] {path} not found yet")
            continue
        ks = sorted((int(k) for k in rows[0]["cov"]), key=int)
        print(f"\n=== {name}  (n={len(rows)})  full-coverage@k: A / Bunion ===")
        groups = defaultdict(list)
        for r in rows:
            groups[keyfn(r)].append(r)
        groups["__ALL__"] = rows
        hdr = "  ".join(f"k={k}" for k in ks)
        print(f"  {'type':<20} {'n':>5}   {hdr}")
        for g in sorted(groups, key=lambda x: (x != "__ALL__", x)):
            rs = groups[g]
            cells = []
            for k in ks:
                sk = str(k)
                fa = 100 * sum(x["cov"][sk]["A"]["full"] for x in rs) / len(rs)
                fb = 100 * sum(x["cov"][sk]["Bunion"]["full"] for x in rs) / len(rs)
                cells.append(f"{fa:4.0f}/{fb:4.0f}")
            label = "ALL" if g == "__ALL__" else g
            print(f"  {label:<20} {len(rs):>5}   {'  '.join(cells)}")


if __name__ == "__main__":
    main()
