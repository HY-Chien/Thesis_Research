"""Measure BM25 recall@k of gold (is_supporting) paragraphs, using SEEK's
actual sub-questions from a result trace.

For each question:
  - gold = set of paragraph idx with is_supporting == True
  - for each decomposition step, run PerQuestionBM25 over the question's pool
    with the resolved sub_q, take top-k
  - union the retrieved idx across steps
  - recall = |gold ∩ union| / |gold|   (also report full-coverage rate)

Usage:
  python recall_at_k.py --dev <dev.jsonl> --trace <result.jsonl> --ks 3 5 10
"""
import argparse, json, sys
sys.path.insert(0, ".")
from seek.retriever import PerQuestionBM25


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def gold_idx(rec):
    return {p["idx"] for p in rec.get("paragraphs", []) if p.get("is_supporting")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[3, 5, 10])
    a = ap.parse_args()

    dev = {str(r["id"]): r for r in load_jsonl(a.dev)}
    trace = load_jsonl(a.trace)

    maxk = max(a.ks)
    # per-k accumulators
    micro_hit = {k: 0 for k in a.ks}   # gold paragraphs recalled
    micro_tot = 0                       # total gold paragraphs
    full_cov = {k: 0 for k in a.ks}    # questions with ALL gold recalled
    n_q = 0
    n_skip_nodecomp = 0

    for t in trace:
        rec = dev.get(str(t["id"]))
        if rec is None:
            continue
        gold = gold_idx(rec)
        if not gold:
            continue
        paras = rec["paragraphs"]
        idx_of = {id(p): p["idx"] for p in paras}  # not needed; topk returns dicts
        bm = PerQuestionBM25(paras)

        # sub-questions actually used by SEEK; fall back to full question
        subqs = [s.get("sub_q") or s.get("raw_sub_q") for s in t.get("steps", [])]
        subqs = [s for s in subqs if s]
        if not subqs:
            subqs = [t["question"]]
            n_skip_nodecomp += 1

        n_q += 1
        micro_tot += len(gold)
        for k in a.ks:
            union = set()
            for sq in subqs:
                for p in bm.topk(sq, k=k):
                    union.add(p["idx"])
            hit = len(gold & union)
            micro_hit[k] += hit
            if hit == len(gold):
                full_cov[k] += 1

    print(f"\n  questions scored: {n_q}   (no-decomp fallback: {n_skip_nodecomp})")
    print(f"  total gold paragraphs: {micro_tot}   pool size: ~{len(paras)}")
    print(f"  {'k':>4} | {'recall(micro)':>14} | {'full-coverage':>14}")
    print("  " + "-" * 40)
    for k in a.ks:
        r = micro_hit[k] / micro_tot if micro_tot else 0
        fc = full_cov[k] / n_q if n_q else 0
        print(f"  {k:>4} | {r*100:>13.1f}% | {fc*100:>13.1f}%")


if __name__ == "__main__":
    main()
