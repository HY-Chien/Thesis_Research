"""M3 — Does decomposition retrieve gold passages better than the original question?

Compares retrieval recall@k of the gold (is_supporting) paragraphs from a
question's distractor pool, under three query strategies at a MATCHED budget:

  A          original full question, one BM25 query, top-k
  B-matched  SEER sub-questions; score each paragraph by its BEST-matching
             sub-q, take the global top-k  (FAIR vs A: same k paragraphs)
  B-union    SEER sub-questions; each sub-q takes its own top-k, then union
             (what SEER ACTUALLY sees; budget up to n_subq * k)

Gold = {idx : is_supporting}. recall = |gold ∩ retrieved| / |gold|.
Also reports full-coverage = fraction of questions with ALL gold recalled.

Only multi-hop datasets with is_supporting labels apply (2wiki/hotpotqa/musique);
PopQA has no is_supporting gold and is single-hop -> N/A.

Usage:
  python recall_ab.py --dev <dev.jsonl> --trace <trace_with_steps.jsonl> --ks 3 5 10
"""
import argparse, json, sys
sys.path.insert(0, ".")
from seer.retriever import PerQuestionBM25, _tok


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def gold_idx(rec):
    return {p["idx"] for p in rec.get("paragraphs", []) if p.get("is_supporting")}


def topk_idx_from_scores(scores, paras, k):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return [paras[i]["idx"] for i in order[:k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[3, 5, 10])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    dev = {str(r["id"]): r for r in load_jsonl(a.dev)}
    trace = load_jsonl(a.trace)
    ks = a.ks

    # accumulators: arm -> k -> running totals. recall is micro (over gold paras).
    arms = ["A", "Bmatched", "Bunion"]
    hit = {arm: {k: 0 for k in ks} for arm in arms}
    full = {arm: {k: 0 for k in ks} for arm in arms}
    gold_tot = 0
    n_q = 0          # questions scored (decomposed, gold present)
    n_singlehop_subq = 0   # questions where SEER produced <2 sub-qs

    for t in trace:
        rec = dev.get(str(t["id"]))
        if rec is None:
            continue
        gold = gold_idx(rec)
        if not gold:
            continue
        paras = rec["paragraphs"]
        bm = PerQuestionBM25(paras)

        subqs = [s.get("sub_q") or s.get("raw_sub_q") for s in t.get("steps", [])]
        subqs = [s for s in subqs if s]
        if len(subqs) < 2:
            n_singlehop_subq += 1
        if not subqs:
            subqs = [t["question"]]

        n_q += 1
        gold_tot += len(gold)

        # --- precompute BM25 scores once per query ---
        scoresA = bm.bm25.get_scores(_tok(t["question"]))
        sub_scores = [bm.bm25.get_scores(_tok(sq)) for sq in subqs]
        # B-matched: per-paragraph max score across sub-qs
        scoresBmax = [max(s[i] for s in sub_scores) for i in range(len(paras))]

        for k in ks:
            # Arm A
            a_idx = set(topk_idx_from_scores(scoresA, paras, k))
            # Arm B-matched (global top-k of best-sub-q score)
            bm_idx = set(topk_idx_from_scores(scoresBmax, paras, k))
            # Arm B-union (each sub-q its own top-k, union)
            bu_idx = set()
            for s in sub_scores:
                bu_idx |= set(topk_idx_from_scores(s, paras, k))

            for arm, retr in (("A", a_idx), ("Bmatched", bm_idx), ("Bunion", bu_idx)):
                h = len(gold & retr)
                hit[arm][k] += h
                if h == len(gold):
                    full[arm][k] += 1

    tag = a.tag or a.trace
    print(f"\n=== {tag} ===")
    print(f"  questions scored: {n_q}   (<2 sub-q: {n_singlehop_subq})   "
          f"gold paras: {gold_tot}   pool~{len(paras)}")
    print(f"  {'k':>3} | {'A recall':>9} {'Bmatch':>8} {'Bunion':>8} "
          f"| {'A fullcov':>9} {'Bmatch':>8} {'Bunion':>8} | {'Δrecall(Bm-A)':>13}")
    print("  " + "-" * 78)
    for k in ks:
        rA = 100 * hit["A"][k] / gold_tot
        rBm = 100 * hit["Bmatched"][k] / gold_tot
        rBu = 100 * hit["Bunion"][k] / gold_tot
        fA = 100 * full["A"][k] / n_q
        fBm = 100 * full["Bmatched"][k] / n_q
        fBu = 100 * full["Bunion"][k] / n_q
        print(f"  {k:>3} | {rA:8.1f}% {rBm:7.1f}% {rBu:7.1f}% "
              f"| {fA:8.1f}% {fBm:7.1f}% {fBu:7.1f}% | {rBm-rA:+12.1f}")


if __name__ == "__main__":
    main()
