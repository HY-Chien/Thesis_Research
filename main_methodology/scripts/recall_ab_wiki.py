"""M3 (honest wiki-21M) — Does decomposition retrieve gold passages better
than the original question, on the GLOBAL open-domain corpus DeepRAG uses?

Honest-pool counterpart of recall_ab.py. Instead of a per-question distractor
pool with is_supporting indices, we query the shared ES `wiki` index
(psgs_w100, 21M) and score recall by GOLD ARTICLE TITLE — the standard
open-domain multi-hop recall metric (IRCoT/DeepRAG).

Three query strategies (same as distractor M3):
  A          original full question, one BM25 query, top-k
  B-matched  SEEK sub-questions; merge all sub-q hits, global top-k by ES
             score  (FAIR vs A: same k passages)
  B-union    SEEK sub-questions; each sub-q its own top-k, then union of titles
             (what SEEK ACTUALLY sees; budget up to n_subq * k)

Gold = {normalised title : is_supporting paragraph}. A retrieved passage hits
a gold if its (normalised) title matches. recall = |gold∩retr| / |gold| (micro);
full-coverage = fraction of questions with ALL gold titles retrieved (the metric
that matters for multi-hop — every hop's article must be present to answer).

Usage:
  python recall_ab_wiki.py --tag HotpotQA --ks 3 5 10 \
      --dev datasets/hotpotqa/raw/dev_first1000.jsonl \
      --trace results/hotpotqa_1000_C_maxent0.3_3ds_terse.jsonl \
      --detail results/hotpotqa_m3_wiki_detail.jsonl
"""
import argparse, json, re, sys
sys.path.insert(0, ".")
from seek.retriever import OpenDomainBM25

_WS = re.compile(r"\s+")


def norm_title(t: str) -> str:
    t = (t or "").lower().replace("_", " ").strip()
    return _WS.sub(" ", t)


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def gold_titles(rec):
    return {norm_title(p["title"]) for p in rec.get("paragraphs", [])
            if p.get("is_supporting") and p.get("title")}


def titles_of(hits):
    """Distinct normalised titles, preserving ES rank order."""
    seen, out = set(), []
    for h in hits:
        nt = norm_title(h.get("title"))
        if nt and nt not in seen:
            seen.add(nt)
            out.append(nt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[3, 5, 10])
    ap.add_argument("--tag", default="")
    ap.add_argument("--corpus", default="wiki")
    ap.add_argument("--detail", default="")
    ap.add_argument("--limit", type=int, default=0, help="cap #questions (0=all)")
    ap.add_argument("--query-mode", choices=["full", "entity"], default="full",
                    help="full = raw sub_q ('X >> rel'); entity = left of '>>' only")
    ap.add_argument("--no-title-field", action="store_true",
                    help="match only paragraph_text (legacy handicapped retrieval)")
    a = ap.parse_args()

    ks = sorted(a.ks)
    maxk = max(ks)
    dev = {str(r["id"]): r for r in load_jsonl(a.dev)}
    trace = load_jsonl(a.trace)
    if a.limit:
        trace = trace[:a.limit]
    R = OpenDomainBM25(corpus_name=a.corpus,
                       query_title_field_too=not a.no_title_field)

    arms = ["A", "Bmatched", "Bunion"]
    hit = {arm: {k: 0 for k in ks} for arm in arms}
    full = {arm: {k: 0 for k in ks} for arm in arms}
    gold_tot = 0
    n_q = 0
    n_singlehop_subq = 0
    detail_fh = open(a.detail, "w") if a.detail else None

    for ti, t in enumerate(trace):
        rec = dev.get(str(t["id"]))
        if rec is None:
            continue
        gold = gold_titles(rec)
        if not gold:
            continue

        subqs = [s.get("sub_q") or s.get("raw_sub_q") for s in t.get("steps", [])]
        subqs = [s for s in subqs if s]
        if a.query_mode == "entity":
            # keep only the entity (left of '>>'); strips the relation tail that
            # adds noise tokens when the goal is to retrieve the gold ARTICLE
            subqs = [s.split(">>")[0].strip() or s for s in subqs]
        if len(subqs) < 2:
            n_singlehop_subq += 1
        if not subqs:
            subqs = [t["question"]]

        n_q += 1
        gold_tot += len(gold)

        # --- retrieve once at maxk, slice per k ---
        A_hits = R.topk(t["question"], k=maxk)                 # [{title,id,score,...}]
        sub_hits = [R.topk(sq, k=maxk) for sq in subqs]

        rec_detail = {"id": t["id"], "type": rec.get("type"),
                      "level": rec.get("level"), "n_gold": len(gold),
                      "gold": sorted(gold), "cov": {}}

        for k in ks:
            a_titles = set(titles_of(A_hits[:k]))
            # B-union: each sub-q its own top-k, union
            bu_titles = set()
            for sh in sub_hits:
                bu_titles |= set(titles_of(sh[:k]))
            # B-matched: merge all sub-q passages, dedupe by passage id (max
            # score), global top-k passages by ES score, then their titles
            merged = {}
            for sh in sub_hits:
                for h in sh:
                    pid = h.get("id")
                    if pid not in merged or h.get("score", 0) > merged[pid].get("score", 0):
                        merged[pid] = h
            ranked = sorted(merged.values(), key=lambda h: -h.get("score", 0))
            bm_titles = set(titles_of(ranked[:k]))

            cov_k = {}
            for arm, retr in (("A", a_titles), ("Bmatched", bm_titles), ("Bunion", bu_titles)):
                h = len(gold & retr)
                hit[arm][k] += h
                isfull = (h == len(gold))
                full[arm][k] += int(isfull)
                cov_k[arm] = {"hit": h, "full": int(isfull)}
            rec_detail["cov"][str(k)] = cov_k

        if detail_fh:
            detail_fh.write(json.dumps(rec_detail, ensure_ascii=False) + "\n")
        if (ti + 1) % 100 == 0:
            print(f"  ... {ti+1} done", flush=True)

    if detail_fh:
        detail_fh.close()

    tag = a.tag or a.trace
    print(f"\n=== {tag} (honest wiki-21M, title-match) ===")
    print(f"  questions scored: {n_q}   (<2 sub-q: {n_singlehop_subq})   "
          f"gold titles: {gold_tot}")
    print(f"  {'k':>3} | {'A recall':>9} {'Bmatch':>8} {'Bunion':>8} "
          f"| {'A fullcov':>9} {'Bmatch':>8} {'Bunion':>8} | {'Δfull(Bu-A)':>11}")
    print("  " + "-" * 80)
    for k in ks:
        rA = 100 * hit["A"][k] / gold_tot
        rBm = 100 * hit["Bmatched"][k] / gold_tot
        rBu = 100 * hit["Bunion"][k] / gold_tot
        fA = 100 * full["A"][k] / n_q
        fBm = 100 * full["Bmatched"][k] / n_q
        fBu = 100 * full["Bunion"][k] / n_q
        print(f"  {k:>3} | {rA:8.1f}% {rBm:7.1f}% {rBu:7.1f}% "
              f"| {fA:8.1f}% {fBm:7.1f}% {fBu:7.1f}% | {fBu-fA:+10.1f}")


if __name__ == "__main__":
    main()
