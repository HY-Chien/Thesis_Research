"""Answer-containing retrieval recall on MuSiQue: SEEK decomposer vs GOLD
sub-questions (advisor request #5, thesis slide p38 follow-up).

Same setting as `results/answer_containing_recall_4ds_open_domain.md`:
open-domain wiki-21M ES index, path-aware retrieval — original question
queries paragraph text only (title-off), sub-questions query title+text
(title-on). hit@k = any of the top-k retrieved passages contains the gold
final answer or one of its aliases (normalised substring). Sub-question arm
retrieves top-k per sub-question and unions the passages (what the pipeline
actually sees).

Arms:
  orig       original question, title-off
  seek       SEEK decomposer sub-questions from the canonical trace
             (placeholders resolved with *predicted* step answers, as at
             inference time)
  gold       MuSiQue gold `question_decomposition` sub-questions with #N
             placeholders resolved with *gold* step answers (oracle
             decomposition + oracle intermediate answers)
  gold_raw   gold sub-questions with placeholders left unresolved
             (isolates how much of the gold gain comes from gold
             intermediate answers vs the decomposition itself)
  + unions with orig for seek/gold.

Usage (from Research/main_methodology):
  python scripts/recall_gold_decomp_musique.py \
      --dev datasets/musique/raw/dev_first1000.jsonl \
      --trace results/musique_1000_verify_v2_pathaware.jsonl \
      --ks 1 3 5 8 10 [--limit 100]
"""
import argparse
import json
import re
import sys

sys.path.insert(0, ".")
from seek.retriever import OpenDomainBM25

_WS = re.compile(r"\s+")


def norm(s: str) -> str:
    return _WS.sub(" ", (s or "").lower().strip())


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def resolve_placeholders(subqs, answers):
    """Replace #N in each sub-question with the given step-N answer."""
    out = []
    for q in subqs:
        for i, a in enumerate(answers, 1):
            if a:
                q = q.replace(f"#{i}", a)
        out.append(q)
    return out


def contains_answer(passages, needles, include_title=True):
    for p in passages:
        text = norm(f"{p.get('title', '')} {p.get('paragraph_text', '')}"
                    if include_title else p.get("paragraph_text", ""))
        if any(n in text for n in needles):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="datasets/musique/raw/dev_first1000.jsonl")
    ap.add_argument("--trace", default="results/musique_1000_verify_v2_pathaware.jsonl")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 8, 10])
    ap.add_argument("--corpus", default="wiki")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--detail", default="")
    ap.add_argument("--base-decomp", default="",
                    help="jsonl from decompose_base_musique.py; adds arms "
                         "base (gold-resolved #N), base_raw, and seek_goldres")
    a = ap.parse_args()

    ks = sorted(a.ks)
    maxk = max(ks)
    dev = {str(r["id"]): r for r in load_jsonl(a.dev)}
    trace = load_jsonl(a.trace)
    if a.limit:
        trace = trace[:a.limit]

    R_on = OpenDomainBM25(corpus_name=a.corpus, query_title_field_too=True)
    R_off = OpenDomainBM25(corpus_name=a.corpus, query_title_field_too=False)

    base_decomp = {}
    if a.base_decomp:
        base_decomp = {str(r["id"]): r for r in load_jsonl(a.base_decomp)}

    arms = ["orig", "seek", "gold", "gold_raw", "orig+seek", "orig+gold"]
    if base_decomp:
        arms += ["base", "base_raw", "seek_goldres"]
    hit = {arm: {k: 0 for k in ks} for arm in arms}
    hit_text = {arm: {k: 0 for k in ks} for arm in arms}
    n_q = 0
    n_gold_subq = {}
    detail_fh = open(a.detail, "w") if a.detail else None

    for ti, t in enumerate(trace):
        rec = dev.get(str(t["id"]))
        if rec is None:
            continue
        needles = {norm(rec["answer"])} | {norm(x) for x in rec.get("answer_aliases") or []}
        needles = {n for n in needles if n}
        if not needles:
            continue

        steps = t.get("steps") or (t.get("decomp") or {}).get("steps") or []
        seek_subqs = [s.get("sub_q") or s.get("raw_sub_q") for s in steps]
        seek_subqs = [s for s in seek_subqs if s]

        gd = rec.get("question_decomposition") or []
        gold_raw_subqs = [d["question"] for d in gd]
        gold_answers = [d.get("answer") or "" for d in gd]
        gold_subqs = resolve_placeholders(gold_raw_subqs, gold_answers)
        if not gold_subqs:
            continue
        n_gold_subq[len(gold_subqs)] = n_gold_subq.get(len(gold_subqs), 0) + 1
        n_q += 1

        orig_hits = R_off.topk(rec["question"], k=maxk)
        seek_hits = [R_on.topk(q, k=maxk) for q in seek_subqs]
        gold_hits = [R_on.topk(q, k=maxk) for q in gold_subqs]
        gold_raw_hits = [R_on.topk(q, k=maxk) for q in gold_raw_subqs]

        base_hits = base_raw_hits = seek_goldres_hits = None
        if base_decomp:
            bd = base_decomp.get(str(t["id"]))
            # fallback to the original question when the base model produced
            # no parseable steps — same behaviour as the pipeline
            base_raw_subqs = (bd or {}).get("subqs") or [rec["question"]]
            base_subqs = resolve_placeholders(base_raw_subqs, gold_answers)
            seek_raw_subqs = [s.get("raw_sub_q") or s.get("sub_q") for s in steps]
            seek_raw_subqs = [s for s in seek_raw_subqs if s] or [rec["question"]]
            seek_goldres_subqs = resolve_placeholders(seek_raw_subqs, gold_answers)
            base_hits = [R_on.topk(q, k=maxk) for q in base_subqs]
            base_raw_hits = [R_on.topk(q, k=maxk) for q in base_raw_subqs]
            seek_goldres_hits = [R_on.topk(q, k=maxk) for q in seek_goldres_subqs]

        rec_detail = {"id": t["id"], "hit": {}}
        for k in ks:
            pools = {
                "orig": orig_hits[:k],
                "seek": [p for h in seek_hits for p in h[:k]],
                "gold": [p for h in gold_hits for p in h[:k]],
                "gold_raw": [p for h in gold_raw_hits for p in h[:k]],
            }
            pools["orig+seek"] = pools["orig"] + pools["seek"]
            pools["orig+gold"] = pools["orig"] + pools["gold"]
            if base_decomp:
                pools["base"] = [p for h in base_hits for p in h[:k]]
                pools["base_raw"] = [p for h in base_raw_hits for p in h[:k]]
                pools["seek_goldres"] = [p for h in seek_goldres_hits for p in h[:k]]
            rec_detail["hit"][str(k)] = {}
            for arm in arms:
                got = contains_answer(pools[arm], needles)
                got_text = contains_answer(pools[arm], needles, include_title=False)
                hit[arm][k] += int(got)
                hit_text[arm][k] += int(got_text)
                rec_detail["hit"][str(k)][arm] = int(got)
                rec_detail["hit"][str(k)][arm + "_textonly"] = int(got_text)
        if detail_fh:
            detail_fh.write(json.dumps(rec_detail, ensure_ascii=False) + "\n")
        if (ti + 1) % 100 == 0:
            print(f"  ... {ti + 1} done", flush=True)

    if detail_fh:
        detail_fh.close()

    print(f"\n=== MuSiQue answer-containing hit@k (N={n_q}, wiki-21M, path-aware) ===")
    print(f"  gold sub-question counts: {dict(sorted(n_gold_subq.items()))}")
    for label, table in (("title+text match", hit), ("text-only match", hit_text)):
        print(f"\n  -- {label} --")
        header = "  " + f"{'k':>3} | " + " ".join(f"{arm:>9}" for arm in arms)
        print(header)
        print("  " + "-" * (len(header)))
        for k in ks:
            row = " ".join(f"{100 * table[arm][k] / n_q:8.2f}%" for arm in arms)
            print(f"  {k:>3} | {row}")


if __name__ == "__main__":
    main()
