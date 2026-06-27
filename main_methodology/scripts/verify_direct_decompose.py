"""Offline evaluation of verifier-based adaptive routing.

This script joins existing forced-direct and forced-decompose JSONL traces,
applies the training-free verifier, and reports EM/F1. It is the fastest way to
test verifier rules before paying for a full dual-run inference.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seek.retriever import PerQuestionBM25, format_context  # noqa: E402
from seek.verifier import choose_answer, exact_match, token_f1  # noqa: E402


def load_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[str(row["id"])] = row
    return rows


def load_dev(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[str(row["id"])] = row
    return rows


def build_open_retriever(corpus: str):
    if corpus == "none":
        return None
    if corpus == "paragraphs":
        return "paragraphs"
    from seek.retriever import OpenDomainBM25
    return OpenDomainBM25(corpus_name=corpus)


def _local_retriever(row: dict | None, retriever):
    if retriever is None:
        return None
    if retriever == "paragraphs":
        paragraphs = (row or {}).get("paragraphs") or []
        return PerQuestionBM25(paragraphs) if paragraphs else None
    return retriever


def evidence_items_for(record: dict, row: dict | None, retriever, top_k: int,
                       *, path: str) -> list[str]:
    local = _local_retriever(row, retriever)
    if local is None:
        return []

    if path == "direct":
        queries = [record.get("question") or (row or {}).get("question") or ""]
    else:
        steps = record.get("steps") or []
        queries = [
            step.get("sub_q") or step.get("raw_sub_q") or ""
            for step in steps
            if step.get("sub_q") or step.get("raw_sub_q")
        ]
        if not queries:
            queries = [record.get("question") or (row or {}).get("question") or ""]

    chunks = []
    for query in queries:
        if query:
            chunks.append(format_context(local.topk(query, k=top_k)))
        else:
            chunks.append("")
    return chunks


def evidence_for(record: dict, row: dict | None, retriever, top_k: int,
                 *, path: str) -> str:
    return "\n\n".join(evidence_items_for(record, row, retriever, top_k, path=path))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct", required=True, type=Path)
    ap.add_argument("--decomp", required=True, type=Path)
    ap.add_argument("--dev", type=Path, default=None)
    ap.add_argument("--corpus", default="none",
                    help="'none', 'paragraphs', or an ES index such as 'wiki'")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--fallback", choices=["auto", "direct", "decomp"], default="auto")
    ap.add_argument("--support-margin", type=float, default=0.35)
    ap.add_argument("--evidence-mode", choices=["final", "chain"], default="final")
    ap.add_argument("--direct-entropy-tau", type=float, default=0.3)
    ap.add_argument("--decomp-post-entropy-tau", type=float, default=0.8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    direct = load_jsonl(args.direct)
    decomp = load_jsonl(args.decomp)
    dev = load_dev(args.dev) if args.dev else {}
    retriever = build_open_retriever(args.corpus)

    ids = sorted(set(direct) & set(decomp))
    if args.limit:
        ids = ids[:args.limit]

    out_f = args.out.open("w", encoding="utf-8") if args.out else None
    em_sum = f1_sum = 0.0
    oracle_em_sum = direct_em_sum = decomp_em_sum = 0.0
    choice_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()

    for idx, qid in enumerate(ids, 1):
        d = direct[qid]
        c = decomp[qid]
        row = dev.get(qid)
        question = d.get("question") or c.get("question") or (row or {}).get("question") or ""
        gold = d.get("gold") or c.get("gold") or (row or {}).get("answer") or ""
        golds = [gold] + ((row or {}).get("answer_aliases") or [])

        direct_evidence = evidence_for(d, row, retriever, args.top_k, path="direct")
        decomp_step_evidences = []
        if args.evidence_mode == "chain":
            decomp_step_evidences = evidence_items_for(
                c, row, retriever, args.top_k, path="decomp"
            )
            decomp_evidence = "\n\n".join(decomp_step_evidences)
        else:
            decomp_evidence = evidence_for(c, row, retriever, args.top_k, path="decomp")
        decision = choose_answer(
            question, d, c,
            direct_evidence=direct_evidence,
            decomp_evidence=decomp_evidence,
            decomp_step_evidences=decomp_step_evidences,
            evidence_mode=args.evidence_mode,
            support_margin=args.support_margin,
            fallback=args.fallback,
            direct_entropy_tau=args.direct_entropy_tau,
            decomp_post_entropy_tau=args.decomp_post_entropy_tau,
        )
        pred = d.get("pred", "") if decision.choice == "direct" else c.get("pred", "")

        direct_em = exact_match(d.get("pred", ""), golds)
        decomp_em = exact_match(c.get("pred", ""), golds)
        final_em = exact_match(pred, golds)
        em_sum += float(final_em)
        f1_sum += token_f1(pred, golds)
        direct_em_sum += float(direct_em)
        decomp_em_sum += float(decomp_em)
        oracle_em_sum += float(direct_em or decomp_em)
        choice_counts[decision.choice] += 1
        reason_counts[decision.reason] += 1

        if out_f:
            out_f.write(json.dumps({
                "id": qid,
                "question": question,
                "gold": gold,
                "direct_pred": d.get("pred", ""),
                "decomp_pred": c.get("pred", ""),
                "pred": pred,
                "choice": decision.choice,
                "reason": decision.reason,
                "evidence_mode": args.evidence_mode,
                "direct_support": decision.direct_support,
                "decomp_support": decision.decomp_support,
                "direct_em": int(direct_em),
                "decomp_em": int(decomp_em),
                "em": int(final_em),
            }, ensure_ascii=False) + "\n")
            out_f.flush()

        if idx % 100 == 0 or idx == len(ids):
            print(f"  {idx}/{len(ids)} EM={em_sum/idx:.3f}", flush=True)

    if out_f:
        out_f.close()

    n = len(ids)
    print(f"\ndirect: {args.direct}")
    print(f"decomp: {args.decomp}")
    print(f"corpus: {args.corpus}  top_k={args.top_k}  fallback={args.fallback}  evidence_mode={args.evidence_mode}")
    print(f"N:            {n}")
    print(f"Direct EM:    {direct_em_sum/n:.1%}")
    print(f"Decomp EM:    {decomp_em_sum/n:.1%}")
    print(f"Verifier EM:  {em_sum/n:.1%}")
    print(f"Verifier F1:  {f1_sum/n:.1%}")
    print(f"Oracle EM:    {oracle_em_sum/n:.1%}")
    print("choices:      " + ", ".join(f"{k}={v}" for k, v in choice_counts.items()))
    print("reasons:      " + ", ".join(f"{k}={v}" for k, v in reason_counts.most_common()))
    if args.out:
        print(f"wrote:        {args.out}")


if __name__ == "__main__":
    main()
