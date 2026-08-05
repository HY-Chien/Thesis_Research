"""Run verifier-based adaptive SEEK routing.

For each question, run both:
  1. forced direct answer with original-question retrieval
  2. normal decomposition pipeline

Then use the training-free verifier to choose the final answer.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seek.pipeline import run_seek  # noqa: E402
from seek.retriever import PerQuestionBM25, format_context  # noqa: E402
from seek.verifier import choose_answer, exact_match, token_f1  # noqa: E402

BASE = os.environ.get(
    "SEEK_BASE_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct"
)
HIGHER_IS_UNCERTAIN_SIGNALS = {
    "max_ent", "mean_ent", "tail_leakage", "tl_sum", "targ_unc"
}


def force_direct_tau(signal: str) -> float:
    return 1e9 if signal in HIGHER_IS_UNCERTAIN_SIGNALS else -1e9


def trace_record(trace) -> dict:
    return {
        "question": trace.question,
        "pred": trace.final_answer,
        "skipped_decompose": trace.skipped_decompose,
        "skip_probe_ent": trace.skip_probe_ent,
        "decomposition": trace.decomposition,
        "steps": [asdict(step) for step in trace.steps],
    }


def make_retriever(row: dict, open_retriever):
    if open_retriever is not None:
        return open_retriever
    paragraphs = row.get("paragraphs") or []
    return PerQuestionBM25(paragraphs) if paragraphs else None


def evidence_for(record: dict, retriever, top_k: int, *, path: str) -> str:
    if retriever is None:
        return ""
    if path == "direct":
        queries = [record.get("question", "")]
    else:
        queries = [
            step.get("sub_q") or step.get("raw_sub_q") or ""
            for step in record.get("steps", [])
            if step.get("sub_q") or step.get("raw_sub_q")
        ] or [record.get("question", "")]
    chunks = []
    for query in queries:
        if query:
            chunks.append(format_context(retriever.topk(query, k=top_k)))
    return "\n\n".join(chunks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", default=BASE,
                    help="Hugging Face model ID or local path (default: "
                         "SEEK_BASE_MODEL or meta-llama/Meta-Llama-3-8B-Instruct)")
    ap.add_argument("--dev", required=True)
    ap.add_argument("--mode", choices=["B", "C"], required=True)
    ap.add_argument("--signal",
                    choices=["tail_leakage", "tl_sum", "max_ent", "mean_ent", "min_lp",
                             "maqa_margin", "mean_margin", "max_margin",
                             "targ_unc", "targ_logit_gap"],
                    default="tail_leakage")
    ap.add_argument("--targ-prefix-k", type=int, default=8,
                    help="number of generated prefix tokens for TARG-style logit-gap averaging")
    ap.add_argument("--targ-beta", type=float, default=1.0,
                    help="temperature beta for TARG uncertainty exp(-gap / beta)")
    ap.add_argument(
        "--tau",
        type=float,
        default=None,
        help="retrieval threshold (default: 0.002305 for Tail Leakage, 0.3 otherwise)",
    )
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--corpus", default="paragraphs",
                    help="'paragraphs' or an ES index such as 'wiki'")
    ap.add_argument("--es-host", default=os.environ.get("SEEK_ES_HOST", "localhost"),
                    help="Elasticsearch host (default: SEEK_ES_HOST or localhost)")
    ap.add_argument("--es-port", type=int,
                    default=int(os.environ.get("SEEK_ES_PORT", "9200")),
                    help="Elasticsearch port (default: SEEK_ES_PORT or 9200)")
    ap.add_argument("--post-retrieval-tau", type=float, default=None)
    ap.add_argument("--post-retrieval-drop-tau", type=float, default=None,
                    help="if set, accept retrieved decomposition-step answers only "
                         "when post-retrieval uncertainty drops by this amount")
    ap.add_argument("--support-margin", type=float, default=0.35)
    ap.add_argument(
        "--direct-confidence-tau", "--direct-entropy-tau",
        dest="direct_confidence_tau", type=float, default=None,
        help="direct-path confidence threshold (legacy option name remains accepted)",
    )
    ap.add_argument(
        "--decomp-post-confidence-tau", "--decomp-post-entropy-tau",
        dest="decomp_post_confidence_tau", type=float, default=None,
        help="post-decomposition confidence threshold (legacy option name remains accepted)",
    )
    ap.add_argument("--fallback", choices=["auto", "direct", "decomp"], default="auto")
    ap.add_argument(
        "--typeaware-fallback",
        action="store_true",
        help="apply the deterministic answer-type fallback after the base verifier",
    )
    ap.add_argument("--typeaware-margin", type=float, default=2.5)
    ap.add_argument("--title-mode", choices=["off", "global", "pathaware"],
                    default="pathaware",
                    help="BM25 title-field matching: off / global (all queries) / "
                         "pathaware (sub-q only, not the long original question)")
    ap.add_argument("--no-evidence", action="store_true",
                    help="canonical v2: skip lexical evidence-support rule "
                         "(use only confidence signals and structural rules)")
    ap.add_argument("--evidence-mode", choices=["final", "chain"], default="final",
                    help="final: support score on final answer; chain: also score "
                         "each decomposition step (v3). Ignored if --no-evidence.")
    ap.add_argument("--direct-context-union", action="store_true",
                    help="for decomposition steps, append original-question top-k "
                         "evidence to each sub-question context")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    is_tail_leakage = args.signal in {"tail_leakage", "tl_sum"}
    if args.tau is None:
        args.tau = 0.002305 if is_tail_leakage else 0.3
    if args.direct_confidence_tau is None:
        args.direct_confidence_tau = 0.0005 if is_tail_leakage else 0.3
    if args.decomp_post_confidence_tau is None:
        args.decomp_post_confidence_tau = 0.0158 if is_tail_leakage else 0.8

    open_retriever = None
    if args.corpus != "paragraphs":
        from seek.retriever import OpenDomainBM25
        open_retriever = OpenDomainBM25(
            corpus_name=args.corpus, host=args.es_host, port=args.es_port
        )

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    rows = [json.loads(line) for line in open(args.dev, encoding="utf-8") if line.strip()]
    if args.limit:
        rows = rows[:args.limit]

    out_f = None
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = out_path.open("w", encoding="utf-8")
    em_sum = f1_sum = 0.0
    direct_em_sum = decomp_em_sum = oracle_em_sum = 0.0
    choices: Counter[str] = Counter()
    reasons: Counter[str] = Counter()

    direct_skip_tau = force_direct_tau(args.signal)

    for idx, row in enumerate(rows, 1):
        direct_trace = run_seek(
            model, tok, row,
            tau=args.tau,
            top_k=args.top_k,
            mode=args.mode,
            signal=args.signal,
            targ_prefix_k=args.targ_prefix_k,
            targ_beta=args.targ_beta,
            skip_decompose_tau=direct_skip_tau,
            post_retrieval_tau=args.post_retrieval_tau,
            post_retrieval_drop_tau=args.post_retrieval_drop_tau,
            open_retriever=open_retriever,
            title_mode=args.title_mode,
        )
        decomp_trace = run_seek(
            model, tok, row,
            tau=args.tau,
            top_k=args.top_k,
            mode=args.mode,
            signal=args.signal,
            targ_prefix_k=args.targ_prefix_k,
            targ_beta=args.targ_beta,
            skip_decompose_tau=None,
            post_retrieval_tau=args.post_retrieval_tau,
            post_retrieval_drop_tau=args.post_retrieval_drop_tau,
            open_retriever=open_retriever,
            title_mode=args.title_mode,
            direct_context_union=args.direct_context_union,
        )

        direct = trace_record(direct_trace)
        decomp = trace_record(decomp_trace)
        retriever = None if args.no_evidence else make_retriever(row, open_retriever)
        direct_ev = "" if args.no_evidence else evidence_for(direct, retriever, args.top_k, path="direct")
        decomp_ev = "" if args.no_evidence else evidence_for(decomp, retriever, args.top_k, path="decomp")
        decomp_step_ev = []
        if (not args.no_evidence) and args.evidence_mode == "chain" and retriever is not None:
            for s in decomp.get("steps", []):
                sq = s.get("sub_q") or s.get("raw_sub_q")
                decomp_step_ev.append(
                    format_context(retriever.topk(sq, k=args.top_k, title_field=True)) if sq else "")
        decision = choose_answer(
            row["question"], direct, decomp,
            direct_evidence=direct_ev,
            decomp_evidence=decomp_ev,
            decomp_step_evidences=decomp_step_ev,
            evidence_mode=args.evidence_mode,
            support_margin=args.support_margin,
            fallback=args.fallback,
            direct_entropy_tau=args.direct_confidence_tau,
            decomp_post_entropy_tau=args.decomp_post_confidence_tau,
            higher_is_uncertain=args.signal in HIGHER_IS_UNCERTAIN_SIGNALS,
        )
        choice = decision.choice
        reason = decision.reason
        if args.typeaware_fallback:
            from replay_typeaware_verifier import choose_typeaware
            choice, reason = choose_typeaware(
                {
                    "question": row["question"],
                    "choice": decision.choice,
                    "reason": decision.reason,
                    "direct_pred": direct["pred"],
                    "decomp_pred": decomp["pred"],
                    "direct": direct,
                    "decomp": decomp,
                },
                direct_entropy_tau=args.direct_confidence_tau,
                decomp_post_tau=args.decomp_post_confidence_tau,
                margin=args.typeaware_margin,
            )
        pred = direct["pred"] if choice == "direct" else decomp["pred"]

        golds = [row["answer"]] + (row.get("answer_aliases") or [])
        direct_em = exact_match(direct["pred"], golds)
        decomp_em = exact_match(decomp["pred"], golds)
        final_em = exact_match(pred, golds)
        em_sum += float(final_em)
        f1_sum += token_f1(pred, golds)
        direct_em_sum += float(direct_em)
        decomp_em_sum += float(decomp_em)
        oracle_em_sum += float(direct_em or decomp_em)
        choices[choice] += 1
        reasons[reason] += 1

        if out_f:
            out_f.write(json.dumps({
                "id": row.get("id"),
                "question": row["question"],
                "gold": row["answer"],
                "signal": args.signal,
                "tau": args.tau,
                "targ_prefix_k": args.targ_prefix_k,
                "targ_beta": args.targ_beta,
                "direct_pred": direct["pred"],
                "decomp_pred": decomp["pred"],
                "pred": pred,
                "choice": choice,
                "reason": reason,
                "base_verifier_choice": decision.choice,
                "base_verifier_reason": decision.reason,
                "direct_support": decision.direct_support,
                "decomp_support": decision.decomp_support,
                "direct_em": int(direct_em),
                "decomp_em": int(decomp_em),
                "em": int(final_em),
                "direct": direct,
                "decomp": decomp,
            }, ensure_ascii=False) + "\n")
            out_f.flush()

        if idx % 25 == 0 or idx == len(rows):
            print(f"  {idx}/{len(rows)} EM={em_sum/idx:.3f} "
                  f"choice={dict(choices)}", flush=True)

    if out_f:
        out_f.close()

    n = len(rows)
    print(f"\nadapter:       {args.adapter}")
    print(f"dev:           {args.dev} ({n} rows)")
    print(f"mode:          {args.mode} signal={args.signal} tau={args.tau} "
          f"top_k={args.top_k} corpus={args.corpus} "
          f"targ_prefix_k={args.targ_prefix_k} targ_beta={args.targ_beta}")
    print(f"verifier:      direct_tau={args.direct_confidence_tau} "
          f"decomp_post_tau={args.decomp_post_confidence_tau} "
          f"typeaware={args.typeaware_fallback}")
    print(f"Direct EM:     {direct_em_sum/n:.1%}")
    print(f"Decomp EM:     {decomp_em_sum/n:.1%}")
    print(f"Verifier EM:   {em_sum/n:.1%}")
    print(f"Verifier F1:   {f1_sum/n:.1%}")
    print(f"Oracle EM:     {oracle_em_sum/n:.1%}")
    print("choices:       " + ", ".join(f"{k}={v}" for k, v in choices.items()))
    print("reasons:       " + ", ".join(f"{k}={v}" for k, v in reasons.most_common()))


if __name__ == "__main__":
    main()
