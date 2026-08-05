"""Drive SEEK end-to-end inference on a dev set; report QA EM/F1 + retrieval rate.

Compares two operating modes:
  B  adapter-toggle  — LoRA on for decompose, off (pure base) for answer
  C  adapter-always  — LoRA on for both decompose AND answer

Usage:
    python run_seek.py --adapter <ckpt> --dev <dev.jsonl> --mode {B,C} \
        [--tau 0.5] [--top-k 5] [--limit N] [--out <jsonl>]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seek.pipeline import run_seek  # noqa: E402

BASE = os.environ.get(
    "SEEK_BASE_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct"
)


def _norm(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em(pred: str, golds: list[str]) -> bool:
    p = _norm(pred)
    return any(p == _norm(g) for g in golds)


def f1(pred: str, golds: list[str]) -> float:
    p = _norm(pred).split()
    best = 0.0
    for g in golds:
        gt = _norm(g).split()
        if not p or not gt:
            best = max(best, float(p == gt))
            continue
        common = Counter(p) & Counter(gt)
        n = sum(common.values())
        if n == 0:
            continue
        prec, rec = n / len(p), n / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


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
        help="retrieval threshold (default: 0.002305 for Tail Leakage, 0.8 otherwise)",
    )
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--corpus", default="paragraphs",
                    help="'paragraphs' = per-question pool (default); otherwise "
                         "an ES index name (e.g. 'wiki') for open-domain "
                         "retrieval matching DeepRAG.")
    ap.add_argument("--es-host", default=os.environ.get("SEEK_ES_HOST", "localhost"),
                    help="Elasticsearch host (default: SEEK_ES_HOST or localhost)")
    ap.add_argument("--es-port", type=int,
                    default=int(os.environ.get("SEEK_ES_PORT", "9200")),
                    help="Elasticsearch port (default: SEEK_ES_PORT or 9200)")
    ap.add_argument("--skip-decompose-tau", type=float, default=None,
                    help="if set, probe original question with retrieval first; "
                         "if signal confident, skip decomposition entirely")
    ap.add_argument("--post-retrieval-tau", type=float, default=None,
                    help="if set, re-measure entropy AFTER retrieval; if the "
                         "answer is still uncertain (retrieval missed gold), "
                         "revert to the model's parametric pre-retrieval answer")
    ap.add_argument("--post-retrieval-drop-tau", type=float, default=None,
                    help="if set, accept retrieved answer only when the chosen "
                         "uncertainty signal drops by at least this amount")
    ap.add_argument("--direct-context-union", action="store_true",
                    help="for decomposition steps, append original-question top-k "
                         "evidence to each sub-question context")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None, help="optional jsonl trace dump")
    args = ap.parse_args()
    if args.tau is None:
        args.tau = 0.002305 if args.signal in {"tail_leakage", "tl_sum"} else 0.8

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

    rows = [
        json.loads(line)
        for line in open(args.dev, encoding="utf-8")
        if line.strip()
    ]
    if args.limit:
        rows = rows[:args.limit]

    out_f = None
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = out_path.open("w", encoding="utf-8")
    n = len(rows)
    em_sum = f1_sum = 0.0
    retr_steps = total_steps = 0
    n_decomp_fail = 0
    n_skipped = 0
    n_fallback = 0

    for i, r in enumerate(rows, 1):
        trace = run_seek(model, tok, r, tau=args.tau, top_k=args.top_k,
                         mode=args.mode, signal=args.signal,
                         targ_prefix_k=args.targ_prefix_k,
                         targ_beta=args.targ_beta,
                         skip_decompose_tau=args.skip_decompose_tau,
                         post_retrieval_tau=args.post_retrieval_tau,
                         post_retrieval_drop_tau=args.post_retrieval_drop_tau,
                         open_retriever=open_retriever,
                         direct_context_union=args.direct_context_union)
        if trace.skipped_decompose:
            n_skipped += 1
        elif not trace.steps:
            n_decomp_fail += 1

        golds = [r["answer"]] + (r.get("answer_aliases") or [])
        e = em(trace.final_answer, golds)
        em_sum += float(e)
        f1_sum += f1(trace.final_answer, golds)
        total_steps += len(trace.steps)
        retr_steps += sum(1 for s in trace.steps if s.retrieved)
        n_fallback += sum(1 for s in trace.steps if s.used_fallback)

        if out_f:
            out_f.write(json.dumps({
                "id": r.get("id"),
                "question": r["question"],
                "gold": r["answer"],
                "pred": trace.final_answer,
                "signal": args.signal,
                "tau": args.tau,
                "targ_prefix_k": args.targ_prefix_k,
                "targ_beta": args.targ_beta,
                "skipped_decompose": trace.skipped_decompose,
                "skip_probe_ent": trace.skip_probe_ent,
                "em": int(e),
                "decomposition": trace.decomposition,
                "steps": [asdict(s) for s in trace.steps],
            }, ensure_ascii=False) + "\n")
            out_f.flush()

        if i % 25 == 0 or i == n:
            print(f"  {i}/{n}  EM={em_sum/i:.3f}  F1={f1_sum/i:.3f}  "
                  f"retr={retr_steps/max(total_steps,1):.2f}  "
                  f"skip={n_skipped}/{i}", flush=True)

    if out_f:
        out_f.close()

    print(f"\nadapter: {args.adapter}")
    print(f"dev:     {args.dev}  ({n} rows)")
    print(f"mode:    {args.mode}   signal={args.signal}   tau={args.tau}   "
          f"top_k={args.top_k}   skip_decompose_tau={args.skip_decompose_tau}   "
          f"targ_prefix_k={args.targ_prefix_k}   targ_beta={args.targ_beta}")
    print(f"EM:           {em_sum/n:.1%}")
    print(f"F1:           {f1_sum/n:.1%}")
    print(f"retrieval:    {retr_steps}/{total_steps} steps = "
          f"{retr_steps/max(total_steps,1):.1%}")
    print(f"skipped:      {n_skipped}/{n} ({n_skipped/n:.1%})")
    print(f"fallback:     {n_fallback}/{retr_steps} retrieved steps reverted to parametric")
    print(f"decomp_fail:  {n_decomp_fail}")


if __name__ == "__main__":
    main()
