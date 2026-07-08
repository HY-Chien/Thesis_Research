"""Direct parametric probe: answer the ORIGINAL question with NO retrieval,
NO decomposition, recording the full uncertainty-signal zoo.

This supplies the missing cell for the direct-only confidence-gated RAG
baseline (contrast to SEEK decompose+gate): joined offline with the existing
direct+always-RAG traces ({ds}_1000_single_k8.jsonl / popqa_..._top3.jsonl),
a threshold on max_ent replays "answer parametrically, retrieve only when
uncertain" for any tau (see scripts/replay_direct_gate.py).

Same model config as the canonical runs: 12K three-source adapter, mode C
(LoRA on), greedy decoding, same no-context answer prompt as the never-RAG
condition (seek.answerer.answer_with_entropy with context=None).

Usage:
    python scripts/run_direct_norag_probe.py --adapter <ckpt> \
        --dev datasets/2wiki/raw/dev_first1000.jsonl \
        --out results/2wiki_1000_direct_norag.jsonl [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seek.answerer import answer_with_entropy  # noqa: E402
from run_seek import BASE, em, f1  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--targ-prefix-k", type=int, default=8)
    ap.add_argument("--targ-beta", type=float, default=1.0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    rows = [json.loads(l) for l in open(args.dev)]
    if args.limit:
        rows = rows[:args.limit]

    n = len(rows)
    em_sum = f1_sum = 0.0
    with open(args.out, "w", encoding="utf-8") as out_f:
        for i, r in enumerate(rows, 1):
            res = answer_with_entropy(
                model, tok, r["question"], context=None,
                targ_prefix_k=args.targ_prefix_k, targ_beta=args.targ_beta)
            golds = [r["answer"]] + (r.get("answer_aliases") or [])
            e = em(res.text, golds)
            em_sum += float(e)
            f1_sum += f1(res.text, golds)
            rec = {
                "id": r.get("id"),
                "question": r["question"],
                "gold": r["answer"],
                "gold_aliases": r.get("answer_aliases") or [],
                "pred": res.text,
                "em": int(e),
            }
            rec.update({k: v for k, v in asdict(res).items() if k != "text"})
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            if i % 25 == 0 or i == n:
                print(f"  {i}/{n}  EM={em_sum/i:.3f}  F1={f1_sum/i:.3f}", flush=True)

    print(f"\nadapter: {args.adapter}")
    print(f"dev:     {args.dev}  ({n} rows)")
    print(f"EM:           {em_sum/n:.1%}")
    print(f"F1:           {f1_sum/n:.1%}")


if __name__ == "__main__":
    main()
