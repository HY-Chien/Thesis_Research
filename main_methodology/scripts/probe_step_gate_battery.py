"""Per-step gate-signal probe: regen every decomp step's NO-CONTEXT tentative
answer with full token stats (top5m / renyi05 / ... for gate-signal swap), and
fill in the missing RETRIEVED branch for steps the canonical gate (max_ent
tau=0.3) skipped.

Replicates seek/pipeline.py exactly: answer prompt = ANSWER_INSTRUCTION on the
stored resolved sub_q, adapter mode C (LoRA ON), greedy 48 tokens; retrieved
branch ctx = merge(sub_q top-8 title_field=True, original-question top-8
title_field=False) per --title-mode pathaware --direct-context-union.
Canonical runs had post_retrieval taus = None (used_fallback == 0 in all
traces), so retrieved-branch answer = post.text unconditionally.

Sanity: regen tentative text must match the stored step `tentative`, and the
recomputed max_ent must match the stored per-step max_ent.

Usage:
    python scripts/probe_step_gate_battery.py \
        --trace results/2wiki_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl \
        --out results/step_gate_battery_2wiki.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from seek.retriever import OpenDomainBM25, format_context  # noqa: E402
from seek.pipeline import _merge_hits  # noqa: E402
from seek.answerer import _prompt  # noqa: E402
from probe_confidence_battery import BASE, greedy_probe  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default="checkpoints/llama3-8b-decomposer-musique-2wiki-popqa")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--corpus", default="wiki")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    retriever = OpenDomainBM25(corpus_name=args.corpus)

    rows = [json.loads(l) for l in open(args.trace, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]

    done = set()
    if args.resume and Path(args.out).exists():
        for line in open(args.out, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((str(r["id"]), r["step_id"]))
            except json.JSONDecodeError:
                pass
    out_f = open(args.out, "a" if args.resume else "w", encoding="utf-8")

    t0 = time.time()
    n_step = n_match = n_ent = n_post = 0
    for i, tr in enumerate(rows, 1):
        rid = str(tr.get("id"))
        steps = (tr.get("decomp") or {}).get("steps") or []
        question = tr["question"]
        direct_hits = None
        for s in steps:
            key = (rid, s["step_id"])
            if key in done:
                continue
            # no-ctx tentative regen (gate-signal branch), adapter ON (mode C)
            g = greedy_probe(model, tok, _prompt(tok, s["sub_q"], None))
            n_step += 1
            match = int(g["pred"] == (s.get("tentative") or "").strip())
            n_match += match
            ent_regen = max(t["ent"] for t in g["tokens"]) if g["tokens"] else -1.0
            n_ent += int(abs(ent_regen - float(s.get("max_ent") or -2)) < 1e-3)
            rec = {
                "id": rid, "step_id": s["step_id"], "sub_q": s["sub_q"],
                "stored_tentative": s.get("tentative"),
                "stored_max_ent": s.get("max_ent"),
                "stored_retrieved": bool(s.get("retrieved")),
                "pred": g["pred"], "match": match, "tokens": g["tokens"],
            }
            if not s.get("retrieved"):
                # missing retrieved branch: rebuild ctx exactly as pipeline
                if direct_hits is None:
                    direct_hits = retriever.topk(
                        question, k=args.top_k, title_field=False)
                subq_hits = retriever.topk(
                    s["sub_q"], k=args.top_k, title_field=True)
                ctx = format_context(_merge_hits(subq_hits, direct_hits))
                p = greedy_probe(model, tok, _prompt(tok, s["sub_q"], ctx))
                rec["post_pred"] = p["pred"]
                rec["post_tokens"] = p["tokens"]
                n_post += 1
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if i % 20 == 0 or i == len(rows):
            out_f.flush()
            dt = time.time() - t0
            print(f"  [{i}/{len(rows)}] steps={n_step} tent_match={n_match}/{n_step} "
                  f"ent_match={n_ent}/{n_step} post_filled={n_post} "
                  f"({dt/max(n_step,1):.2f}s/step)", flush=True)
    out_f.close()
    print(f"DONE {args.out} steps={n_step} tent_match={n_match} ent_match={n_ent} "
          f"post={n_post}", flush=True)


if __name__ == "__main__":
    main()
