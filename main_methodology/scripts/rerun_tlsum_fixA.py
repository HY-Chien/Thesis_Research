"""Full-protocol rerun: prompt fix A (orig-question-grounded step answering)
with the step gate driven by tl_sum instead of max_ent.

Combines the two 2026-07-10 upgrades end to end (SS4.15 fix A + SS4.16 tl_sum):

  step prompts   carry "Original question: ..." (tentative + ctx probes),
                 exactly config A of rerun_prompt_fix.py
  step gate      retrieve iff tl_sum >= --tau (default 0.002305, the
                 accuracy-first operating point). tl_sum = -sum_t
                 log(top5m_t) over non-eos answer tokens, identical to
                 analyze_confidence_battery.compute_signals.
  everything else as canonical / rerun_prompt_fix config A: decomposition
                 reused from the canonical trace (greedy decomposer, so regen
                 would be identical), direct path untouched, retrieval =
                 sub_q top-8 title-on + orig-question top-8 title-off union,
                 steps adapter ON, plain synthesis adapter OFF, then
                 rescue -> verifier v2 -> typeaware -> alias-aware EM/F1.

The retrieved-branch answer is generated and stored for EVERY step (even the
ones the tl_sum gate skips) so the fix-A-trace gate curve can be replayed
offline later; the chain itself always follows the tl_sum gate decision.
For rate comparison the canonical max_ent tau=0.3 decision is recorded too.

Usage:
    python scripts/rerun_tlsum_fixA.py --ds hotpotqa --shard 0 --num-shards 2 \
        --out results/tlsum_fixA_hotpotqa_shard0.jsonl --resume
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from seek.retriever import OpenDomainBM25, format_context  # noqa: E402
from seek.pipeline import _merge_hits, _substitute  # noqa: E402
from seek.answerer import _prompt  # noqa: E402
from seek.verifier import token_f1  # noqa: E402
from probe_confidence_battery import BASE, greedy_probe  # noqa: E402
from analyze_confidence_battery import compute_signals  # noqa: E402
from rerun_prompt_fix import TRACE, TOP_K, evaluate, synthesize  # noqa: E402

TLSUM_TAU = 0.002305   # accuracy-first Tail Leakage operating point
MAXENT_TAU = 0.3       # canonical gate, recorded for rate comparison only


def step_sigs(tokens: list[dict]) -> tuple[float, float]:
    sig = compute_signals({"tokens": tokens}) or {}
    return float(sig.get("max_ent", 0.0)), float(sig.get("tl_sum", 0.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default="checkpoints/llama3-8b-decomposer-musique-2wiki-popqa")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--corpus", default="wiki")
    ap.add_argument("--tau", type=float, default=TLSUM_TAU)
    ap.add_argument(
        "--title-mode",
        choices=("off", "global", "pathaware"),
        default="pathaware",
        help=("off: text only for both paths; global: title+text for both; "
              "pathaware: direct text only, sub-question title+text"),
    )
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    title_direct = args.title_mode == "global"
    title_subq = args.title_mode != "off"

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    retriever = OpenDomainBM25(corpus_name=args.corpus)

    rows = [json.loads(l) for l in open(ROOT / TRACE.format(ds=args.ds),
                                        encoding="utf-8")]
    golds_map = {}
    for line in open(ROOT / f"results/conf_battery_{args.ds}.jsonl",
                     encoding="utf-8"):
        r = json.loads(line)
        golds_map[str(r["id"])] = [r["gold"]] + (r.get("aliases") or [])
    rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]
    if args.limit:
        rows = rows[:args.limit]

    done = set()
    if args.resume and Path(args.out).exists():
        for line in open(args.out, encoding="utf-8"):
            try:
                done.add(str(json.loads(line)["id"]))
            except json.JSONDecodeError:
                pass
    out_f = open(args.out, "a" if args.resume else "w", encoding="utf-8")

    t0 = time.time()
    n_done = 0
    run_em = old_em = 0
    n_steps = n_ret = n_ret_maxent = 0
    for i, row in enumerate(rows, 1):
        rid = str(row["id"])
        if rid in done:
            continue
        question = row["question"]
        golds = golds_map.get(rid) or (row["gold"] if isinstance(row["gold"], list)
                                       else [row["gold"]])
        stored_steps = (row.get("decomp") or {}).get("steps") or []
        rec = {"ds": args.ds, "id": rid, "question": question,
               "tau": args.tau, "title_mode": args.title_mode,
               "replay_em": int(row.get("replay_em", 0)),
               "replay_pred": row.get("replay_pred") or ""}
        if not stored_steps:
            rec["tl"] = {"em": rec["replay_em"],
                         "f1": token_f1(rec["replay_pred"], golds),
                         "pred": rec["replay_pred"],
                         "choice": row.get("replay_choice"), "unchanged": True}
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            n_done += 1
            run_em += rec["tl"]["em"]
            old_em += rec["replay_em"]
            continue

        direct_hits = retriever.topk(
            question, k=TOP_K, title_field=title_direct
        )

        answers: dict[int, str] = {}
        new_steps = []
        for s in stored_steps:
            resolved = _substitute(s["raw_sub_q"], answers)
            g = greedy_probe(model, tok, _prompt(tok, resolved, None,
                                                 orig_question=question))
            max_ent, tl_sum = step_sigs(g["tokens"])
            retrieved = tl_sum >= args.tau

            # retrieved branch always generated (offline replay material)
            subq_hits = retriever.topk(
                resolved, k=TOP_K, title_field=title_subq
            )
            ctx = format_context(_merge_hits(subq_hits, direct_hits))
            p = greedy_probe(model, tok, _prompt(tok, resolved, ctx,
                                                 orig_question=question))
            post_ent, post_tl = step_sigs(p["tokens"])

            final_ans = p["pred"] if retrieved else g["pred"]
            answers[s["step_id"]] = final_ans
            n_steps += 1
            n_ret += int(retrieved)
            n_ret_maxent += int(max_ent >= MAXENT_TAU)
            new_steps.append({
                "step_id": s["step_id"], "raw_sub_q": s["raw_sub_q"],
                "sub_q": resolved, "tentative": g["pred"],
                "final_answer": final_ans, "retrieved": retrieved,
                "max_ent": max_ent, "tl_sum": tl_sum,
                "post_signal_value": post_ent if retrieved else -1.0,
                "post_pred": p["pred"], "post_max_ent": post_ent,
                "post_tl_sum": post_tl,
                "tokens": g["tokens"], "post_tokens": p["tokens"],
            })

        ns = [SimpleNamespace(sub_q=s["sub_q"], final_answer=s["final_answer"])
              for s in new_steps]
        with model.disable_adapter():
            synth = synthesize(model, tok, question, ns, None)

        rec["tl"] = evaluate(row, golds, new_steps, synth)
        rec["new_steps"] = new_steps
        rec["step_changed"] = sum(
            int((n["final_answer"] or "") != (o["final_answer"] or ""))
            for n, o in zip(new_steps, stored_steps))
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
        n_done += 1
        run_em += rec["tl"]["em"]
        old_em += rec["replay_em"]
        if n_done % 25 == 0:
            dt = time.time() - t0
            print(f"[{args.ds} shard {args.shard}] {i}/{len(rows)} "
                  f"({dt/max(n_done,1):.1f}s/q) running EM "
                  f"old={old_em/n_done:.3f} tl_fixA={run_em/n_done:.3f} "
                  f"retr={n_ret}/{n_steps} "
                  f"(maxent-would={n_ret_maxent}/{n_steps})",
                  flush=True)
    out_f.close()
    print(f"done {args.ds} shard {args.shard}: {n_done} questions, "
          f"retrieve {n_ret}/{n_steps} steps "
          f"(maxent tau=0.3 would {n_ret_maxent}/{n_steps}), "
          f"{(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
