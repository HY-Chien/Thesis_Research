"""Prompt-fix rerun: original-question-grounded step answering + evidence-aware
synthesis, evaluated at the main-table protocol.

Motivated by SS4.14: 57% of the HotpotQA questions lost to DeepRAG fail at the
STEP-ANSWER stage (gold in retrieved ctx, but the sub-question dropped the
disambiguating qualifiers and the model extracts the wrong entity), and 18%
fail at SYNTHESIS (a step held the gold answer but the chain-only synthesis
prompt lost it). Neither the step prompt (answerer._prompt) nor the synthesis
prompt (pipeline._synthesize) sees the original question / any evidence.

Configs per question (decomposition reused from the canonical trace; direct
path untouched; gate = max_ent tau=0.3; retrieval = sub_q top-8 title-on +
original-question top-8 title-off union; post_retrieval taus None; steps with
adapter ON, synthesis with adapter OFF -- all as canonical):

  A  step prompts carry "Original question: ..." (tentative + ctx probes);
     plain canonical synthesis
  B  stored canonical steps; synthesis prompt gains per-step top-1 evidence
  C  A's steps + evidence synthesis

Each config's decomp pred goes through _rescue_value_answer -> verifier v2
choose_answer -> typeaware replay -> alias-aware EM / token F1.

Usage:
    python scripts/rerun_prompt_fix.py --ds hotpotqa --shard 0 --num-shards 2 \
        --out results/prompt_fix_hotpotqa_shard0.jsonl --resume
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
from seek.pipeline import (_merge_hits, _substitute, _rescue_value_answer,  # noqa: E402
                           SYNTHESIZE_INSTRUCTION)
from seek.answerer import answer_with_entropy  # noqa: E402
from seek.verifier import choose_answer, exact_match, token_f1  # noqa: E402
from replay_typeaware_verifier import choose_typeaware  # noqa: E402
from probe_confidence_battery import BASE  # noqa: E402

TRACE = "results/{ds}_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl"
TAU = 0.3
TOP_K = 8
EV_CHARS = 400
EV_MAX = 4

EV_RULE = ("- Retrieved evidence snippets are provided; if a sub-answer "
           "conflicts with the evidence or the question, trust the evidence.\n")


@torch.no_grad()
def synthesize(model, tok, question: str, steps, evidence: list[str] | None,
               max_new_tokens: int = 48) -> str:
    sub = "\n".join(f"- {s.sub_q}  ->  {s.final_answer}" for s in steps)
    instr = SYNTHESIZE_INSTRUCTION
    body = f"Sub-answers:\n{sub}\n\nQuestion: {question}\nAnswer:"
    if evidence:
        ev = "\n".join(f"- {e}" for e in evidence[:EV_MAX])
        instr = SYNTHESIZE_INSTRUCTION.replace(
            "- Output the answer phrase ONLY", EV_RULE + "- Output the answer phrase ONLY")
        body = f"Evidence:\n{ev}\n\n{body}"
    msgs = [{"role": "user", "content": f"{instr}\n{body}"}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()


def top1_snippet(hits: list[dict]) -> str | None:
    if not hits:
        return None
    h = hits[0]
    text = h.get("text") or h.get("paragraph_text") or ""
    return f"{h.get('title', '')}: {text[:EV_CHARS]}"


def evaluate(row, golds, steps, pred_raw):
    """rescue -> verifier v2 -> typeaware, mirroring sweep_step_gate_signals."""
    ns = [SimpleNamespace(sub_q=s["sub_q"], final_answer=s["final_answer"])
          for s in steps]
    rescued = _rescue_value_answer(row["question"], pred_raw, ns)
    decomp_rec = {
        "question": row["question"], "pred": rescued,
        "skipped_decompose": False,
        "decomposition": (row.get("decomp") or {}).get("decomposition") or [],
        "steps": [{
            "step_id": s["step_id"], "raw_sub_q": s["raw_sub_q"],
            "sub_q": s["sub_q"], "tentative": s["tentative"],
            "final_answer": s["final_answer"], "retrieved": s["retrieved"],
            "max_ent": s["max_ent"], "signal_value": s["max_ent"],
            "post_signal_value": s["post_signal_value"],
        } for s in steps],
    }
    ver = choose_answer(row["question"], row.get("direct") or {}, decomp_rec,
                        fallback="auto")
    synth_row = {
        "question": row["question"], "choice": ver.choice, "reason": ver.reason,
        "direct_pred": row.get("direct_pred") or "", "decomp_pred": rescued,
        "direct": row.get("direct") or {}, "decomp": decomp_rec,
    }
    choice, _ = choose_typeaware(synth_row, direct_entropy_tau=0.3,
                                 decomp_post_tau=0.8, margin=2.5)
    if choice == "direct":
        pred = row.get("direct_pred") or ""
        em = int(row.get("direct_em", 0))
    else:
        pred = rescued
        em = int(exact_match(rescued, golds))
    return {"choice": choice, "pred": pred, "decomp_pred": rescued,
            "em": em, "f1": token_f1(pred, golds)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default="checkpoints/llama3-8b-decomposer-musique-2wiki-popqa")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--corpus", default="wiki")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
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
    run_em = {"A": 0, "B": 0, "C": 0, "old": 0}
    for i, row in enumerate(rows, 1):
        rid = str(row["id"])
        if rid in done:
            continue
        question = row["question"]
        golds = golds_map.get(rid) or (row["gold"] if isinstance(row["gold"], list)
                                       else [row["gold"]])
        stored_steps = (row.get("decomp") or {}).get("steps") or []
        rec = {"ds": args.ds, "id": rid, "question": question,
               "replay_em": int(row.get("replay_em", 0)),
               "replay_pred": row.get("replay_pred") or ""}
        if not stored_steps:
            # no decomposition -> nothing for the prompt fix to act on
            for cfg in ("A", "B", "C"):
                rec[cfg] = {"em": rec["replay_em"], "f1": token_f1(
                    rec["replay_pred"], golds), "pred": rec["replay_pred"],
                    "choice": row.get("replay_choice"), "unchanged": True}
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            continue

        direct_hits = retriever.topk(question, k=TOP_K, title_field=False)

        # ---- config A/C step loop (orig-question-grounded prompts) ----
        answers: dict[int, str] = {}
        new_steps, new_evidence = [], []
        for s in stored_steps:
            resolved = _substitute(s["raw_sub_q"], answers)
            res = answer_with_entropy(model, tok, resolved,
                                      orig_question=question)
            sig = res.max_ent
            retrieved = sig >= TAU
            final_ans = res.text
            post_sig = -1.0
            if retrieved:
                subq_hits = retriever.topk(resolved, k=TOP_K, title_field=True)
                ctx = format_context(_merge_hits(subq_hits, direct_hits))
                post = answer_with_entropy(model, tok, resolved, context=ctx,
                                           orig_question=question)
                final_ans = post.text
                post_sig = post.max_ent
                snip = top1_snippet(subq_hits)
                if snip:
                    new_evidence.append(snip)
            answers[s["step_id"]] = final_ans
            new_steps.append({
                "step_id": s["step_id"], "raw_sub_q": s["raw_sub_q"],
                "sub_q": resolved, "tentative": res.text,
                "final_answer": final_ans, "retrieved": retrieved,
                "max_ent": sig, "post_signal_value": post_sig,
            })

        # ---- stored-step evidence (config B) ----
        stored_evidence = []
        for s in stored_steps:
            if s.get("retrieved"):
                snip = top1_snippet(
                    retriever.topk(s["sub_q"], k=TOP_K, title_field=True))
                if snip:
                    stored_evidence.append(snip)

        ns_new = [SimpleNamespace(sub_q=s["sub_q"], final_answer=s["final_answer"])
                  for s in new_steps]
        ns_old = [SimpleNamespace(sub_q=s["sub_q"], final_answer=s["final_answer"])
                  for s in stored_steps]
        with model.disable_adapter():
            synth_a = synthesize(model, tok, question, ns_new, None)
            synth_c = synthesize(model, tok, question, ns_new, new_evidence)
            synth_b = synthesize(model, tok, question, ns_old, stored_evidence)

        rec["A"] = evaluate(row, golds, new_steps, synth_a)
        rec["C"] = evaluate(row, golds, new_steps, synth_c)
        rec["B"] = evaluate(row, golds, stored_steps, synth_b)
        rec["new_steps"] = new_steps
        rec["step_changed"] = sum(
            int((n["final_answer"] or "") != (o["final_answer"] or ""))
            for n, o in zip(new_steps, stored_steps))
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
        n_done += 1
        for cfg in ("A", "B", "C"):
            run_em[cfg] += rec[cfg]["em"]
        run_em["old"] += rec["replay_em"]
        if n_done % 25 == 0:
            dt = time.time() - t0
            print(f"[{args.ds} shard {args.shard}] {i}/{len(rows)} "
                  f"({dt/max(n_done,1):.1f}s/q) running EM "
                  f"old={run_em['old']/n_done:.3f} A={run_em['A']/n_done:.3f} "
                  f"B={run_em['B']/n_done:.3f} C={run_em['C']/n_done:.3f}",
                  flush=True)
    out_f.close()
    print(f"done {args.ds} shard {args.shard}: {n_done} questions, "
          f"{(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
