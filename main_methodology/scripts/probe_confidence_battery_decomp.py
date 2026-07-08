"""Confidence-signal battery probe on the DECOMP path's final synthesis step.

The deployed decomp answer = _synthesize(question, steps) run with the LoRA
adapter DISABLED, then _rescue_value_answer post-fix. The synthesis prompt is
fully determined by the saved trace (steps' sub_q + final_answer), so this
probe needs NO retrieval:

  1. greedy synthesis regen with output_scores  -> per-token stats
  2. N sampled syntheses                        -> consistency signals
  3. teacher-forced candidate scores            -> answer-level posterior
  4. teacher-forced answer under the BARE question (no sub-answers)
     -> derivation support gain (delta_ctx analogue)
  5. P(True) with sub-answers as context / without

Labels: EM of the rescued regen answer (should match stored decomp_em) and EM
of the raw synthesis output.

Usage:
    python scripts/probe_confidence_battery_decomp.py \
        --trace results/2wiki_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl \
        --dev datasets/2wiki/raw/dev_first1000.jsonl \
        --out results/conf_battery_decomp_2wiki.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from seek.pipeline import SYNTHESIZE_INSTRUCTION, _rescue_value_answer  # noqa: E402
from seek.answerer import ANSWER_INSTRUCTION  # noqa: E402
from seek.verifier import exact_match, token_f1  # noqa: E402
from probe_confidence_battery import (  # noqa: E402
    BASE, greedy_probe, sample_probe, teacher_force_score, p_true_probe)


def synth_prompt(tok, question: str, steps: list[dict]) -> str:
    sub = "\n".join(f"- {s['sub_q']}  ->  {s['final_answer']}" for s in steps)
    body = f"Sub-answers:\n{sub}\n\nQuestion: {question}\nAnswer:"
    msgs = [{"role": "user", "content": f"{SYNTHESIZE_INSTRUCTION}\n{body}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def bare_prompt(tok, question: str) -> str:
    msgs = [{"role": "user", "content": f"{ANSWER_INSTRUCTION}\n\n{question}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default="checkpoints/llama3-8b-decomposer-musique-2wiki-popqa")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--n-sample", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    aliases = {}
    for line in open(args.dev, encoding="utf-8"):
        r = json.loads(line)
        aliases[str(r.get("id"))] = r.get("answer_aliases") or []

    rows = [json.loads(l) for l in open(args.trace, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]

    done = set()
    if args.resume and Path(args.out).exists():
        for line in open(args.out, encoding="utf-8"):
            try:
                done.add(str(json.loads(line).get("id")))
            except json.JSONDecodeError:
                pass
    out_f = open(args.out, "a" if args.resume else "w", encoding="utf-8")

    t0 = time.time()
    n_done = n_match = n_skip = 0
    for i, tr in enumerate(rows, 1):
        rid = str(tr.get("id"))
        if rid in done:
            continue
        steps = (tr.get("decomp") or {}).get("steps") or []
        if not steps:
            n_skip += 1
            continue
        question = tr["question"]
        golds = [tr["gold"]] + aliases.get(rid, [])
        prompt = synth_prompt(tok, question, steps)

        # synthesis runs with the decomposer LoRA DISABLED (pipeline.py)
        with model.disable_adapter():
            g = greedy_probe(model, tok, prompt)
            raw_pred = g["pred"]
            samples = sample_probe(model, tok, prompt, args.n_sample, args.temperature)
            cand_texts: list[str] = []
            for t in [raw_pred] + [s["text"] for s in samples]:
                t = (t or "").strip()
                if t and t not in cand_texts:
                    cand_texts.append(t)
            cand_lps = teacher_force_score(model, tok, prompt, cand_texts)
            ctx_subans = "\n".join(f"- {s['sub_q']}  ->  {s['final_answer']}" for s in steps)
            ptrue_ctx = p_true_probe(model, tok, question, raw_pred, ctx_subans) if raw_pred else 0.5
            ptrue_noctx = p_true_probe(model, tok, question, raw_pred, None) if raw_pred else 0.5
            noctx_lp = (teacher_force_score(model, tok, bare_prompt(tok, question), [raw_pred])[0]
                        if raw_pred else 0.0)

        # deployed post-fix: value->entity rescue on the raw synthesis output
        step_ns = [SimpleNamespace(sub_q=s.get("sub_q") or "",
                                   final_answer=s.get("final_answer") or "")
                   for s in steps]
        rescued = _rescue_value_answer(question, raw_pred, step_ns)
        em_raw = int(exact_match(raw_pred, golds))
        em_rescued = int(exact_match(rescued, golds))

        n_done += 1
        n_match += int(rescued == tr.get("decomp_pred"))
        rec = {
            "id": rid,
            "question": question,
            "gold": tr["gold"],
            "aliases": aliases.get(rid, []),
            "pred": raw_pred,
            "rescued_pred": rescued,
            "em": em_raw,
            "em_rescued": em_rescued,
            "f1": round(token_f1(raw_pred, golds), 4),
            "stored_pred": tr.get("decomp_pred"),
            "stored_decomp_em": tr.get("decomp_em"),
            "n_steps": len(steps),
            "step_max_ents": [s.get("max_ent") for s in steps],
            "tokens": g["tokens"],
            "samples": samples,
            "cand_texts": cand_texts,
            "cand_mean_lps": [round(x, 6) for x in cand_lps],
            "noctx_mean_lp": round(noctx_lp, 6),
            "p_true_ctx": round(ptrue_ctx, 6),
            "p_true_noctx": round(ptrue_noctx, 6),
        }
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if i % 20 == 0 or i == len(rows):
            out_f.flush()
            dt = time.time() - t0
            print(f"  [{i}/{len(rows)}] match_stored={n_match}/{n_done} skip={n_skip} "
                  f"({dt/max(n_done,1):.2f}s/q)", flush=True)
    out_f.close()
    print(f"DONE {args.out} rows={n_done} rescued-pred match={n_match}/{n_done} skip={n_skip}",
          flush=True)


if __name__ == "__main__":
    main()
