"""Synthesis-redo A/B on existing decompose traces (no re-retrieval, no re-decompose).

Reuses each trace's recorded sub-answers and ONLY re-runs the final synthesis
step, comparing three variants on a target question-type subset (default 2Wiki
comparison + bridge_comparison):

  greedy     current pipeline synthesis (do_sample=False) — baseline reproduced
  selfcheck  structured comparison self-check: extract each value, compare,
             output the ENTITY (forces cross-step reasoning)
  sc         self-consistency: sample N synthesises, majority-vote the answer

All variants run with the decomposer LoRA DISABLED (matches pipeline) and then
apply the same value->entity rescue. Isolates "synthesis reasoning" gains from
retrieval — the gold sub-answers are whatever the trace already has.

Usage:
  python scripts/resynth_selfcheck.py \
    --trace results/2wiki_1000_wiki_k8_titlefix.jsonl \
    --dev datasets/2wiki/raw/dev_first1000.jsonl \
    --adapter checkpoints/llama3-8b-decomposer-musique-2wiki-popqa \
    --variants greedy selfcheck sc --sc-n 5 --out results/2wiki_resynth.jsonl
"""
import argparse, json, sys
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seek.pipeline import _synthesize, _rescue_value_answer  # noqa: E402
from seek.verifier import exact_match, token_f1, normalize  # noqa: E402

BASE = "/workspace/S113065528/RAG-DDR/models/Llama3-8B-Instruct"

SELFCHECK_INSTRUCTION = (
    "You answer a question using sub-answers from a decomposition. Reason explicitly:\n"
    "1. For each candidate entity, read off the value being compared (year / number / "
    "length) from the sub-answers.\n"
    "2. Decide which value satisfies the question's criterion "
    "(earliest / latest / older / younger / longer / first / ...).\n"
    "3. The answer is the ENTITY that has that value — NEVER output the value (a date/"
    "number) itself.\n"
    "For a yes/no question, answer Yes or No based on whether the sub-answers agree.\n"
    "For a plain single-fact question, return the minimal answer phrase.\n"
    "Think in one or two short sentences, then end with a final line EXACTLY:\n"
    "Answer: <entity or Yes/No>\n"
)


class _Step:
    __slots__ = ("sub_q", "final_answer")

    def __init__(self, sub_q, final_answer):
        self.sub_q = sub_q or ""
        self.final_answer = final_answer or ""


def _parse_answer_line(text: str) -> str:
    """Take the span after the last 'Answer:' line; fallback = last non-empty line."""
    ans = ""
    for line in text.splitlines():
        s = line.strip()
        if s.lower().startswith("answer:"):
            ans = s.split(":", 1)[1].strip()
    if not ans:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        ans = lines[-1] if lines else text.strip()
    return ans


@torch.no_grad()
def _synth_selfcheck(model, tok, q, steps, max_new_tokens=160):
    sub = "\n".join(f"- {s.sub_q}  ->  {s.final_answer}" for s in steps)
    body = f"Sub-answers:\n{sub}\n\nQuestion: {q}\nReasoning:"
    msgs = [{"role": "user", "content": f"{SELFCHECK_INSTRUCTION}\n{body}"}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return _parse_answer_line(text)


@torch.no_grad()
def _synth_sc(model, tok, q, steps, n=5, temp=0.7):
    votes = []
    for _ in range(n):
        sub = "\n".join(f"- {s.sub_q}  ->  {s.final_answer}" for s in steps)
        # reuse the production synthesis prompt but sample
        from seek.pipeline import SYNTHESIZE_INSTRUCTION
        body = f"Sub-answers:\n{sub}\n\nQuestion: {q}\nAnswer:"
        msgs = [{"role": "user", "content": f"{SYNTHESIZE_INSTRUCTION}\n{body}"}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**enc, max_new_tokens=48, do_sample=True, temperature=temp,
                             top_p=0.95, pad_token_id=tok.eos_token_id)
        votes.append(tok.decode(out[0, enc["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip())
    norm_counts = Counter(normalize(v) for v in votes)
    best_norm, _ = norm_counts.most_common(1)[0]
    for v in votes:                       # return a raw form of the winning answer
        if normalize(v) == best_norm:
            return v
    return votes[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--types", nargs="+", default=["comparison", "bridge_comparison"])
    ap.add_argument("--variants", nargs="+", default=["greedy", "selfcheck"],
                    choices=["greedy", "selfcheck", "sc"])
    ap.add_argument("--sc-n", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    dev = {str(json.loads(l)["id"]): json.loads(l) for l in open(a.dev) if l.strip()}
    trace = [json.loads(l) for l in open(a.trace) if l.strip()]

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                                device_map="cuda")
    model = PeftModel.from_pretrained(base, a.adapter)
    model.eval()

    def syn_ctx():  # FRESH cm each call — peft disable_adapter CMs are single-use
        return model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()

    sums = {v: 0.0 for v in a.variants}
    f1s = {v: 0.0 for v in a.variants}
    rec_sum = 0.0
    n = 0
    out_f = open(a.out, "w", encoding="utf-8") if a.out else None

    for t in trace:
        d = dev.get(str(t["id"]))
        if not d or d.get("type") not in a.types:
            continue
        golds = [d["answer"]] + list(d.get("answer_aliases") or [])
        steps = [_Step(s.get("sub_q"), s.get("final_answer")) for s in t.get("steps", [])]
        if not steps:
            continue
        n += 1
        rec_sum += float(t.get("em") or 0)
        row = {"id": t["id"], "type": d["type"], "gold": d["answer"],
               "recorded_pred": t.get("pred"), "recorded_em": t.get("em")}
        with syn_ctx():
            for v in a.variants:
                if v == "greedy":
                    raw = _synthesize(model, tok, t["question"], steps)
                elif v == "selfcheck":
                    raw = _synth_selfcheck(model, tok, t["question"], steps)
                else:
                    raw = _synth_sc(model, tok, t["question"], steps, n=a.sc_n)
                pred = _rescue_value_answer(t["question"], raw, steps)
                em = float(exact_match(pred, golds))
                sums[v] += em
                f1s[v] += token_f1(pred, golds)
                row[f"{v}_pred"] = pred
                row[f"{v}_em"] = em
        if out_f:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if n % 50 == 0:
            msg = "  ".join(f"{v}={100*sums[v]/n:.1f}" for v in a.variants)
            print(f"  ... {n}  recorded={100*rec_sum/n:.1f}  {msg}", flush=True)
        if a.limit and n >= a.limit:
            break

    if out_f:
        out_f.close()
    print(f"\n=== resynth A/B on {a.trace}  types={a.types}  n={n} ===")
    print(f"  recorded EM (trace as-is): {100*rec_sum/n:5.1f}")
    for v in a.variants:
        print(f"  {v:<10} EM {100*sums[v]/n:5.1f}   F1 {100*f1s[v]/n:5.1f}")


if __name__ == "__main__":
    main()
