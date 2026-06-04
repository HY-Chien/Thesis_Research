"""Quantitatively evaluate a trained LoRA decomposer on a dev set.

Runs the decomposer over every dev question (batched), parses the `#N:`
output back into steps + dependency edges, and scores against gold:

  - parse_ok     : output has >=1 valid `#N:` line with consecutive ids 1..N
  - count_match  : generated step count == gold step count
  - deps_valid   : every #M reference points to a strictly earlier step
  - topo_match   : per-step dependency sets == gold's (count-matched rows only)

Gold dependency edges:
  - MuSiQue: from `#N` placeholders in question_decomposition[*].question
  - 2Wiki  : from `evidences` triples (fact matching an earlier entity = dep)

Usage:
    python eval_decomposer.py --adapter <ckpt> --dev <dev.jsonl> \
        [--limit N] [--batch-size 48]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "/workspace/S113065528/RAG-DDR/models/Llama3-8B-Instruct"
INSTRUCTION = (
    "Decompose the following multi-hop question into an ordered list of "
    "atomic sub-questions. Each sub-question must be answerable in a single "
    "retrieval/reasoning step. When a sub-question depends on the answer of "
    "an earlier one, refer to that step with a #N placeholder (e.g. #1). "
    "Output one sub-question per line, prefixed with #<index>:"
)

STEP_RE = re.compile(r"^#(\d+):\s*(.*)$")
REF_RE = re.compile(r"#(\d+)")


def parse_steps(text: str) -> list[tuple[int, list[int]]]:
    steps = []
    for line in text.splitlines():
        m = STEP_RE.match(line.strip())
        if not m:
            continue
        sid = int(m.group(1))
        deps = sorted({int(x) for x in REF_RE.findall(m.group(2))})
        steps.append((sid, deps))
    return steps


def musique_gold(r: dict) -> list[tuple[int, list[int]]]:
    out = []
    for i, s in enumerate(r["question_decomposition"], 1):
        deps = sorted({int(x) for x in REF_RE.findall(s.get("question", ""))})
        out.append((i, deps))
    return out


def twowiki_gold(r: dict) -> list[tuple[int, list[int]]]:
    entity_at, out = {}, []
    for i, ev in enumerate(r.get("evidences") or [], 1):
        fact = (ev.get("fact") or "").strip()
        dep = entity_at.get(fact.lower())
        out.append((i, [dep] if dep else []))
        ent = (ev.get("entity") or "").strip()
        if ent:
            entity_at[ent.lower()] = i
    return out


def hop_of(r: dict) -> str:
    """MuSiQue: id prefix; 2Wiki: type field."""
    rid = r.get("id", "")
    if "__" in rid:
        return rid.split("__")[0]
    return r.get("type", "?")


def topo(steps):
    return tuple(tuple(d) for _, d in steps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=48)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = [json.loads(l) for l in open(args.dev)]
    if args.limit:
        rows = rows[:args.limit]
    gold_fn = musique_gold if "question_decomposition" in rows[0] else twowiki_gold

    tot = Counter()
    parse_ok = Counter()
    count_match = Counter()
    deps_valid = Counter()
    topo_match = Counter()

    n = len(rows)
    for start in range(0, n, args.batch_size):
        batch = rows[start:start + args.batch_size]
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": INSTRUCTION + "\n" + r["question"]}],
                tokenize=False, add_generation_prompt=True)
            for r in batch
        ]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=256, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
        gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                               skip_special_tokens=True)
        for r, g in zip(batch, gen):
            hop = hop_of(r)
            tot[hop] += 1
            steps = parse_steps(g)
            gold = gold_fn(r)
            ids = [s for s, _ in steps]
            if not (steps and ids == list(range(1, len(steps) + 1))):
                continue
            parse_ok[hop] += 1
            if all(all(1 <= d < sid for d in deps) for sid, deps in steps):
                deps_valid[hop] += 1
            if len(steps) == len(gold):
                count_match[hop] += 1
                if topo(steps) == topo(gold):
                    topo_match[hop] += 1
        print(f"  {min(start + args.batch_size, n)}/{n}", end="\r", flush=True)

    hops = sorted(tot)
    T = sum(tot.values())

    def line(name, c, denom):
        d = sum(denom.values())
        return f"{name:13s}{sum(c.values()):6d}/{d:<6d} = {sum(c.values())/max(d,1):6.1%}"

    print(" " * 30, end="\r")
    print(f"\nadapter: {args.adapter}")
    print(f"dev:     {args.dev}  ({T} questions)\n")
    print(line("parse_ok", parse_ok, tot), "  (of all)")
    print(line("deps_valid", deps_valid, parse_ok), "  (of parsed)")
    print(line("count_match", count_match, parse_ok), "  (of parsed)")
    print(line("topo_match", topo_match, count_match), "  (of count-matched)")
    print(f"\n{'class':20s}{'n':>6s}{'parse%':>9s}{'count%':>9s}{'topo%':>9s}")
    for h in hops:
        print(f"{h:20s}{tot[h]:6d}"
              f"{parse_ok[h]/max(tot[h],1)*100:8.1f}%"
              f"{count_match[h]/max(parse_ok[h],1)*100:8.1f}%"
              f"{topo_match[h]/max(count_match[h],1)*100:8.1f}%")


if __name__ == "__main__":
    main()
