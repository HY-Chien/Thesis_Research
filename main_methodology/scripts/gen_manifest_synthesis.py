"""Generate syntheses for the step-gate manifest (GPU worker).

Each manifest row is a unique (question, answer-vector) whose final synthesis
is not stored anywhere yet. Prompt/decoding replicates the deployed synthesis
exactly (LoRA disabled, greedy) via probe_confidence_battery_decomp.synth_prompt.
Shardable across GPUs with --shard/--num-shards; resume by (id, vec_hash).

Usage:
    python scripts/gen_manifest_synthesis.py --shard 0 --num-shards 2 \
        --out results/step_gate_synth_shard0.jsonl
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
from probe_confidence_battery import BASE, greedy_probe  # noqa: E402
from probe_confidence_battery_decomp import synth_prompt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="results/step_gate_manifest.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--adapter", default="checkpoints/llama3-8b-decomposer-musique-2wiki-popqa")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    # question text lives in the trace; index it once
    questions: dict[str, str] = {}
    for ds in ["2wiki", "hotpotqa", "musique", "popqa"]:
        path = Path(f"results/{ds}_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl")
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            questions[f"{ds}:{r['id']}"] = r["question"]

    jobs = []
    for i, line in enumerate(open(args.manifest, encoding="utf-8")):
        if i % args.num_shards != args.shard:
            continue
        jobs.append(json.loads(line))

    done = set()
    if args.resume and Path(args.out).exists():
        for line in open(args.out, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["id"], r["vec_hash"]))
            except json.JSONDecodeError:
                pass

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    out_f = open(args.out, "a" if args.resume else "w", encoding="utf-8")
    t0 = time.time()
    n = 0
    for j, m in enumerate(jobs, 1):
        if (m["id"], m["vec_hash"]) in done:
            continue
        steps = [{"sub_q": q, "final_answer": a}
                 for q, a in zip(m["sub_qs"], m["answers"])]
        prompt = synth_prompt(tok, questions[f"{m['ds']}:{m['id']}"], steps)
        with model.disable_adapter():
            g = greedy_probe(model, tok, prompt)
        out_f.write(json.dumps({
            "ds": m["ds"], "id": m["id"], "vec_hash": m["vec_hash"],
            "pred": g["pred"],
        }, ensure_ascii=False) + "\n")
        n += 1
        if j % 50 == 0 or j == len(jobs):
            out_f.flush()
            dt = time.time() - t0
            print(f"  [{j}/{len(jobs)}] gen={n} ({dt/max(n,1):.2f}s/synth)",
                  flush=True)
    out_f.close()
    print(f"DONE {args.out} generated={n}", flush=True)


if __name__ == "__main__":
    main()
