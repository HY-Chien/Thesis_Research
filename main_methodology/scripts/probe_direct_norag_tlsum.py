"""Record tail-leakage statistics for the direct, no-retrieval candidate.

The legacy direct-gate probe stored maximum entropy but not top-five mass.
This script regenerates the same greedy no-context answer and stores the
per-token probability summaries required to compute ``tl_sum``.  The output
is checked against ``{ds}_1000_direct_norag.jsonl`` question by question.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from probe_confidence_battery import BASE, _prompt, greedy_probe  # noqa: E402


DATASETS = {
    "hotpotqa": ROOT / "datasets/hotpotqa/raw/dev_first1000.jsonl",
    "2wiki": ROOT / "datasets/2wiki/raw/dev_first1000.jsonl",
    "musique": ROOT / "datasets/musique/raw/dev_first1000.jsonl",
    "popqa": ROOT / "datasets/popqa/raw/dev.jsonl",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=DATASETS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default="checkpoints/llama3-8b-decomposer-musique-2wiki-popqa")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(DATASETS[args.ds], encoding="utf-8")]
    rows = rows[:1000]
    rows = [row for i, row in enumerate(rows) if i % args.num_shards == args.shard]
    legacy = {
        str(row["id"]): row
        for row in map(json.loads, open(ROOT / f"results/{args.ds}_1000_direct_norag.jsonl", encoding="utf-8"))
    }

    done: set[str] = set()
    out_path = Path(args.out)
    if args.resume and out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            try:
                done.add(str(json.loads(line)["id"]))
            except json.JSONDecodeError:
                pass

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    out_f = open(out_path, "a" if args.resume else "w", encoding="utf-8")
    t0 = time.time()
    generated = matched = 0
    for i, row in enumerate(rows, 1):
        rid = str(row["id"])
        if rid in done:
            continue
        probe = greedy_probe(model, tok, _prompt(tok, row["question"], None))
        old_pred = (legacy.get(rid) or {}).get("pred", "")
        is_match = probe["pred"].strip() == old_pred.strip()
        rec = {
            "ds": args.ds,
            "id": rid,
            "question": row["question"],
            "pred": probe["pred"],
            "legacy_pred": old_pred,
            "pred_match": is_match,
            "tokens": probe["tokens"],
        }
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
        generated += 1
        matched += int(is_match)
        if generated % 25 == 0:
            elapsed = time.time() - t0
            print(f"[{args.ds} shard {args.shard}] {i}/{len(rows)} "
                  f"match={matched}/{generated} {elapsed/generated:.2f}s/q", flush=True)
    out_f.close()
    print(f"done {args.ds} shard {args.shard}: generated={generated}, "
          f"prediction matches={matched}/{generated}")


if __name__ == "__main__":
    main()
