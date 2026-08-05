"""Build PopQA 1-step SFT samples for the decomposer.

Each PopQA question is a single KG triple (subj, prop, obj). The natural
"decomposition" is therefore exactly one step:
    #1: <subj> >> <prop>

This teaches the decomposer that 1 step is a valid output when the question is
already atomic — closing the over-decomposition gap that hurts PopQA dev EM.

We hold out the 1000 dev qids used for evaluation.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

INSTRUCTION = (
    "Decompose the following multi-hop question into an ordered list of "
    "atomic sub-questions. Each sub-question must be answerable in a single "
    "retrieval/reasoning step. When a sub-question depends on the answer of "
    "an earlier one, refer to that step with a #N placeholder (e.g. #1). "
    "Output one sub-question per line, prefixed with #<index>:"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent.parent
    ap.add_argument("--dev", default=root / "datasets/popqa/raw/dev.jsonl",
                    type=Path)
    ap.add_argument("--out", default=root / "datasets/popqa/gold/popqa_decomp_lf.json",
                    type=Path)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dev_ids = set()
    with open(args.dev) as f:
        for line in f:
            dev_ids.add(str(json.loads(line)["id"]))
    print(f"holdout dev qids: {len(dev_ids)}")

    ds = load_dataset("akariasai/PopQA")["test"]
    print(f"loaded PopQA: {len(ds)} rows")

    pool = [r for r in ds if str(r["id"]) not in dev_ids]
    print(f"candidate pool after dev holdout: {len(pool)}")

    rng = random.Random(args.seed)
    rng.shuffle(pool)
    sampled = pool[: args.n]
    print(f"sampled {len(sampled)} rows")

    out = []
    for r in sampled:
        out.append({
            "instruction": INSTRUCTION,
            "input": r["question"],
            "output": f"#1: {r['subj']} >> {r['prop']}",
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(out)} samples -> {args.out}")
    print("first 3:")
    for x in out[:3]:
        print(json.dumps(x, ensure_ascii=False))


if __name__ == "__main__":
    main()
