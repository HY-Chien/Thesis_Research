"""Convert a gold SFT jsonl into LLaMA-Factory alpaca-format training data.

The SEEK decomposer learns: complex multi-hop question -> ordered list of
atomic sub-questions. Dependencies are expressed inline with MuSiQue `#N`
placeholders (e.g. "When was #1 founded?"). The decomposer does NOT predict
sub-answers — those come later from SEEK's entropy-probing step.

Input  : musique_gold_sft_4000.jsonl  (rows carry `gold_decomposition`)
Output : a JSON array of {instruction, input, output} for LLaMA-Factory.

Usage:
    python build_lf_dataset.py <gold_sft.jsonl> --out <path.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

INSTRUCTION = (
    "Decompose the following multi-hop question into an ordered list of "
    "atomic sub-questions. Each sub-question must be answerable in a single "
    "retrieval/reasoning step. When a sub-question depends on the answer of "
    "an earlier one, refer to that step with a #N placeholder (e.g. #1). "
    "Output one sub-question per line, prefixed with #<index>:"
)


def render_output(gold_decomposition: list[dict]) -> str:
    """Render gold steps as `#1: <sub_question>` lines (no answers)."""
    return "\n".join(
        f"#{s['step_id']}: {s['sub_question']}" for s in gold_decomposition
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="gold sft jsonl with gold_decomposition field")
    ap.add_argument("--out", required=True, help="output .json (alpaca array)")
    args = ap.parse_args()

    examples = []
    with open(args.input) as f:
        for line in f:
            r = json.loads(line)
            examples.append({
                "instruction": INSTRUCTION,
                "input": r["question"],
                "output": render_output(r["gold_decomposition"]),
            })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        json.dump(examples, fout, ensure_ascii=False, indent=1)

    print(f"output: {out_path}  ({len(examples)} examples)")
    print("\n--- example[0] ---")
    print("instruction:", examples[0]["instruction"][:80], "...")
    print("input:", examples[0]["input"])
    print("output:")
    print(examples[0]["output"])


if __name__ == "__main__":
    main()
