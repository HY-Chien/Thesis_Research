#!/usr/bin/env python3
"""Generate fixed-chain synthesis outputs for the final tl_sum step-gate ablation.

The input is *only* ``results/traces/{ds}/tlsum_fixA_{ds}_shard{0,1}.jsonl``.
Those
records contain, for every deployed-chain step, both the no-retrieval
``tentative`` answer and the retrieved ``post_pred`` answer, together with the
tentative ``tl_sum`` and retrieved-answer ``post_tl_sum`` signals.

For each threshold, this replay chooses::

    post_pred  if step.tl_sum >= tau
    tentative  otherwise

and regenerates the final synthesis with the same adapter-disabled Llama-3-8B
prompt used by ``rerun_tlsum_fixA.py``.  Later sub-question strings remain the
resolved strings saved by the deployed trace.  This is therefore a
``fixed-chain replay`` rather than a claim that an alternate first-step answer
would leave every later query unchanged in a live rerun.

Synthesis is deduplicated per question: thresholds that produce the same
sequence of selected step answers share one model generation.

Example (two GPUs)::

    CUDA_VISIBLE_DEVICES=0 python scripts/replay_tlsum_fixA_step_gate_synthesis.py \
      --shard 0 --num-shards 2 \
      --out results/generated/tlsum_fixA_step_gate_synth_shard0.jsonl --resume
    CUDA_VISIBLE_DEVICES=1 python scripts/replay_tlsum_fixA_step_gate_synthesis.py \
      --shard 1 --num-shards 2 \
      --out results/generated/tlsum_fixA_step_gate_synth_shard1.jsonl --resume
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe_confidence_battery import BASE
from rerun_prompt_fix import synthesize


ROOT = Path(__file__).resolve().parent.parent
DATASETS = ("hotpotqa", "2wiki", "musique", "popqa")

# The finite values are the pooled step-signal quantiles used by the existing
# sensitivity analysis.  tau=0 retrieves every step because tl_sum >= 0.
DEFAULT_POLICIES: tuple[tuple[str, float], ...] = (
    ("always", 0.0),
    ("tau_0.002305", 0.002305),
    ("deployed", 0.008461),
    ("tau_0.02216", 0.02216),
    ("tau_0.04417", 0.04417),
    ("tau_0.07467", 0.07467),
    ("tau_0.1165", 0.1165),
    ("never", math.inf),
)


def load_rows() -> list[dict]:
    rows: list[dict] = []
    for ds in DATASETS:
        ds_rows: list[dict] = []
        for shard in (0, 1):
            path = (
                ROOT / "results" / "traces" / ds
                / f"tlsum_fixA_{ds}_shard{shard}.jsonl"
            )
            with path.open(encoding="utf-8") as handle:
                ds_rows.extend(json.loads(line) for line in handle if line.strip())
        # IDs make sharding deterministic and independent of file order.
        ds_rows.sort(key=lambda row: str(row["id"]))
        rows.extend(ds_rows)
    return rows


def selected_answers(steps: list[dict], tau: float) -> tuple[str, ...]:
    return tuple(
        (step.get("post_pred") or "")
        if float(step["tl_sum"]) >= tau
        else (step.get("tentative") or "")
        for step in steps
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.shard < args.num_shards:
        raise ValueError("--shard must be in [0, --num-shards)")

    rows = [
        row for index, row in enumerate(load_rows())
        if index % args.num_shards == args.shard
    ]
    if args.limit:
        rows = rows[:args.limit]

    done: set[tuple[str, str]] = set()
    if args.resume and args.out.exists():
        with args.out.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    done.add((row["ds"], str(row["id"])))
                except (json.JSONDecodeError, KeyError):
                    continue

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # The original run called synthesize() inside PeftModel.disable_adapter().
    # Loading the identical base checkpoint directly gives that same model.
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    started = time.time()
    generated = processed = 0
    with args.out.open(mode, encoding="utf-8") as output:
        for index, row in enumerate(rows, 1):
            identity = (row["ds"], str(row["id"]))
            if identity in done:
                continue
            steps = row.get("new_steps") or []
            if not steps:
                raise ValueError(f"{identity}: tlsum_fixA row has no new_steps")

            policy_answers = {
                name: selected_answers(steps, tau)
                for name, tau in DEFAULT_POLICIES
            }
            unique_answers: list[tuple[str, ...]] = []
            for answers in policy_answers.values():
                if answers not in unique_answers:
                    unique_answers.append(answers)

            syntheses: list[str] = []
            for answers in unique_answers:
                synth_steps = [
                    SimpleNamespace(sub_q=step["sub_q"], final_answer=answer)
                    for step, answer in zip(steps, answers)
                ]
                syntheses.append(
                    synthesize(model, tokenizer, row["question"], synth_steps, None)
                )
                generated += 1

            policy_to_synthesis = {
                name: unique_answers.index(answers)
                for name, answers in policy_answers.items()
            }
            record = {
                "ds": row["ds"],
                "id": str(row["id"]),
                "policies": {name: None if math.isinf(tau) else tau for name, tau in DEFAULT_POLICIES},
                "policy_to_synthesis": policy_to_synthesis,
                "syntheses": syntheses,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            processed += 1

            if processed % 25 == 0:
                elapsed = time.time() - started
                print(
                    f"[shard {args.shard}] {index}/{len(rows)} rows; "
                    f"generated={generated}; {elapsed / processed:.2f}s/row",
                    flush=True,
                )

    elapsed = time.time() - started
    print(
        f"done shard={args.shard}: rows={processed}, syntheses={generated}, "
        f"minutes={elapsed / 60:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
