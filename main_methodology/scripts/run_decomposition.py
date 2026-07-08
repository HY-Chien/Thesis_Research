"""Run OSS sub-question decomposition over a dataset's raw split (vLLM backend).

Portable: pass --model-path pointing at any local HF checkpoint. Use
--oss-harmony for GPT-OSS models (extracts the Harmony `final` channel).

vLLM gives real tensor parallelism across both GPUs plus batched decoding —
roughly an order of magnitude faster than the old transformers `device_map`
path, which serialised a MoE model onto one GPU at a time.

Examples
--------
# Local smoke test with GPT-OSS-20B (first 10 rows):
python run_decomposition.py \
    --dataset musique --split train \
    --model-path /workspace/hf_cache/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee \
    --oss-harmony --limit 10 \
    --output ../datasets/musique/oss-20b/oss_smoketest.jsonl

# Full run:
python run_decomposition.py \
    --dataset hotpotqa --split train \
    --model-path /workspace/hf_cache/hub/models--openai--gpt-oss-20b/snapshots/<snap> \
    --oss-harmony \
    --output ../datasets/hotpotqa/oss-20b/train_with_decomposition.jsonl

Output rows = raw row + {generated_decomposition, raw_decomposition_output,
decomposition_error}, matching the schema of the existing
selected_2048_with_oss120b_decomposition.jsonl file.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Allow running from any cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from decompose_prompt import build_prompt, parse_output  # noqa: E402

DATASETS_ROOT = Path(__file__).resolve().parent.parent / "datasets"


def raw_path(dataset: str, split: str) -> Path:
    return DATASETS_ROOT / dataset / "raw" / f"{split}.jsonl"


def load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def harmony_final(text: str) -> str:
    """Extract the GPT-OSS Harmony `final` channel content."""
    m = re.search(
        r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)",
        text,
        re.DOTALL,
    )
    return m.group(1).strip() if m else text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["musique", "hotpotqa", "2wiki"])
    ap.add_argument("--split", required=True, choices=["train", "dev"])
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=None, help="process first N rows")
    ap.add_argument("--oss-harmony", action="store_true",
                    help="extract GPT-OSS Harmony final channel before parsing")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--resume", action="store_true",
                    help="skip ids already present in --output")
    ap.add_argument("--tensor-parallel-size", type=int, default=2)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--no-enforce-eager", action="store_true",
                    help="enable vLLM torch.compile/CUDA graphs "
                         "(default off: prag's torch 2.10 breaks the compile path)")
    ap.add_argument("--batch-size", type=int, default=512,
                    help="rows per vLLM generate call / output flush interval")
    args = ap.parse_args()

    in_path = raw_path(args.dataset, args.split)
    if not in_path.exists():
        sys.exit(f"raw file not found: {in_path}")
    rows = load_rows(in_path)
    if args.limit:
        rows = rows[: args.limit]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if args.resume and out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:  # noqa: BLE001
                    pass
        print(f"resume: {len(done_ids)} rows already done")

    pending = [r for r in rows if r["id"] not in done_ids]
    print(f"{len(pending)} rows to process (of {len(rows)} total)", flush=True)
    if not pending:
        print("nothing to do")
        return

    print(f"Loading model from {args.model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        enforce_eager=not args.no_enforce_eager,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        skip_special_tokens=not args.oss_harmony,  # keep channel markers for harmony
    )

    n_ok = n_err = 0
    t0 = time.time()
    mode = "a" if (args.resume and out_path.exists()) else "w"
    with out_path.open(mode, encoding="utf-8") as fout:
        for start in range(0, len(pending), args.batch_size):
            chunk = pending[start:start + args.batch_size]
            prompts = []
            for row in chunk:
                system_msg, user_msg = build_prompt(row)
                prompts.append(tok.apply_chat_template(
                    [{"role": "system", "content": system_msg},
                     {"role": "user", "content": user_msg}],
                    tokenize=False, add_generation_prompt=True,
                ))
            outs = llm.generate(prompts, sampling)

            for row, out in zip(chunk, outs):
                text = out.outputs[0].text
                content = harmony_final(text) if args.oss_harmony else text

                decomp, err = parse_output(
                    content, n_paragraphs=len(row["paragraphs"]))
                if err:
                    n_err += 1
                else:
                    n_ok += 1

                rec = dict(row)
                rec["generated_decomposition"] = decomp
                rec["raw_decomposition_output"] = content
                rec["decomposition_error"] = err
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            done = start + len(chunk)
            rate = done / (time.time() - t0)
            print(f"  {done}/{len(pending)}  ok={n_ok} err={n_err}  "
                  f"{rate:.2f} rows/s", flush=True)

    print(f"\nDone. ok={n_ok} err={n_err}  ->  {out_path}")


if __name__ == "__main__":
    main()
