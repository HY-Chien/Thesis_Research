"""LLM-as-judge for confident hallucination subtype classification.

Reads a list of question records (question + gold + 5 sample preds), prompts
an LLM judge to classify each into {HI_correct, B1, B2, B3, Other}, and writes
JSON output with label + reason.

Supports two backends (selected via --model):
  - gpt_oss_20b   (default; OpenAI GPT-OSS-20B, MoE, MXFP4 quantized)
  - qwen3_32b     (Qwen3-32B dense bf16; uses 2-GPU device_map=auto)

Input JSON format: a list of records with keys
  qid, question, golds, preds (list of 5 strings), [stratum, em_rate, n_uniq]

Output JSON format: list of records with
  qid, label, reason, raw_response

Usage:
  conda activate prag
  python judge_subtype.py --model gpt_oss_20b \
      --input step4_validation_30.json \
      --output step4_judge_gptoss_30.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATHS = {
    "gpt_oss_20b": "/workspace/hf_cache/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee",
    "qwen3_32b": "/workspace/hf_cache/hub/models--Qwen--Qwen3-32B/snapshots",
}

VALID_LABELS = {"HI_correct", "B1", "B2", "B3", "Other"}

JUDGE_INSTRUCTION = """You are classifying why a question-answering model succeeds or fails.

The model under test (Llama-3-8B-Instruct) was given a question and produced 5 different sampled answers (temperature=0.7). You will see the question, the gold answer, and the 5 sampled predictions. Classify the question into ONE of these categories:

- HI_correct: At least 4 of 5 predictions match the gold answer SEMANTICALLY (allowing for minor article/punctuation/string differences, e.g. "Richmond River" matches gold "Richmond"; "1922" matches gold "October 1922"). The model knows the answer.

- B1 (knowledge error): The model is confidently wrong with a CONSISTENT or near-consistent wrong answer (>=3 of 5 share the same wrong answer, or all preds cluster semantically around one wrong concept). The model has a fixed but incorrect factual belief. Schema parsing is correct (the answer TYPE is right, just wrong fact).

- B2 (random filling): The model is wrong with DIFFERENT wrong answers across samples (n_uniq is high, e.g. 4-5 different wrong predictions). This indicates the model can't parse the question stably or can't retrieve coherent facts. Often the wrong answers are even of different TYPES (one is a date, another is a name, etc.).

- B3 (role confusion): The model returns an entity that appears in the QUESTION as a CONSTRAINT (e.g., question asks "Who directed the film starring X with Y?" and the model returns Y or X — the constraint entities — instead of the actual director). This is a parse-pattern failure: model treats a question constraint as the answer. Confirm B3 only when (a) the predicted entity is clearly named in the question as a filter/co-actor/co-director/etc., AND (b) it's not coincidental name overlap (e.g., "Argand lamp" named after "Aimé Argand" is NOT B3 — the answer legitimately contains the inventor's name).

- Other: Doesn't fit cleanly. Mid em_rate (e.g., 2/5 or 3/5 correct), genuinely uncertain, or the question is malformed.

Respond with ONLY a single JSON object on one line, no markdown, no explanation outside the JSON:
{"label": "<one of: HI_correct, B1, B2, B3, Other>", "reason": "<one short sentence>"}"""


def build_user_message(rec: dict) -> str:
    """Format the per-question content for the judge."""
    question = rec["question"]
    golds = rec.get("golds", [])
    gold_str = " | ".join(golds) if isinstance(golds, list) else str(golds)
    preds = rec.get("preds", [])
    em_rate = rec.get("em_rate", "?")
    n_uniq = rec.get("n_uniq", "?")

    lines = [
        f"Question: {question}",
        f"Gold answer: {gold_str}",
        f"5 sampled predictions (em_rate={em_rate}, n_uniq={n_uniq}):",
    ]
    for i, p in enumerate(preds, 1):
        lines.append(f"  {i}. {p}")
    return "\n".join(lines)


def parse_label(text: str) -> tuple[str, str]:
    """Extract label + reason from judge response. Robust to extra text."""
    # Try JSON on any line
    json_match = re.search(r"\{[^{}]*\"label\"[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            obj = json.loads(json_match.group(0))
            label = obj.get("label", "").strip()
            reason = obj.get("reason", "").strip()
            if label in VALID_LABELS:
                return label, reason
            # Try to recover misspellings
            for vl in VALID_LABELS:
                if vl.lower() == label.lower():
                    return vl, reason
        except json.JSONDecodeError:
            pass

    # Fallback: regex search for label
    for vl in VALID_LABELS:
        if re.search(rf"\b{re.escape(vl)}\b", text):
            return vl, "(parsed from free-form text)"

    return "PARSE_ERROR", text[:200]


def load_model(model_key: str):
    """Load model + tokenizer. Returns (tok, model, gen_kwargs)."""
    path = MODEL_PATHS[model_key]
    if model_key == "qwen3_32b":
        # Resolve snapshot dir
        snap_dir = Path(path)
        if snap_dir.exists() and snap_dir.is_dir():
            subdirs = list(snap_dir.iterdir())
            if subdirs:
                path = str(subdirs[0])

    print(f"Loading {model_key} from {path}", flush=True)
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    if model_key == "gpt_oss_20b":
        # MXFP4 already in checkpoint — load with trust default
        model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
        )
    elif model_key == "qwen3_32b":
        # 32B bf16 needs 2× L40S
        model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    else:
        raise ValueError(f"unknown model: {model_key}")
    model.eval()

    gen_kwargs = {
        "max_new_tokens": 2048,
        "do_sample": False,
        "temperature": 1.0,
        "pad_token_id": tok.pad_token_id,
    }
    return tok, model, gen_kwargs


@torch.no_grad()
def run_judge(tok, model, gen_kwargs, system_msg: str, user_msg: str, model_key: str) -> str:
    """Run one judge call, return raw decoded text after the prompt."""
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    output = model.generate(**inputs, **gen_kwargs)
    new_tokens = output[0][prompt_len:]
    text = tok.decode(new_tokens, skip_special_tokens=False)

    # GPT-OSS Harmony format: extract content of final channel
    if model_key == "gpt_oss_20b":
        # Look for "<|start|>assistant<|channel|>final<|message|>...<|end|>"
        # or the content after the last "<|channel|>final<|message|>" up to <|end|> or end
        final_match = re.search(
            r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)",
            text,
            re.DOTALL,
        )
        if final_match:
            return final_match.group(1).strip()
        # Fallback 1: search for any JSON object with "label" key in the entire response
        # (could be in analysis channel if model never reached final)
        json_in_analysis = re.search(r"\{[^{}]*\"label\"[^{}]*\}", text, re.DOTALL)
        if json_in_analysis:
            return json_in_analysis.group(0)
        # Fallback 2: strip special tokens
        return tok.decode(new_tokens, skip_special_tokens=True).strip()
    else:
        return tok.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_PATHS.keys()), default="gpt_oss_20b")
    ap.add_argument("--input", required=True, help="JSON file with list of records")
    ap.add_argument("--output", required=True, help="Output JSON path")
    ap.add_argument("--limit", type=int, default=None, help="Process first N records only")
    ap.add_argument("--resume", action="store_true", help="Skip qids already in output")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    data = json.load(in_path.open())
    if args.limit:
        data = data[: args.limit]

    done = {}
    if args.resume and out_path.exists():
        try:
            existing = json.load(out_path.open())
            done = {r["qid"]: r for r in existing if "qid" in r}
            print(f"Resume: {len(done)} qids already done")
        except Exception:
            pass

    tok, model, gen_kwargs = load_model(args.model)

    results = list(done.values()) if done else []
    n_total = len(data)
    label_counts = {}
    t_start = time.time()

    for i, rec in enumerate(data, 1):
        qid = rec.get("qid", f"rec_{i}")
        if qid in done:
            label_counts[done[qid]["label"]] = label_counts.get(done[qid]["label"], 0) + 1
            continue

        user_msg = build_user_message(rec)
        try:
            t0 = time.time()
            raw = run_judge(tok, model, gen_kwargs, JUDGE_INSTRUCTION, user_msg, args.model)
            dt = time.time() - t0
            label, reason = parse_label(raw)
        except Exception as e:
            print(f"[{i}/{n_total}] {qid[:12]} ERROR: {e}", flush=True)
            label, reason, raw, dt = "ERROR", str(e), "", 0.0

        out_rec = {
            "qid": qid,
            "stratum": rec.get("stratum"),
            "em_rate": rec.get("em_rate"),
            "n_uniq": rec.get("n_uniq"),
            "label": label,
            "reason": reason,
            "raw_response": raw,
            "judge_model": args.model,
            "elapsed_s": round(dt, 2),
        }
        results.append(out_rec)
        label_counts[label] = label_counts.get(label, 0) + 1

        elapsed_total = time.time() - t_start
        print(
            f"[{i:4d}/{n_total}] {qid[:12]} stratum={rec.get('stratum','-'):11s} "
            f"label={label:11s} ({dt:.1f}s; total {elapsed_total/60:.1f}m) "
            f"counts={label_counts}",
            flush=True,
        )

        # Periodic checkpoint
        if i % 25 == 0 or i == n_total:
            with out_path.open("w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    with out_path.open("w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Wrote {out_path}")
    print(f"Final label counts: {label_counts}")
    print(f"Total time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
