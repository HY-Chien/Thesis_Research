"""Decompose MuSiQue dev questions with the UNTRAINED base model (no LoRA).

Same base checkpoint, same DECOMP_INSTRUCTION prompt, same greedy decoding as
the SEEK decomposer at inference — the only difference is that no adapter is
loaded. Output feeds `recall_gold_decomp_musique.py --base-decomp` as the
"why train the decomposer" arm (advisor request #5, second half).

Usage (from Research/main_methodology):
  python scripts/decompose_base_musique.py \
      --dev datasets/musique/raw/dev_first1000.jsonl \
      --out results/musique_1000_base_decomp.jsonl
"""
import argparse
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, ".")
from seek.pipeline import DECOMP_INSTRUCTION, STEP_RE

BASE = "/workspace/S113065528/RAG-DDR/models/Llama3-8B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="datasets/musique/raw/dev_first1000.jsonl")
    ap.add_argument("--out", default="results/musique_1000_base_decomp.jsonl")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForCausalLM.from_pretrained(
        a.base, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    recs = [json.loads(l) for l in open(a.dev) if l.strip()]
    if a.limit:
        recs = recs[:a.limit]

    with open(a.out, "w") as fh, torch.no_grad():
        for i, r in enumerate(recs):
            msgs = [{"role": "user",
                     "content": DECOMP_INSTRUCTION + "\n" + r["question"]}]
            prompt = tok.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
            enc = tok(prompt, return_tensors="pt").to(model.device)
            out = model.generate(**enc, max_new_tokens=a.max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0, enc["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip()
            subqs = [m.group(2) for line in text.splitlines()
                     if (m := STEP_RE.match(line.strip()))]
            fh.write(json.dumps({"id": r["id"], "question": r["question"],
                                 "raw_output": text, "subqs": subqs},
                                ensure_ascii=False) + "\n")
            fh.flush()
            if (i + 1) % 50 == 0:
                print(f"  ... {i + 1}/{len(recs)}", flush=True)
    print(f"done -> {a.out}")


if __name__ == "__main__":
    main()
