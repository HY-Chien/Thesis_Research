"""Quick smoke test: run the trained LoRA decomposer on a few dev questions."""
import json
import random
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "/workspace/S113065528/RAG-DDR/models/Llama3-8B-Instruct"
ADAPTER = "/home/S113065528/Research/main_methodology/checkpoints/llama3-8b-decomposer-musique"
DEV = "/home/S113065528/Research/main_methodology/datasets/musique/raw/dev.jsonl"
INSTRUCTION = (
    "Decompose the following multi-hop question into an ordered list of "
    "atomic sub-questions. Each sub-question must be answerable in a single "
    "retrieval/reasoning step. When a sub-question depends on the answer of "
    "an earlier one, refer to that step with a #N placeholder (e.g. #1). "
    "Output one sub-question per line, prefixed with #<index>:"
)

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

rows = [json.loads(l) for l in open(DEV)]
# one question per hop class
by_hop = {}
random.Random(0).shuffle(rows)
for r in rows:
    h = r["id"].split("__")[0]
    by_hop.setdefault(h, r)
samples = [by_hop[h] for h in sorted(by_hop)]

for r in samples:
    msgs = [{"role": "user", "content": INSTRUCTION + "\n" + r["question"]}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=256, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    gold = "\n".join(f"#{i}: {s['question']}"
                     for i, s in enumerate(r["question_decomposition"], 1))
    print("=" * 70)
    print(f"[{r['id'].split('__')[0]}] {r['question']}")
    print("--- generated ---")
    print(gen.strip())
    print("--- gold ---")
    print(gold)
