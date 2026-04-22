"""
Patch existing result JSONL files with Dynamic SC-RAG outputs.

Reads existing {dataset}_{model_tag}.jsonl (which has no-RAG and single-RAG
already computed), runs token-by-token dynamic sentence-level SC-triggered RAG
on each question, adds dyn_* fields, writes updated JSONL + summary.

Usage:
  python run_dynamic_patch.py --dataset hotpotqa
  python run_dynamic_patch.py --dataset 2wikimultihopqa
  python run_dynamic_patch.py --dataset musique
  python run_dynamic_patch.py --dataset popqa
"""

import argparse
import json
import logging
import math
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
from dataset_eval import DATASET_CLASSES, build_fewshot_prompt, extract_answer, score
from retriever_bm25 import BM25Retriever

# --- CLI ------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, choices=list(DATASET_CLASSES))
parser.add_argument("--model_id", default="meta-llama/Meta-Llama-3-8B-Instruct")
parser.add_argument("--model_tag", default="meta-llama-3-8b-instruct")
parser.add_argument("--hf_cache", default="/home/S113065528/Adaptive-RAG/hf_cache")
parser.add_argument("--topk", type=int, default=3)
parser.add_argument("--max_new_tokens", type=int, default=100)
parser.add_argument("--n_shots", type=int, default=8)
parser.add_argument("--es_index", default="wiki")
parser.add_argument("--device", default="cuda")
parser.add_argument("--output_dir",
                    default="/home/S113065528/Research/deeprag_compare/results")
parser.add_argument("--log_dir",
                    default="/home/S113065528/Research/deeprag_compare/logs")
# Dynamic RAG settings
parser.add_argument("--dyn_sc_threshold", type=float, default=None)
parser.add_argument("--dyn_drop_threshold", type=float, default=1.5)
parser.add_argument("--dyn_trigger_mode", default="either",
                    choices=["absolute", "drop", "either", "both"])
parser.add_argument("--dyn_max_retrievals", type=int, default=2)
parser.add_argument("--dyn_min_sent_tokens", type=int, default=4)
args = parser.parse_args()

MODEL_TAG = args.model_tag
os.makedirs(args.log_dir, exist_ok=True)

IN_FILE  = os.path.join(args.output_dir, f"{args.dataset}_{MODEL_TAG}.jsonl")
OUT_FILE = IN_FILE   # overwrite in place
SUMMARY_FILE = os.path.join(args.output_dir, f"{args.dataset}_{MODEL_TAG}_summary.json")
LOG_FILE = os.path.join(args.log_dir, f"{args.dataset}_{MODEL_TAG}_dynamic_patch.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
              logging.StreamHandler()],
)

# --- Sentence boundary detector -------------------------------------------

_SENT_END_RE = re.compile(r'[.!?]\s*$|\n\s*$')


# --- Model loading --------------------------------------------------------

def load_model(model_id, cache_dir, device):
    token = os.environ.get("HF_TOKEN")
    try:
        tok = AutoTokenizer.from_pretrained(
            model_id, cache_dir=cache_dir, local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, token=token)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    try:
        m = AutoModelForCausalLM.from_pretrained(
            model_id, cache_dir=cache_dir, device_map=device,
            torch_dtype=torch.bfloat16, local_files_only=True)
    except Exception:
        m = AutoModelForCausalLM.from_pretrained(
            model_id, cache_dir=cache_dir, device_map=device,
            torch_dtype=torch.bfloat16, token=token)
    m.eval()
    return tok, m


# --- Prompt builders ------------------------------------------------------

def build_no_rag_prompt(ds_cls, fewshot_prefix, question):
    return fewshot_prefix + ds_cls.test_input(question) + " "


def build_rag_prompt(ds_cls, fewshot_prefix, question, docs_text):
    return (fewshot_prefix
            + "Context (retrieved from Wikipedia):\n" + docs_text + "\n\n"
            + ds_cls.test_input(question) + " ")


# --- regenerate_answer fallback -------------------------------------------

@torch.no_grad()
def regenerate_answer(model, tok, prompt, cot_text, max_new=20):
    for cut in ("Question:", "\n\n", "Note:"):
        pos = cot_text.find(cut)
        if pos > 0:
            cot_text = cot_text[:pos]
    cot_text = cot_text.rstrip()
    cont = prompt + cot_text + " So the answer is "
    ids = tok(cont, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    return cot_text + " So the answer is " + gen.strip()


# --- Dynamic SC-RAG -------------------------------------------------------

@torch.no_grad()
def generate_dynamic_rag(model, tok, retriever, ds_cls, fewshot_prefix, question,
                          max_new, topk, sc_threshold, drop_threshold, trigger_mode,
                          max_retrievals, min_sent_tokens):
    V = float(model.config.vocab_size)
    log_V = math.log(V)
    device = model.device

    def _make_prompt(docs_txt=None):
        if docs_txt:
            return build_rag_prompt(ds_cls, fewshot_prefix, question, docs_txt)
        return build_no_rag_prompt(ds_cls, fewshot_prefix, question)

    prompt_ids = tok(_make_prompt(), return_tensors="pt",
                     add_special_tokens=False).input_ids.to(device)
    out = model(prompt_ids, use_cache=True)
    past_kv = out.past_key_values
    last_logits = out.logits[0, -1, :].float()

    generated_ids = []
    sc_per_token = []
    sentence_scs = []
    trigger_events = []
    current_sent_start = 0
    n_retrievals = 0

    def _pick(logits):
        lp = F.log_softmax(logits, dim=-1)
        return int(logits.argmax()), (-lp.sum() / V - log_V).item()

    def _is_sent_end(recent):
        return bool(_SENT_END_RE.search(recent))

    def _should_trigger(sent_sc, history):
        abs_fire  = sc_threshold is not None and sent_sc < sc_threshold
        drop_fire = False
        if drop_threshold is not None and len(history) >= 2:
            drop_fire = (float(np.mean(history[-2:])) - sent_sc) >= drop_threshold
        if trigger_mode == "absolute": return abs_fire
        if trigger_mode == "drop":    return drop_fire
        if trigger_mode == "both":    return abs_fire and drop_fire
        return abs_fire or drop_fire

    step = 0
    while step < max_new:
        next_id, tok_sc = _pick(last_logits)
        if next_id == tok.eos_token_id:
            break
        generated_ids.append(next_id)
        sc_per_token.append(tok_sc)
        step += 1

        look_back = min(10, len(generated_ids))
        recent_text = tok.decode(generated_ids[-look_back:], skip_special_tokens=True)
        n_in_sent = step - current_sent_start

        if _is_sent_end(recent_text) and n_in_sent >= min_sent_tokens:
            sent_scs = sc_per_token[current_sent_start:]
            if sent_scs:
                mean_sc = float(np.mean(sent_scs))
                sentence_scs.append(mean_sc)

                if n_retrievals < max_retrievals and _should_trigger(mean_sc, sentence_scs[:-1]):
                    partial_cot = tok.decode(generated_ids, skip_special_tokens=True)
                    query = question + " " + partial_cot[-300:]
                    docs = retriever.search(query, topk=topk)
                    if docs:
                        docs_txt = BM25Retriever.format_context(docs)
                        new_prompt_ids = tok(_make_prompt(docs_txt), return_tensors="pt",
                                             add_special_tokens=False).input_ids.to(device)
                        partial_ids = torch.tensor([generated_ids], device=device)
                        full_ids = torch.cat([new_prompt_ids, partial_ids], dim=1)
                        out2 = model(full_ids, use_cache=True)
                        past_kv = out2.past_key_values
                        last_logits = out2.logits[0, -1, :].float()
                        n_retrievals += 1
                        trigger_events.append({
                            "step": step, "sent_sc": mean_sc,
                            "titles": [d["title"] for d in docs],
                        })
                        current_sent_start = step
                        continue
            current_sent_start = step

        new_input = torch.tensor([[next_id]], device=device)
        out = model(new_input, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        last_logits = out.logits[0, -1, :].float()

    text = tok.decode(generated_ids, skip_special_tokens=True).strip()
    return {
        "text": text,
        "sc": float(np.mean(sc_per_token)) if sc_per_token else 0.0,
        "n_tokens": len(generated_ids),
        "n_retrievals": n_retrievals,
        "sentence_scs": sentence_scs,
        "trigger_events": trigger_events,
    }


# --- Summary (mirrors run_experiment.py summarize()) ----------------------

def summarize(records, dataset_tag):
    n = len(records)

    def agg(em_f, f1_f, r_rate):
        return {
            "em": sum(r[em_f] for r in records) / n,
            "f1": sum(r[f1_f] for r in records) / n,
            "retrieve_rate": r_rate,
        }

    dyn_retrieve_rate  = sum(1 for r in records if r.get("dyn_n_retrievals", 0) > 0) / n
    dyn_mean_retrievals = sum(r.get("dyn_n_retrievals", 0) for r in records) / n

    methods = {
        "no_rag":      agg("nr_em",  "nr_f1",  0.0),
        "single_rag":  agg("rag_em", "rag_f1", 1.0),
        "dynamic_rag": {**agg("dyn_em", "dyn_f1", dyn_retrieve_rate),
                        "mean_retrievals": dyn_mean_retrievals},
    }

    sc_arr = sorted(r["nr_sc"] for r in records)
    for rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        tau = sc_arr[max(0, int(rate * n) - 1)]
        em_s = f1_s = rc = 0
        for r in records:
            if r["nr_sc"] >= tau:
                em_s += r["nr_em"]; f1_s += r["nr_f1"]
            else:
                em_s += r["rag_em"]; f1_s += r["rag_f1"]; rc += 1
        methods[f"sc_gated_rate{int(rate*100):02d}"] = {
            "em": em_s / n, "f1": f1_s / n,
            "retrieve_rate": rc / n,
            "sc_threshold": tau,
        }

    return {
        "dataset": dataset_tag,
        "model": args.model_id,
        "n_questions": n,
        "max_new_tokens": args.max_new_tokens,
        "topk": args.topk,
        "n_shots": args.n_shots,
        "dyn_config": {
            "sc_threshold":    args.dyn_sc_threshold,
            "drop_threshold":  args.dyn_drop_threshold,
            "trigger_mode":    args.dyn_trigger_mode,
            "max_retrievals":  args.dyn_max_retrievals,
        },
        "methods": methods,
    }


# --- Main -----------------------------------------------------------------

def main():
    if not os.path.exists(IN_FILE):
        logging.error(f"Input file not found: {IN_FILE}")
        sys.exit(1)

    logging.info(f"Loading existing records from {IN_FILE}")
    records = [json.loads(l) for l in open(IN_FILE) if l.strip()]
    logging.info(f"  {len(records)} records")

    ds_cls = DATASET_CLASSES[args.dataset]
    fewshot = build_fewshot_prompt(ds_cls, args.n_shots)

    logging.info(f"Loading model {args.model_id} ...")
    tok, model = load_model(args.model_id, args.hf_cache, args.device)

    retriever = BM25Retriever(index=args.es_index)
    _ = retriever.search("test", topk=1)
    logging.info("Retriever OK.")

    logging.info(f"Dynamic RAG config: drop_threshold={args.dyn_drop_threshold} "
                 f"sc_threshold={args.dyn_sc_threshold} "
                 f"mode={args.dyn_trigger_mode} max_retrievals={args.dyn_max_retrievals}")

    t0 = time.time()
    n_triggered_total = 0

    for rec in tqdm(records, desc=f"Dynamic RAG [{args.dataset}]"):
        golds = rec["golds"]

        g_dyn = generate_dynamic_rag(
            model, tok, retriever, ds_cls, fewshot, rec["question"],
            max_new=args.max_new_tokens, topk=args.topk,
            sc_threshold=args.dyn_sc_threshold,
            drop_threshold=args.dyn_drop_threshold,
            trigger_mode=args.dyn_trigger_mode,
            max_retrievals=args.dyn_max_retrievals,
            min_sent_tokens=args.dyn_min_sent_tokens,
        )
        dyn_text = g_dyn["text"]
        if "the answer is" not in dyn_text:
            p_fallback = build_no_rag_prompt(ds_cls, fewshot, rec["question"])
            dyn_text = regenerate_answer(model, tok, p_fallback, g_dyn["text"])

        pred_dyn = extract_answer(dyn_text)
        sc_dyn = score(pred_dyn, golds)

        rec["dyn_text"]          = dyn_text
        rec["dyn_pred"]          = pred_dyn
        rec["dyn_sc"]            = g_dyn["sc"]
        rec["dyn_n_tok"]         = g_dyn["n_tokens"]
        rec["dyn_n_retrievals"]  = g_dyn["n_retrievals"]
        rec["dyn_sentence_scs"]  = g_dyn["sentence_scs"]
        rec["dyn_em"]            = sc_dyn["em"]
        rec["dyn_f1"]            = sc_dyn["f1"]

        if g_dyn["n_retrievals"] > 0:
            n_triggered_total += 1

    elapsed = time.time() - t0
    logging.info(f"Done. Elapsed={elapsed/60:.1f} min | "
                 f"Questions with retrieval: {n_triggered_total}/{len(records)}")

    # Save updated records (overwrite original)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logging.info(f"Updated records -> {OUT_FILE}")

    # Recompute summary
    summary = summarize(records, args.dataset)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logging.info(f"Updated summary -> {SUMMARY_FILE}")

    logging.info("\n=== Methods ===")
    for k, v in summary["methods"].items():
        logging.info(f"  {k:<22s}  EM={v['em']*100:5.2f}  F1={v['f1']*100:5.2f}  "
                     f"Retrieve={v['retrieve_rate']:.2f}")


if __name__ == "__main__":
    main()
