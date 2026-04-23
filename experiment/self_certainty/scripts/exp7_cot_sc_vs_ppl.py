"""
Experiment 7: SC vs PPL under Chain-of-Thought generation.

Hypothesis: SC should win (or at least close the gap with PPL) when the output
is long reasoning, because SC aggregates distribution information over every
token — which matters more when correctness depends on a reasoning chain rather
than a single entity token.

Method: Same as exp4 but with a CoT prompt and longer max_new_tokens.
  - Prompt: "Answer the given question. Always conduct and show your
    step-by-step reasoning first, then give the final answer."
  - SC / PPL are computed over the FULL CoT output (reasoning + final answer).
  - For correctness, we extract the final answer (after "final answer"/
    "answer:" markers, or the last line) and run the exp4 EM/F1 check.

Usage:
  python exp7_cot_sc_vs_ppl.py --dataset 2wiki --max_questions 500
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import json
import logging
import math
import re
import string
import collections

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="Exp7: CoT SC vs PPL")
parser.add_argument("--dataset", type=str, default="2wiki",
                    choices=["nq", "2wiki", "hotpotqa", "musique", "squad", "trivia"])
parser.add_argument("--max_questions", type=int, default=500)
parser.add_argument("--n_quintiles", type=int, default=5)
parser.add_argument("--max_new_tokens", type=int, default=512)
args = parser.parse_args()

DATA_DIR = "/home/S113065528/Adaptive-RAG/processed_data"
DATASET_MAP = {
    "2wiki": "2wikimultihopqa", "nq": "nq", "hotpotqa": "hotpotqa",
    "musique": "musique", "squad": "squad", "trivia": "trivia",
}
DATASET_NAME = DATASET_MAP[args.dataset]
DATA_FILE = os.path.join(DATA_DIR, DATASET_NAME, "test_subsampled.jsonl")
DS_TAG = args.dataset

BASE_DIR = "/home/S113065528/Adaptive-RAG/experiment/self_certainty"
OUTPUT_FILE = os.path.join(BASE_DIR, "results", f"exp7_cot_{DS_TAG}_results.json")
SUMMARY_FILE = os.path.join(BASE_DIR, "results", f"exp7_cot_{DS_TAG}_summary.json")
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
LOG_FILE = os.path.join(BASE_DIR, "logs", f"exp7_cot_{DS_TAG}.log")

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
HF_CACHE = "/home/S113065528/Adaptive-RAG/hf_cache"
HF_TOKEN = os.environ.get("HF_TOKEN", None)
MAX_NEW_TOKENS = args.max_new_tokens

os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
              logging.StreamHandler()],
)

BLUE = "#4C72B0"; ORANGE = "#DD8452"; RED = "#C44E52"


def normalize_answer(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def compute_f1(pred, gold):
    pt, gt = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = collections.Counter(pt) & collections.Counter(gt)
    num = sum(common.values())
    if not pt or not gt:
        return int(pt == gt)
    if num == 0:
        return 0
    p, r = num / len(pt), num / len(gt)
    return 2 * p * r / (p + r)


def check_correct(pred, golds):
    pn = normalize_answer(pred)
    for g in golds:
        if pn == normalize_answer(g):
            return True
        if compute_f1(pred, g) >= 0.5:
            return True
    return False


# Extract final answer from CoT output
FINAL_PATTERNS = [
    r"final answer\s*[:：]\s*(.+?)(?:\n|$)",
    r"\banswer\s*[:：]\s*(.+?)(?:\n|$)",
    r"the answer is\s*(.+?)(?:\.|\n|$)",
    r"so,?\s*(.+?)(?:\.|\n|$)",
]


def extract_final_answer(cot_text):
    text = cot_text.strip()
    for pat in FINAL_PATTERNS:
        m = re.findall(pat, text, flags=re.IGNORECASE)
        if m:
            return m[-1].strip().strip(".").strip()
    # Fallback: last non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else text


def load_data(filepath, max_n=None):
    data = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            q = item["question_text"]
            answers = []
            for a in item.get("answers_objects", []):
                answers.extend(a.get("spans", []))
            if q and answers:
                data.append({"id": item.get("question_id", item.get("id", "")),
                             "question": q, "answers": answers})
            if max_n and len(data) >= max_n:
                break
    return data


def build_prompt(tokenizer, question):
    messages = [
        {"role": "system", "content":
            "Answer the given question. Always conduct and show your "
            "step-by-step reasoning first, then give the final answer. "
            "End with a line: 'Final Answer: <your answer>'."},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate_and_score(model, tokenizer, question):
    V = model.config.vocab_size
    prompt_text = build_prompt(tokenizer, question)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt",
                           add_special_tokens=False).input_ids.to(model.device)
    prompt_len = prompt_ids.shape[1]

    output_ids = model.generate(
        prompt_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    answer_ids = output_ids[0][prompt_len:]
    cot_text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    n_tok = len(answer_ids)

    if n_tok == 0:
        return {"cot": "", "final_answer": "", "sc": 0.0,
                "ppl": float("inf"), "n_tokens": 0}

    full_ids = output_ids[:, :prompt_len + n_tok]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(full_ids).logits

    ans_logits = logits[:, prompt_len - 1: prompt_len + n_tok - 1, :]\
        .squeeze(0).float()
    ans_token_ids = answer_ids.to(model.device)
    log_probs = F.log_softmax(ans_logits, dim=-1)

    token_lp = log_probs.gather(1, ans_token_ids.unsqueeze(1)).squeeze(1)
    avg_nll = -token_lp.mean().item()
    ppl = math.exp(min(avg_nll, 20))

    lp_sum = log_probs.sum(dim=-1)
    V_f = float(V)
    sc_per = -1.0 / V_f * lp_sum - math.log(V_f)
    sc = sc_per.mean().item()

    final = extract_final_answer(cot_text)
    return {"cot": cot_text, "final_answer": final,
            "sc": sc, "ppl": ppl, "n_tokens": n_tok}


def plot_quintiles(results, n_q, tag):
    sorted_sc = sorted(results, key=lambda r: r["sc"])
    sorted_ppl = sorted(results, key=lambda r: r["ppl"])
    n = len(sorted_sc); qs = n // n_q
    sc_acc, ppl_acc, sc_lab, ppl_lab = [], [], [], []
    for q in range(n_q):
        s, e = q * qs, (q * qs + qs if q < n_q - 1 else n)
        sg, pg = sorted_sc[s:e], sorted_ppl[s:e]
        sc_acc.append(sum(r["is_correct"] for r in sg) / len(sg) * 100)
        ppl_acc.append(sum(r["is_correct"] for r in pg) / len(pg) * 100)
        sc_lab.append(f"Q{q+1}\n({sg[0]['sc']:.1f}-{sg[-1]['sc']:.1f})")
        ppl_lab.append(f"Q{q+1}\n({pg[0]['ppl']:.2f}-{pg[-1]['ppl']:.2f})")

    overall = sum(r["is_correct"] for r in results) / n * 100
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Exp 7 (CoT): SC vs PPL Quintiles — {tag.upper()}",
                 fontsize=13, fontweight="bold")
    x = np.arange(n_q)

    for ax, vals, labs, color, title in [
        (axes[0], sc_acc, sc_lab, BLUE, "SC Quintiles (Q1=lowest SC=least confident)"),
        (axes[1], ppl_acc, ppl_lab, ORANGE, "PPL Quintiles (Q1=lowest PPL=most confident)"),
    ]:
        bars = ax.bar(x, vals, color=color, edgecolor="black", width=0.6)
        ax.axhline(overall, color="black", ls="--", label=f"Overall: {overall:.1f}%")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                    f"{v:.1f}%", ha="center", va="bottom", fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=9)
        ax.set_ylabel("Accuracy (%)"); ax.set_title(title)
        ax.set_ylim(0, 100); ax.legend()

    ax = axes[2]; w = 0.3; x2 = np.arange(2)
    labels = ["Least confident\n(SC-Q1 / PPL-Q5)",
              "Most confident\n(SC-Q5 / PPL-Q1)"]
    sc_v = [sc_acc[0], sc_acc[-1]]
    pp_v = [ppl_acc[-1], ppl_acc[0]]
    b1 = ax.bar(x2 - w/2, sc_v, w, label="SC", color=BLUE, edgecolor="black")
    b2 = ax.bar(x2 + w/2, pp_v, w, label="PPL", color=ORANGE, edgecolor="black")
    for b, v in zip(list(b1) + list(b2), sc_v + pp_v):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_xticks(x2); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Accuracy (%)"); ax.set_title("Extreme-quintile comparison")
    ax.set_ylim(0, 100); ax.legend()

    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, f"exp7_cot_quintiles_{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logging.info(f"Saved: {out}"); plt.close(fig)


def main():
    logging.info(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=HF_CACHE, token=HF_TOKEN, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir=HF_CACHE, device_map="cuda",
        torch_dtype=torch.bfloat16, token=HF_TOKEN, local_files_only=True)
    model.eval()

    logging.info(f"Dataset: {args.dataset}   Max Q: {args.max_questions}   "
                 f"max_new_tokens: {MAX_NEW_TOKENS}")
    data = load_data(DATA_FILE, args.max_questions)
    logging.info(f"Loaded {len(data)} questions\n")

    results = []
    for item in tqdm(data, desc="CoT generating"):
        s = generate_and_score(model, tokenizer, item["question"])
        is_correct = check_correct(s["final_answer"], item["answers"])\
            if s["final_answer"] else False
        entry = {
            "id": item["id"], "question": item["question"],
            "ground_truths": item["answers"],
            "cot": s["cot"], "final_answer": s["final_answer"],
            "sc": s["sc"], "ppl": s["ppl"], "n_tokens": s["n_tokens"],
            "is_correct": is_correct,
        }
        results.append(entry)
        logging.info(f"[{item['id']}] {'O' if is_correct else 'X'} "
                     f"SC={s['sc']:.3f} PPL={s['ppl']:.2f} "
                     f"n_tok={s['n_tokens']} final={s['final_answer'][:60]}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logging.info(f"\nResults -> {OUTPUT_FILE}")

    n = len(results)
    acc = sum(r["is_correct"] for r in results) / n * 100
    logging.info(f"Overall accuracy: {acc:.1f}%")

    labels = [1 if r["is_correct"] else 0 for r in results]
    sc_s = [r["sc"] for r in results]
    ppl_s = [-r["ppl"] for r in results]
    if len(set(labels)) > 1:
        auc_sc = roc_auc_score(labels, sc_s)
        auc_ppl = roc_auc_score(labels, ppl_s)
        logging.info(f"AUC — SC: {auc_sc:.4f}  PPL: {auc_ppl:.4f}")
    else:
        auc_sc = auc_ppl = None

    sorted_sc = sorted(results, key=lambda r: r["sc"])
    qs = n // args.n_quintiles
    qsumm = []
    for q in range(args.n_quintiles):
        s, e = q * qs, (q * qs + qs if q < args.n_quintiles - 1 else n)
        g = sorted_sc[s:e]
        a = sum(r["is_correct"] for r in g) / len(g) * 100
        qsumm.append({"quintile": f"Q{q+1}", "n": len(g),
                      "accuracy": round(a, 2),
                      "mean_sc": round(float(np.mean([r["sc"] for r in g])), 4),
                      "sc_range": [round(g[0]["sc"], 4), round(g[-1]["sc"], 4)],
                      "mean_n_tokens": round(float(np.mean([r["n_tokens"] for r in g])), 1)})
        logging.info(f"  Q{q+1}: acc={a:.1f}%  mean_SC={qsumm[-1]['mean_sc']:.2f}  "
                     f"mean_n_tok={qsumm[-1]['mean_n_tokens']}")

    gap = qsumm[-1]["accuracy"] - qsumm[0]["accuracy"]
    logging.info(f"Q5-Q1 gap: {gap:.1f}pp")

    summary = {
        "dataset": args.dataset, "mode": "cot",
        "n_questions": n,
        "mean_n_tokens": round(float(np.mean([r["n_tokens"] for r in results])), 1),
        "overall_accuracy": round(acc, 2),
        "auc_sc": round(auc_sc, 4) if auc_sc else None,
        "auc_ppl": round(auc_ppl, 4) if auc_ppl else None,
        "quintiles": qsumm, "q1_q5_gap": round(gap, 2),
    }
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logging.info(f"Summary -> {SUMMARY_FILE}")

    plot_quintiles(results, args.n_quintiles, DS_TAG)
    logging.info("Done.")


if __name__ == "__main__":
    main()
