"""
Experiment 1: SC vs PPL Stability Under Query Rewriting
========================================================
Goal: Show that SC is more stable than PPL when the same question
      is paraphrased (rewritten / expanded).

Key design: fix the ANSWER, only change the question (prompt).
  - Generate answer A using the ORIGINAL question (greedy)
  - Compute SC/PPL with: original_Q + A, rewritten_Q + A, expansion_Q + A
  - Compare deltas: SC should be more stable than PPL

Compares TWO rewriting methods:
  - Rewritten: simple paraphrase (predict_nq_rewritten_qwen.json)
  - Expansion: verbose expansion (predict_nq_rewritten_qwen_expansion.json)

Output:
  - exp1_stability_results.json
  - figures/exp1_stability_comparison.png
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import json
import logging
import math
import re
import string

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ================= Settings =================
ORIGINAL_FILE  = "/home/S113065528/Adaptive-RAG/experiment/query_rewrite/dataset/predict_nq_original.json"
REWRITTEN_FILE = "/home/S113065528/Adaptive-RAG/experiment/query_rewrite/dataset/predict_nq_rewritten_qwen.json"
EXPANSION_FILE = "/home/S113065528/Adaptive-RAG/experiment/query_rewrite/dataset/predict_nq_rewritten_qwen_expansion.json"

OUTPUT_DIR  = "/home/S113065528/Adaptive-RAG/experiment/self_certainty"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "exp1_stability_results.json")
FIGURE_DIR  = os.path.join(OUTPUT_DIR, "figures")
LOG_FILE    = os.path.join(OUTPUT_DIR, "exp1_stability.log")

MODEL_ID  = "meta-llama/Llama-3.1-8B-Instruct"
HF_CACHE  = "/home/S113065528/Adaptive-RAG/hf_cache"
HF_TOKEN  = os.environ.get("HF_TOKEN", None)
MAX_NEW_TOKENS = 64
# =============================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

BLUE = "#4C72B0"
ORANGE = "#DD8452"
GREEN = "#55A868"
RED = "#C44E52"


def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the question with a short phrase. "
                "Do not explain. Output only the answer."
            ),
        },
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def generate_answer(model, tokenizer, question: str) -> str:
    prompt_text = build_prompt(tokenizer, question)
    input_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(model.device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_ids = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


@torch.no_grad()
def compute_sc_and_ppl(model, tokenizer, question: str, answer: str) -> tuple:
    if not answer:
        return 0.0, float("inf"), 0

    V = model.config.vocab_size
    prompt_text = build_prompt(tokenizer, question)

    prompt_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    ).input_ids
    prompt_len = prompt_ids.shape[1]

    answer_ids = tokenizer(
        answer, return_tensors="pt", add_special_tokens=False
    ).input_ids
    answer_len = answer_ids.shape[1]

    if answer_len == 0:
        return 0.0, float("inf"), 0

    full_ids = torch.cat([prompt_ids, answer_ids], dim=1).to(model.device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(full_ids).logits

    answer_logits = logits[
        :, prompt_len - 1 : prompt_len + answer_len - 1, :
    ].squeeze(0).float()

    answer_token_ids = answer_ids.squeeze(0).to(model.device)
    log_probs_all = F.log_softmax(answer_logits, dim=-1)

    # PPL
    token_log_probs = log_probs_all.gather(
        1, answer_token_ids.unsqueeze(1)
    ).squeeze(1)
    avg_nll = -token_log_probs.mean().item()
    ppl = math.exp(min(avg_nll, 20))

    # SC
    logprob_sum = log_probs_all.sum(dim=-1)
    V_float = float(V)
    sc_per_token = -1.0 / V_float * logprob_sum - math.log(V_float)
    sc = sc_per_token.mean().item()

    return sc, ppl, answer_len


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    den = (sum((x-mx)**2 for x in xs) * sum((y-my)**2 for y in ys)) ** 0.5
    return num / den if den != 0 else 0.0


def plot_comparison(results):
    """
    Plot 2 rows x 3 cols:
      Row 1: Original vs Rewritten  (SC scatter, PPL scatter, bar chart)
      Row 2: Original vs Expansion  (SC scatter, PPL scatter, bar chart)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Exp 1: SC vs PPL — Stability Under Query Rewriting\n(Fixed answer, only prompt changes)",
                 fontsize=14, fontweight="bold")

    row_configs = [
        ("Rewritten", "sc_rewritten", "ppl_rewritten"),
        ("Expansion", "sc_expansion", "ppl_expansion"),
    ]

    for row_idx, (label, sc_key, ppl_key) in enumerate(row_configs):
        sc_orig = [r["sc_orig"] for r in results]
        sc_var  = [r[sc_key] for r in results]
        ppl_orig = [r["ppl_orig"] for r in results]
        ppl_var  = [r[ppl_key] for r in results]

        # Use log scale for PPL to handle outliers
        log_ppl_orig = [math.log(p) if p > 0 else 0 for p in ppl_orig]
        log_ppl_var  = [math.log(p) if p > 0 else 0 for p in ppl_var]

        # SC scatter
        ax = axes[row_idx][0]
        corr_sc = pearson(sc_orig, sc_var)
        ax.scatter(sc_orig, sc_var, alpha=0.3, s=15, color=BLUE)
        lims = [min(min(sc_orig), min(sc_var)), max(max(sc_orig), max(sc_var))]
        ax.plot(lims, lims, "--k", linewidth=1.5, label="y = x")
        ax.set_xlabel("SC (original)", fontsize=10)
        ax.set_ylabel(f"SC ({label.lower()})", fontsize=10)
        ax.set_title(f"Self-Certainty Stability\nOriginal vs {label}", fontsize=11)
        ax.text(0.05, 0.95, f"r = {corr_sc:.3f}", transform=ax.transAxes,
                fontsize=12, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.legend(fontsize=8, loc="lower right")

        # PPL scatter (log scale)
        ax = axes[row_idx][1]
        corr_log_ppl = pearson(log_ppl_orig, log_ppl_var)
        ax.scatter(log_ppl_orig, log_ppl_var, alpha=0.3, s=15, color=ORANGE)
        lims = [min(min(log_ppl_orig), min(log_ppl_var)),
                max(max(log_ppl_orig), max(log_ppl_var))]
        ax.plot(lims, lims, "--k", linewidth=1.5, label="y = x")
        ax.set_xlabel("log(PPL) (original)", fontsize=10)
        ax.set_ylabel(f"log(PPL) ({label.lower()})", fontsize=10)
        ax.set_title(f"Perplexity Stability (log scale)\nOriginal vs {label}", fontsize=11)
        ax.text(0.05, 0.95, f"r = {corr_log_ppl:.3f}", transform=ax.transAxes,
                fontsize=12, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.legend(fontsize=8, loc="lower right")

        # Bar chart: median relative delta
        ax = axes[row_idx][2]
        rel_sc  = sorted([abs(r[sc_key] - r["sc_orig"]) / abs(r["sc_orig"])
                          if r["sc_orig"] != 0 else 0 for r in results])
        rel_ppl = sorted([abs(r[ppl_key] - r["ppl_orig"]) / abs(r["ppl_orig"])
                          if r["ppl_orig"] != 0 else 0 for r in results])
        med_sc  = rel_sc[len(rel_sc) // 2] * 100
        med_ppl = rel_ppl[len(rel_ppl) // 2] * 100

        bars = ax.bar(["SC", "PPL"], [med_sc, med_ppl], color=[BLUE, ORANGE],
                       edgecolor="black", linewidth=0.5, width=0.5)
        for bar, val in zip(bars, [med_sc, med_ppl]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.set_ylabel("Median relative change (%)", fontsize=10)
        ax.set_title(f"Median |Δ| / original\nOriginal vs {label}\n(lower = more stable)", fontsize=11)
        ax.set_ylim(0, max(med_sc, med_ppl) * 1.4)

    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, "exp1_stability_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logging.info(f"Saved: {out}")
    plt.close(fig)


def main():
    logging.info(f"Loading model {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=HF_CACHE, token=HF_TOKEN, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=HF_CACHE,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
        local_files_only=True,
    )
    model.eval()
    logging.info("Model loaded.\n" + "=" * 60)

    # Load all 3 versions
    with open(ORIGINAL_FILE, encoding="utf-8") as f:
        originals = json.load(f)
    with open(REWRITTEN_FILE, encoding="utf-8") as f:
        rewrittens = json.load(f)
    with open(EXPANSION_FILE, encoding="utf-8") as f:
        expansions = json.load(f)

    orig_map = {item["id"]: item["question"] for item in originals}
    rew_map  = {item["id"]: item["question"] for item in rewrittens}
    exp_map  = {item["id"]: item["question"] for item in expansions}

    # Use IDs common to all 3 files
    common_ids = [item["id"] for item in originals
                  if item["id"] in rew_map and item["id"] in exp_map]
    logging.info(f"Matched question triples: {len(common_ids)}\n")

    results = []

    for qid in tqdm(common_ids, desc="Processing"):
        orig_q = orig_map[qid]
        rew_q  = rew_map[qid]
        exp_q  = exp_map[qid]

        # Generate answer using ORIGINAL question only (fixed)
        answer = generate_answer(model, tokenizer, orig_q)

        # Compute SC/PPL with SAME answer, 3 different prompts
        sc_orig, ppl_orig, _ = compute_sc_and_ppl(model, tokenizer, orig_q, answer)
        sc_rew,  ppl_rew,  _ = compute_sc_and_ppl(model, tokenizer, rew_q,  answer)
        sc_exp,  ppl_exp,  _ = compute_sc_and_ppl(model, tokenizer, exp_q,  answer)

        entry = {
            "id": qid,
            "original_question": orig_q,
            "rewritten_question": rew_q,
            "expansion_question": exp_q,
            "fixed_answer": answer,
            "sc_orig":       round(sc_orig,  6),
            "sc_rewritten":  round(sc_rew,   6),
            "sc_expansion":  round(sc_exp,   6),
            "ppl_orig":      round(ppl_orig, 4),
            "ppl_rewritten": round(ppl_rew,  4),
            "ppl_expansion": round(ppl_exp,  4),
        }
        results.append(entry)

        logging.info(f"[{qid}]")
        logging.info(f"  Orig: {orig_q}")
        logging.info(f"  Rew:  {rew_q[:80]}...")
        logging.info(f"  Exp:  {exp_q[:80]}...")
        logging.info(f"  Answer (fixed): {answer}")
        logging.info(f"  SC:  {sc_orig:.4f} → rew={sc_rew:.4f} / exp={sc_exp:.4f}")
        logging.info(f"  PPL: {ppl_orig:.4f} → rew={ppl_rew:.4f} / exp={ppl_exp:.4f}")
        logging.info("-" * 60)

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # --- Summary for each variant ---
    for label, sc_key, ppl_key in [("Rewritten", "sc_rewritten", "ppl_rewritten"),
                                    ("Expansion", "sc_expansion", "ppl_expansion")]:
        sc_orig_list = [r["sc_orig"] for r in results]
        sc_var_list  = [r[sc_key] for r in results]
        ppl_orig_list = [r["ppl_orig"] for r in results]
        ppl_var_list  = [r[ppl_key] for r in results]
        log_ppl_orig = [math.log(x) if x > 0 else 0 for x in ppl_orig_list]
        log_ppl_var  = [math.log(x) if x > 0 else 0 for x in ppl_var_list]

        rel_sc  = sorted([abs(s2-s1)/abs(s1) if s1 != 0 else 0
                          for s1, s2 in zip(sc_orig_list, sc_var_list)])
        rel_ppl = sorted([abs(p2-p1)/abs(p1) if p1 != 0 else 0
                          for p1, p2 in zip(ppl_orig_list, ppl_var_list)])

        def med(lst):
            return lst[len(lst) // 2]

        corr_sc      = pearson(sc_orig_list, sc_var_list)
        corr_log_ppl = pearson(log_ppl_orig, log_ppl_var)

        logging.info(f"\n{'='*60}")
        logging.info(f"  Original vs {label}")
        logging.info(f"  SC  median rel |Δ|:  {med(rel_sc)*100:.2f}%   corr: {corr_sc:.4f}")
        logging.info(f"  PPL median rel |Δ|:  {med(rel_ppl)*100:.2f}%   log(PPL) corr: {corr_log_ppl:.4f}")
        logging.info(f"  SC more stable: {med(rel_sc) < med(rel_ppl) and corr_sc > corr_log_ppl}")

    logging.info(f"\nResults saved to {OUTPUT_FILE}")

    # --- Plot ---
    plot_comparison(results)
    logging.info(f"Figures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
