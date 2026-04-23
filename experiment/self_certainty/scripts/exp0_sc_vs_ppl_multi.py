"""
Experiment 0 (Multi-dataset): Self-Certainty vs PPL Direct Comparison
=====================================================================
Runs SC vs PPL separation on 4 conditions:
  1. NQ original
  2. NQ rewritten
  3. 2WikiMultiHopQA original
  4. 2WikiMultiHopQA rewritten

For each condition, generates answers with Llama-3.1-8B-Instruct (no retrieval),
evaluates correctness, and computes SC + PPL on the generated answer.

Output: one JSON per condition + a combined summary JSON for plotting.
"""

import json
import logging
import math
import os
import re
import string
import collections

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= Settings =================
DATASETS = {
    "nq_original": {
        "questions": "/home/S113065528/Adaptive-RAG/experiment/query_rewrite/dataset/predict_nq_original.json",
        "ground_truth": "/home/S113065528/Adaptive-RAG/processed_data/nq/test_subsampled.jsonl",
    },
    "nq_rewritten": {
        "questions": "/home/S113065528/Adaptive-RAG/experiment/query_rewrite/dataset/predict_nq_rewritten_qwen.json",
        "ground_truth": "/home/S113065528/Adaptive-RAG/processed_data/nq/test_subsampled.jsonl",
    },
    "2wiki_original": {
        "questions": "/home/S113065528/Adaptive-RAG/experiment/query_rewrite/dataset/predict_2wikimultihopqa_original.json",
        "ground_truth": "/home/S113065528/Adaptive-RAG/processed_data/2wikimultihopqa/test_subsampled.jsonl",
    },
    "2wiki_rewritten": {
        "questions": "/home/S113065528/Adaptive-RAG/experiment/query_rewrite/dataset/predict_2wikimultihopqa_rewritten_qwen.json",
        "ground_truth": "/home/S113065528/Adaptive-RAG/processed_data/2wikimultihopqa/test_subsampled.jsonl",
    },
}

OUTPUT_DIR = "/home/S113065528/Adaptive-RAG/experiment/self_certainty"
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "exp0_multi_summary.json")
LOG_FILE = os.path.join(OUTPUT_DIR, "exp0_multi.log")

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
HF_CACHE = "/home/S113065528/Adaptive-RAG/hf_cache"
HF_TOKEN = os.environ.get("HF_TOKEN", None)
MAX_NEW_TOKENS = 64
# =============================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)


# ---------- Answer evaluation ----------
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text, flags=re.UNICODE)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = normalize_answer(a_gold).split()
    pred_toks = normalize_answer(a_pred).split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if not gold_toks or not pred_toks:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def max_over_ground_truths(metric_fn, prediction, ground_truths):
    return max(metric_fn(prediction, gt) for gt in ground_truths)


# ---------- Load data ----------
def load_ground_truth(filepath):
    """Load ground truth from test_subsampled.jsonl, return {question_id: [answers]}."""
    gt = {}
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            qid = item["question_id"]
            answers = []
            for ans_obj in item.get("answers_objects", []):
                answers.extend(ans_obj.get("spans", []))
            if answers:
                gt[qid] = answers
    return gt


def load_questions(filepath):
    """Load questions from predict_*.json, return [{id, question}]."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    return [{"id": item["id"], "question": item["question"]} for item in data]


# ---------- Build chat prompt ----------
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


# ---------- SC and PPL computation ----------
@torch.no_grad()
def compute_sc_and_ppl(model, tokenizer, prompt_text, answer_text):
    V = model.config.vocab_size

    prompt_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    ).input_ids
    prompt_len = prompt_ids.shape[1]

    answer_ids = tokenizer(
        answer_text, return_tensors="pt", add_special_tokens=False
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


# ---------- ROC / AUC ----------
def compute_auc(scores_pos, scores_neg, higher_is_positive=True):
    labels = [1] * len(scores_pos) + [0] * len(scores_neg)
    scores = list(scores_pos) + list(scores_neg)
    if not higher_is_positive:
        scores = [-s for s in scores]
    pairs = sorted(zip(scores, labels), reverse=True)
    tp = fp = 0
    total_pos = sum(labels)
    total_neg = len(labels) - total_pos
    if total_pos == 0 or total_neg == 0:
        return 0.0
    fprs, tprs = [0.0], [0.0]
    for _, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tprs.append(tp / total_pos)
        fprs.append(fp / total_neg)
    auc = sum(
        (fprs[i+1] - fprs[i]) * (tprs[i+1] + tprs[i]) / 2
        for i in range(len(fprs) - 1)
    )
    return auc


def run_one_dataset(name, questions, gt_map, model, tokenizer):
    """Run SC/PPL evaluation on one dataset condition."""
    logging.info(f"\n{'='*60}")
    logging.info(f"  Dataset: {name}")
    logging.info(f"{'='*60}\n")

    results = []
    skipped = 0

    for item in tqdm(questions, desc=name):
        qid = item["id"]
        question = item["question"]
        ground_truths = gt_map.get(qid)
        if not ground_truths:
            skipped += 1
            continue

        # Generate answer
        prompt_text = build_prompt(tokenizer, question)
        input_ids = tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_ids = output_ids[0][input_ids.shape[1]:]
        predicted_answer = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        # Evaluate
        em = max_over_ground_truths(compute_exact, predicted_answer, ground_truths)
        f1 = max_over_ground_truths(compute_f1, predicted_answer, ground_truths)
        is_correct = em == 1 or f1 >= 0.5

        # SC and PPL
        sc, ppl, n_tokens = compute_sc_and_ppl(
            model, tokenizer, prompt_text, predicted_answer
        )

        results.append({
            "id": qid,
            "question": question,
            "predicted_answer": predicted_answer,
            "ground_truths": ground_truths,
            "em": em,
            "f1": round(f1, 4),
            "is_correct": is_correct,
            "self_certainty": round(sc, 6),
            "ppl": round(ppl, 4),
            "n_answer_tokens": n_tokens,
        })

        logging.info(f"[{qid}] {'✓' if is_correct else '✗'}  SC={sc:.4f}  PPL={ppl:.2f}  A: {predicted_answer[:50]}")

    if skipped > 0:
        logging.info(f"  (skipped {skipped} questions without ground truth)")

    # Save per-dataset results
    out_path = os.path.join(OUTPUT_DIR, f"exp0_{name}_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logging.info(f"Saved: {out_path}")

    # Compute summary stats
    correct = [r for r in results if r["is_correct"]]
    incorrect = [r for r in results if not r["is_correct"]]

    sc_correct = [r["self_certainty"] for r in correct]
    sc_incorrect = [r["self_certainty"] for r in incorrect]
    ppl_correct = [r["ppl"] for r in correct if r["ppl"] != float("inf")]
    ppl_incorrect = [r["ppl"] for r in incorrect if r["ppl"] != float("inf")]

    auc_sc = compute_auc(sc_correct, sc_incorrect, higher_is_positive=True)
    auc_ppl = compute_auc(ppl_correct, ppl_incorrect, higher_is_positive=False)

    def safe_avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    summary = {
        "name": name,
        "total": len(results),
        "correct": len(correct),
        "incorrect": len(incorrect),
        "accuracy": round(len(correct) / len(results) * 100, 1) if results else 0,
        "auc_sc": round(auc_sc, 4),
        "auc_ppl": round(auc_ppl, 4),
        "avg_sc_correct": round(safe_avg(sc_correct), 4),
        "avg_sc_incorrect": round(safe_avg(sc_incorrect), 4),
        "avg_ppl_correct": round(safe_avg(ppl_correct), 4),
        "avg_ppl_incorrect": round(safe_avg(ppl_incorrect), 4),
    }

    logging.info(f"\n--- {name} Summary ---")
    logging.info(f"  Accuracy: {summary['accuracy']}%  ({len(correct)}/{len(results)})")
    logging.info(f"  AUC SC:  {auc_sc:.4f}")
    logging.info(f"  AUC PPL: {auc_ppl:.4f}")

    return summary


def main():
    logging.info(f"Loading model {MODEL_ID}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=HF_CACHE, token=HF_TOKEN)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=HF_CACHE,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    )
    model.eval()
    logging.info("Model loaded.\n")

    all_summaries = []

    for name, paths in DATASETS.items():
        questions = load_questions(paths["questions"])
        gt_map = load_ground_truth(paths["ground_truth"])
        summary = run_one_dataset(name, questions, gt_map, model, tokenizer)
        all_summaries.append(summary)

    # Save combined summary
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    # Final report
    logging.info("\n" + "=" * 70)
    logging.info("FINAL SUMMARY — SC vs PPL AUC across all conditions")
    logging.info("=" * 70)
    logging.info(f"{'Condition':<25} {'Accuracy':>10} {'AUC_SC':>10} {'AUC_PPL':>10} {'Winner':>10}")
    logging.info("-" * 70)
    for s in all_summaries:
        winner = "SC" if s["auc_sc"] > s["auc_ppl"] else "PPL"
        logging.info(f"{s['name']:<25} {s['accuracy']:>9.1f}% {s['auc_sc']:>10.4f} {s['auc_ppl']:>10.4f} {winner:>10}")
    logging.info(f"\nSummary saved to {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
