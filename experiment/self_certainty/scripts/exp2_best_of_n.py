"""
Experiment 2: Best-of-N Selection — SC vs PPL (faithful to the paper)
=====================================================================
For each question, sample N candidate answers (with CoT reasoning),
compute SC and PPL for each, then select the best candidate by each metric.

Compare: SC-selected accuracy vs PPL-selected accuracy vs greedy vs random.

Setup (following the paper):
  - Sampling: temperature=0.6, top_p=0.9
  - N = 16 candidates per question
  - Chain-of-thought prompting (so SC has more tokens to measure)
  - Model: Llama-3.1-8B-Instruct

Datasets: NQ (500), 2WikiMultiHopQA (500)
"""

import json
import logging
import math
import os
import random
import re
import string
import collections

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= Settings =================
DATASETS = {
    # "nq": {
    #     "questions": "/home/S113065528/Adaptive-RAG/experiment/query_rewrite/dataset/predict_nq_original.json",
    #     "ground_truth": "/home/S113065528/Adaptive-RAG/processed_data/nq/test_subsampled.jsonl",
    # },
    "2wiki": {
        "questions": "/home/S113065528/Adaptive-RAG/experiment/query_rewrite/dataset/predict_2wikimultihopqa_original.json",
        "ground_truth": "/home/S113065528/Adaptive-RAG/processed_data/2wikimultihopqa/test_subsampled.jsonl",
    },
}

OUTPUT_DIR = "/home/S113065528/Adaptive-RAG/experiment/self_certainty"
LOG_FILE = os.path.join(OUTPUT_DIR, "exp2_best_of_n.log")

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
HF_CACHE = "/home/S113065528/Adaptive-RAG/hf_cache"
HF_TOKEN = os.environ.get("HF_TOKEN", None)

N_CANDIDATES = 8        # Best-of-N: number of sampled candidates
BORDA_P = 0.3           # Borda exponent (paper uses 0.3 and 1.2)
TEMPERATURE = 0.6       # following the paper
TOP_P = 0.9             # following the paper
MAX_NEW_TOKENS = 256    # longer for CoT reasoning
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


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def max_over_ground_truths(metric_fn, prediction, ground_truths):
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def is_correct(prediction, ground_truths):
    em = max_over_ground_truths(compute_exact, prediction, ground_truths)
    f1 = max_over_ground_truths(compute_f1, prediction, ground_truths)
    return em == 1 or f1 >= 0.5


# ---------- Data loading ----------
def load_ground_truth(filepath):
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
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    return [{"id": item["id"], "question": item["question"]} for item in data]


# ---------- Prompts ----------
def build_cot_prompt(tokenizer, question: str) -> str:
    """Chain-of-thought prompt: model reasons step by step, then gives final answer."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a knowledgeable assistant. "
                "Think through the question step by step, then provide the final answer. "
                "Format your response as:\n"
                "Reasoning: <your step-by-step thinking>\n"
                "Answer: <short final answer>"
            ),
        },
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_greedy_prompt(tokenizer, question: str) -> str:
    """Direct answer prompt for greedy baseline."""
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


def extract_final_answer(full_response: str) -> str:
    """Extract the final answer from a CoT response."""
    # Try to find "Answer: ..." pattern
    patterns = [
        r"[Aa]nswer\s*:\s*(.+?)(?:\n|$)",
        r"[Ff]inal [Aa]nswer\s*:\s*(.+?)(?:\n|$)",
        r"[Tt]he answer is\s*:?\s*(.+?)(?:\n|\.?$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, full_response, re.DOTALL)
        if match:
            answer = match.group(1).strip().rstrip(".")
            # Take only the first line of the extracted answer
            answer = answer.split("\n")[0].strip()
            if answer:
                return answer
    # Fallback: use the last non-empty line
    lines = [l.strip() for l in full_response.strip().split("\n") if l.strip()]
    if lines:
        last = lines[-1]
        # Remove common prefixes
        for prefix in ["Answer:", "answer:", "So,", "Therefore,", "Thus,"]:
            if last.startswith(prefix):
                last = last[len(prefix):].strip()
        return last
    return full_response.strip()


# ---------- SC and PPL computation ----------
@torch.no_grad()
def compute_sc_and_ppl(model, tokenizer, prompt_text, answer_text):
    """
    Compute Self-Certainty and PPL on the generated answer tokens.
    SC = -(1/nV) * sum_i sum_j log(V * p(j|context))
       = -(1/V) * mean_i(sum_j log p(j|...)) - log(V)
    """
    if not answer_text or not answer_text.strip():
        return 0.0, float("inf"), 0

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

    # Chunk if sequence is very long to avoid OOM
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(full_ids).logits

    answer_logits = logits[
        :, prompt_len - 1 : prompt_len + answer_len - 1, :
    ].squeeze(0).float()  # (answer_len, V)

    answer_token_ids = answer_ids.squeeze(0).to(model.device)

    log_probs_all = F.log_softmax(answer_logits, dim=-1)

    # PPL: based on the probability of the actual generated token
    token_log_probs = log_probs_all.gather(
        1, answer_token_ids.unsqueeze(1)
    ).squeeze(1)
    avg_nll = -token_log_probs.mean().item()
    ppl = math.exp(min(avg_nll, 20))

    # SC: based on peakedness of the full distribution (KL from uniform)
    logprob_sum = log_probs_all.sum(dim=-1)  # sum over vocab
    V_float = float(V)
    sc_per_token = -1.0 / V_float * logprob_sum - math.log(V_float)
    sc = sc_per_token.mean().item()

    return sc, ppl, answer_len


def borda_select(candidates, score_key, higher_is_better=True, p=BORDA_P):
    """
    Borda Count selection (Eq. 11 from the paper).
    1. Sort candidates by score_key → get rank r (1 = best)
    2. Each candidate contributes v(r) = (N - r + 1)^p votes to its answer
    3. The answer with the most total votes wins.
    """
    N = len(candidates)
    # Sort by score to assign ranks
    sorted_idxs = sorted(range(N), key=lambda i: candidates[i][score_key],
                         reverse=higher_is_better)
    # r[i] = rank of candidate i (1-based, 1 = best)
    ranks = [0] * N
    for rank_pos, idx in enumerate(sorted_idxs):
        ranks[idx] = rank_pos + 1  # rank 1 = highest score

    # Accumulate votes per normalized answer
    answer_votes = collections.defaultdict(float)
    answer_repr = {}  # normalized → original string
    for i, c in enumerate(candidates):
        r = ranks[i]
        vote = (N - r + 1) ** p
        norm = normalize_answer(c["final_answer"])
        answer_votes[norm] += vote
        if norm not in answer_repr:
            answer_repr[norm] = c["final_answer"]

    best_norm = max(answer_votes, key=answer_votes.__getitem__)
    return answer_repr[best_norm]


def run_best_of_n(dataset_name, questions, gt_map, model, tokenizer):
    """Run Best-of-N experiment on one dataset."""
    logging.info(f"\n{'='*70}")
    logging.info(f"  Dataset: {dataset_name}  |  N={N_CANDIDATES}  |  T={TEMPERATURE}  |  Borda p={BORDA_P}")
    logging.info(f"{'='*70}\n")

    results = []

    # Counters for final accuracy
    stats = {
        "sc_correct": 0,           # SC best-of-N
        "ppl_correct": 0,          # PPL best-of-N
        "greedy_correct": 0,
        "random_correct": 0,
        "self_consistency_correct": 0,   # pure majority vote (p=0)
        "borda_sc_correct": 0,           # Borda with SC ranking (paper's method)
        "borda_ppl_correct": 0,          # Borda with PPL ranking
        "oracle_correct": 0,
        "total": 0,
    }

    for item in tqdm(questions, desc=f"Best-of-{N_CANDIDATES} ({dataset_name})"):
        qid = item["id"]
        question = item["question"]
        ground_truths = gt_map.get(qid)
        if not ground_truths:
            continue

        stats["total"] += 1
        cot_prompt = build_cot_prompt(tokenizer, question)
        cot_input_ids = tokenizer(
            cot_prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(model.device)

        # --- Step 1: Generate N candidates with sampling ---
        candidates = []
        for _ in range(N_CANDIDATES):
            with torch.no_grad():
                output_ids = model.generate(
                    cot_input_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_ids = output_ids[0][cot_input_ids.shape[1]:]
            full_response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            final_answer = extract_final_answer(full_response)

            # Compute SC and PPL on the full response (including reasoning)
            sc, ppl, n_tokens = compute_sc_and_ppl(
                model, tokenizer, cot_prompt, full_response
            )

            correct = is_correct(final_answer, ground_truths)

            candidates.append({
                "full_response": full_response,
                "final_answer": final_answer,
                "is_correct": correct,
                "sc": round(sc, 6),
                "ppl": round(ppl, 4),
                "n_tokens": n_tokens,
            })

        # --- Step 2: Greedy baseline ---
        greedy_prompt = build_greedy_prompt(tokenizer, question)
        greedy_input_ids = tokenizer(
            greedy_prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(model.device)
        with torch.no_grad():
            greedy_output = model.generate(
                greedy_input_ids,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        greedy_ids = greedy_output[0][greedy_input_ids.shape[1]:]
        greedy_answer = tokenizer.decode(greedy_ids, skip_special_tokens=True).strip()
        greedy_correct = is_correct(greedy_answer, ground_truths)

        # --- Step 3: Selection methods ---
        # Filter: candidates with empty answers get -inf SC / inf PPL (like the paper)
        for c in candidates:
            if not c["final_answer"] or not c["final_answer"].strip():
                c["sc"] = -1e9
                c["ppl"] = float("inf")

        best_sc_idx = max(range(len(candidates)), key=lambda i: candidates[i]["sc"])
        best_ppl_idx = min(range(len(candidates)), key=lambda i: candidates[i]["ppl"])
        random_idx = random.randint(0, len(candidates) - 1)

        # Self-consistency: pure majority vote (Borda p=0)
        sc_answer = borda_select(candidates, "sc", higher_is_better=True, p=0)
        sc_correct_flag = is_correct(sc_answer, ground_truths)

        # Borda + SC ranking (paper's main method)
        borda_sc_answer = borda_select(candidates, "sc", higher_is_better=True, p=BORDA_P)
        borda_sc_correct = is_correct(borda_sc_answer, ground_truths)

        # Borda + PPL ranking (comparison)
        borda_ppl_answer = borda_select(candidates, "ppl", higher_is_better=False, p=BORDA_P)
        borda_ppl_correct = is_correct(borda_ppl_answer, ground_truths)

        # Oracle: at least one candidate is correct
        oracle_correct = any(c["is_correct"] for c in candidates)

        # Update stats
        stats["sc_correct"] += int(candidates[best_sc_idx]["is_correct"])
        stats["ppl_correct"] += int(candidates[best_ppl_idx]["is_correct"])
        stats["greedy_correct"] += int(greedy_correct)
        stats["random_correct"] += int(candidates[random_idx]["is_correct"])
        stats["self_consistency_correct"] += int(sc_correct_flag)
        stats["borda_sc_correct"] += int(borda_sc_correct)
        stats["borda_ppl_correct"] += int(borda_ppl_correct)
        stats["oracle_correct"] += int(oracle_correct)

        entry = {
            "id": qid,
            "question": question,
            "ground_truths": ground_truths,
            "greedy_answer": greedy_answer,
            "greedy_correct": greedy_correct,
            "best_sc_answer": candidates[best_sc_idx]["final_answer"],
            "best_ppl_answer": candidates[best_ppl_idx]["final_answer"],
            "self_consistency_answer": sc_answer,
            "borda_sc_answer": borda_sc_answer,
            "borda_ppl_answer": borda_ppl_answer,
            "candidates": candidates,
        }
        results.append(entry)

        # Log progress
        n = stats["total"]
        if n % 20 == 0 or n <= 5:
            logging.info(
                f"[{n:3d}] {qid}  "
                f"Greedy={'✓' if greedy_correct else '✗'}  "
                f"SC-BoN={'✓' if candidates[best_sc_idx]['is_correct'] else '✗'}  "
                f"PPL-BoN={'✓' if candidates[best_ppl_idx]['is_correct'] else '✗'}  "
                f"SelfCons={'✓' if sc_correct_flag else '✗'}  "
                f"Borda-SC={'✓' if borda_sc_correct else '✗'}  "
                f"Oracle={'✓' if oracle_correct else '✗'}"
            )

    # --- Save results ---
    out_path = os.path.join(OUTPUT_DIR, f"exp2_{dataset_name}_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # --- Summary ---
    total = stats["total"]
    summary = {
        "dataset": dataset_name,
        "n_candidates": N_CANDIDATES,
        "temperature": TEMPERATURE,
        "total_questions": total,
        "accuracy_sc": round(stats["sc_correct"] / total * 100, 2),
        "accuracy_ppl": round(stats["ppl_correct"] / total * 100, 2),
        "accuracy_greedy": round(stats["greedy_correct"] / total * 100, 2),
        "accuracy_random": round(stats["random_correct"] / total * 100, 2),
        "accuracy_self_consistency": round(stats["self_consistency_correct"] / total * 100, 2),
        "accuracy_borda_sc": round(stats["borda_sc_correct"] / total * 100, 2),
        "accuracy_borda_ppl": round(stats["borda_ppl_correct"] / total * 100, 2),
        "accuracy_oracle": round(stats["oracle_correct"] / total * 100, 2),
        "borda_p": BORDA_P,
    }

    logging.info(f"\n{'='*55}")
    logging.info(f"  {dataset_name} — Best-of-{N_CANDIDATES} Results  (Borda p={BORDA_P})")
    logging.info(f"{'='*55}")
    logging.info(f"  {'Method':<28} {'Accuracy':>10}")
    logging.info(f"  {'-'*40}")
    logging.info(f"  {'Greedy':<28} {summary['accuracy_greedy']:>9.2f}%")
    logging.info(f"  {'Random (1 of N)':<28} {summary['accuracy_random']:>9.2f}%")
    logging.info(f"  {'PPL Best-of-N':<28} {summary['accuracy_ppl']:>9.2f}%")
    logging.info(f"  {'SC Best-of-N':<28} {summary['accuracy_sc']:>9.2f}%")
    logging.info(f"  {'Self-Consistency (majority)':<28} {summary['accuracy_self_consistency']:>9.2f}%")
    logging.info(f"  {'Borda (SC ranking, p={:.1f})':<28} {summary['accuracy_borda_sc']:>9.2f}%".format(BORDA_P))
    logging.info(f"  {'Borda (PPL ranking, p={:.1f})':<28} {summary['accuracy_borda_ppl']:>9.2f}%".format(BORDA_P))
    logging.info(f"  {'Oracle':<28} {summary['accuracy_oracle']:>9.2f}%")

    return summary


def main():
    logging.info(f"Loading model {MODEL_ID}...")
    logging.info(f"Best-of-N: N={N_CANDIDATES}, T={TEMPERATURE}, top_p={TOP_P}")
    logging.info(f"Max new tokens: {MAX_NEW_TOKENS} (CoT reasoning)\n")

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
        summary = run_best_of_n(name, questions, gt_map, model, tokenizer)
        all_summaries.append(summary)

    # Save combined summary
    summary_path = os.path.join(OUTPUT_DIR, "exp2_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    # Final report
    logging.info(f"\n{'='*70}")
    logging.info(f"FINAL SUMMARY — Best-of-{N_CANDIDATES} Selection")
    logging.info(f"{'='*70}")
    logging.info(f"{'Dataset':<10} {'Greedy':>7} {'Random':>7} {'PPL-BoN':>8} {'SC-BoN':>8} {'SelfCons':>9} {'Borda-SC':>9} {'Borda-PPL':>10} {'Oracle':>7}")
    logging.info("-" * 82)
    for s in all_summaries:
        logging.info(
            f"{s['dataset']:<10} "
            f"{s['accuracy_greedy']:>6.1f}% "
            f"{s['accuracy_random']:>6.1f}% "
            f"{s['accuracy_ppl']:>7.1f}% "
            f"{s['accuracy_sc']:>7.1f}% "
            f"{s['accuracy_self_consistency']:>8.1f}% "
            f"{s['accuracy_borda_sc']:>8.1f}% "
            f"{s['accuracy_borda_ppl']:>9.1f}% "
            f"{s['accuracy_oracle']:>6.1f}%"
        )
    logging.info(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()
