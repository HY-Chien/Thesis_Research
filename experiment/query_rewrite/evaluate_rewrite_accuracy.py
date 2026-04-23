"""
Compare Adaptive-RAG accuracy: Original vs Rewritten vs Expansion queries

The classifier assigns A/B/C labels to each query version.
We then look up the actual answer from the corresponding QA prediction file
(A=nor_qa, B=oner_qa, C=ircot_qa) and evaluate against ground truth.

Classification result files:
  - original:   t5_large_original_nq_result_epoch_30/dict_id_pred_results.json
  - rewritten:  t5_large_rewritten_nq_result_epoch_30/dict_id_pred_results.json
  - expansion:  t5_large_rewritten_nq_expansion_result_epoch_30/dict_id_pred_results.json

QA prediction files (flan-t5-xl, NQ):
  - A (nor):   predictions/test/nor_qa_flan_t5_xl_nq_.../prediction__nq_to_nq__test_subsampled.json
  - B (oner):  predictions/test/oner_qa_flan_t5_xl_nq_.../prediction__nq_to_nq__test_subsampled.json
  - C (ircot): predictions/test/ircot_qa_flan_t5_xl_nq_.../prediction__nq_to_nq__test_subsampled.json
"""

import json
import re
import string
import collections
import os

# ================= Paths =================
BASE = "/home/S113065528/Adaptive-RAG"
EXPERIMENT_BASE = f"{BASE}/experiment/query_rewrite/experiment/query_rewrite"

CLASSIFICATION_FILES = {
    "original":  f"{EXPERIMENT_BASE}/t5_large_original_nq_result_epoch_30/dict_id_pred_results.json",
    "rewritten": f"{EXPERIMENT_BASE}/t5_large_rewritten_nq_result_epoch_30/dict_id_pred_results.json",
    "expansion": f"{EXPERIMENT_BASE}/t5_large_rewritten_nq_expansion_result_epoch_30/dict_id_pred_results.json",
}

# QA prediction files (answers from each retrieval strategy, based on original NQ questions)
QA_PREDICTIONS = {
    "A": f"{BASE}/predictions/test/nor_qa_flan_t5_xl_nq____prompt_set_1/prediction__nq_to_nq__test_subsampled.json",
    "B": f"{BASE}/predictions/test/oner_qa_flan_t5_xl_nq____prompt_set_1___bm25_retrieval_count__15___distractor_count__1/prediction__nq_to_nq__test_subsampled.json",
    "C": f"{BASE}/predictions/test/ircot_qa_flan_t5_xl_nq____prompt_set_1___bm25_retrieval_count__6___distractor_count__1/prediction__nq_to_nq__test_subsampled.json",
}

GROUND_TRUTH_FILE = f"{BASE}/predictions/test/nor_qa_flan_t5_xl_nq____prompt_set_1/ground_truth__nq_to_nq__test_subsampled.json"
OUTPUT_DIR = f"{BASE}/experiment/query_rewrite"
# =========================================


# ---------- Evaluation ----------
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text, flags=re.UNICODE)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_em(prediction, ground_truths):
    pred_norm = normalize_answer(prediction)
    return max(int(pred_norm == normalize_answer(gt)) for gt in ground_truths)


def compute_f1(prediction, ground_truths):
    def token_f1(pred, gold):
        pred_toks = normalize_answer(pred).split()
        gold_toks = normalize_answer(gold).split()
        common = collections.Counter(pred_toks) & collections.Counter(gold_toks)
        num_same = sum(common.values())
        if not pred_toks or not gold_toks:
            return int(pred_toks == gold_toks)
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_toks)
        recall = num_same / len(gold_toks)
        return (2 * precision * recall) / (precision + recall)
    return max(token_f1(prediction, gt) for gt in ground_truths)


# ---------- Load files ----------
def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_one_version(version_name, classification_file, qa_preds, ground_truth):
    """
    Evaluate accuracy for one version (original / rewritten / expansion).
    Returns a dict with EM, F1, label distribution, and per-label accuracy.
    """
    clf = load_json(classification_file)

    label_counts = {"A": 0, "B": 0, "C": 0}
    label_em = {"A": [], "B": [], "C": []}
    label_f1 = {"A": [], "B": [], "C": []}
    skipped = 0

    for qid, info in clf.items():
        label = info["prediction"]
        label_counts[label] += 1

        # Ground truth answers
        gts = ground_truth.get(qid)
        if not gts:
            skipped += 1
            continue

        # Predicted answer from the corresponding QA strategy
        pred_answer = qa_preds[label].get(qid)
        if pred_answer is None:
            skipped += 1
            continue

        em = compute_em(pred_answer, gts)
        f1 = compute_f1(pred_answer, gts)
        label_em[label].append(em)
        label_f1[label].append(f1)

    total = sum(len(v) for v in label_em.values())
    all_em = [s for v in label_em.values() for s in v]
    all_f1 = [s for v in label_f1.values() for s in v]

    result = {
        "version": version_name,
        "total_evaluated": total,
        "skipped": skipped,
        "overall_em": round(sum(all_em) / len(all_em) * 100, 2) if all_em else 0,
        "overall_f1": round(sum(all_f1) / len(all_f1) * 100, 2) if all_f1 else 0,
        "label_distribution": label_counts,
        "per_label": {},
    }

    for label in ["A", "B", "C"]:
        em_scores = label_em[label]
        f1_scores = label_f1[label]
        result["per_label"][label] = {
            "count": label_counts[label],
            "em": round(sum(em_scores) / len(em_scores) * 100, 2) if em_scores else 0,
            "f1": round(sum(f1_scores) / len(f1_scores) * 100, 2) if f1_scores else 0,
        }

    return result


def print_table(results):
    versions = [r["version"] for r in results]
    header_w = 28

    print("\n" + "=" * 70)
    print("Adaptive-RAG Accuracy: Original vs Rewritten vs Expansion (NQ)")
    print("=" * 70)

    # Overall
    print(f"\n{'Metric':<20}", end="")
    for v in versions:
        print(f"{v:>16}", end="")
    print()
    print("-" * (20 + 16 * len(versions)))

    for metric in ["overall_em", "overall_f1"]:
        label = "EM (%)" if "em" in metric else "F1 (%)"
        print(f"{label:<20}", end="")
        for r in results:
            print(f"{r[metric]:>15.2f}%", end="")
        print()

    # Label distribution
    print(f"\n{'Label Dist.':<20}", end="")
    for v in versions:
        print(f"{v:>16}", end="")
    print()
    print("-" * (20 + 16 * len(versions)))
    for label in ["A", "B", "C"]:
        strategy = {"A": "A (no-retrieval)", "B": "B (single-step)", "C": "C (multi-step)"}[label]
        print(f"  {strategy:<18}", end="")
        for r in results:
            cnt = r["label_distribution"][label]
            pct = cnt / r["total_evaluated"] * 100
            print(f"{cnt:>10} ({pct:4.1f}%)", end="")
        print()

    # Per-label EM
    print(f"\n{'Per-label EM':<20}", end="")
    for v in versions:
        print(f"{v:>16}", end="")
    print()
    print("-" * (20 + 16 * len(versions)))
    for label in ["A", "B", "C"]:
        strategy = {"A": "A (no-retr.)", "B": "B (single)", "C": "C (multi)"}[label]
        print(f"  {strategy:<18}", end="")
        for r in results:
            print(f"{r['per_label'][label]['em']:>15.2f}%", end="")
        print()

    print()


def main():
    print("Loading QA prediction files...")
    qa_preds = {label: load_json(path) for label, path in QA_PREDICTIONS.items()}
    ground_truth = load_json(GROUND_TRUTH_FILE)
    print(f"Ground truth: {len(ground_truth)} questions\n")

    results = []
    for version, clf_file in CLASSIFICATION_FILES.items():
        if not os.path.exists(clf_file):
            print(f"[SKIP] {version}: file not found → {clf_file}")
            continue
        print(f"Evaluating: {version}...")
        r = evaluate_one_version(version, clf_file, qa_preds, ground_truth)
        results.append(r)
        print(f"  EM={r['overall_em']}%  F1={r['overall_f1']}%  labels={r['label_distribution']}")

    print_table(results)

    # Save results
    out_path = os.path.join(OUTPUT_DIR, "rewrite_accuracy_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
