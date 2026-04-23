"""
Plot Exp1 Stability results split by answer correctness.

Shows that:
- SC is stable (~5%) regardless of whether the answer is correct or incorrect
- PPL is unstable, especially for incorrect answers (60-68%)
- PPL's instability is NOT because it "detects wrong answers" — it's random noise
"""

import json
import math
import os
import re
import string
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================= Paths =================
STABILITY_FILE = "/home/S113065528/Adaptive-RAG/experiment/self_certainty/exp1_stability_results.json"
GT_FILE = "/home/S113065528/Adaptive-RAG/processed_data/nq/test_subsampled.jsonl"
FIGURE_DIR = "/home/S113065528/Adaptive-RAG/experiment/self_certainty/figures"
# =========================================

os.makedirs(FIGURE_DIR, exist_ok=True)

BLUE = "#4C72B0"
ORANGE = "#DD8452"
GREEN = "#55A868"
RED = "#C44E52"


# ---------- Answer evaluation ----------
def normalize_answer(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def compute_f1(pred, gold):
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    common = Counter(pred_toks) & Counter(gold_toks)
    num = sum(common.values())
    if not pred_toks or not gold_toks:
        return int(pred_toks == gold_toks)
    if num == 0:
        return 0
    p = num / len(pred_toks)
    r = num / len(gold_toks)
    return 2 * p * r / (p + r)


def is_correct(pred, gts):
    pred_n = normalize_answer(pred)
    for gt in gts:
        if pred_n == normalize_answer(gt):
            return True
        if compute_f1(pred, gt) >= 0.5:
            return True
    return False


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den != 0 else 0.0


def median(lst):
    s = sorted(lst)
    n = len(s)
    return s[n // 2]


# ---------- Load data ----------
with open(STABILITY_FILE, encoding="utf-8") as f:
    results = json.load(f)

# Load ground truth
gt_map = {}
with open(GT_FILE, encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        qid = item["question_id"]
        answers = []
        for ao in item.get("answers_objects", []):
            answers.extend(ao.get("spans", []))
        gt_map[qid] = answers

# Split into correct / incorrect
correct_results = []
incorrect_results = []
for r in results:
    gts = gt_map.get(r["id"], [])
    if gts and is_correct(r["fixed_answer"], gts):
        correct_results.append(r)
    else:
        incorrect_results.append(r)

print(f"Total: {len(results)}")
print(f"Correct: {len(correct_results)}")
print(f"Incorrect: {len(incorrect_results)}")

# ---------- Compute stats per group ----------
variants = [
    ("Rewritten", "sc_rewritten", "ppl_rewritten"),
    ("Expansion", "sc_expansion", "ppl_expansion"),
]

stats = {}
for group_name, group in [("Correct", correct_results), ("Incorrect", incorrect_results)]:
    for label, sc_key, ppl_key in variants:
        key = f"{group_name}_{label}"

        rel_sc = [abs(r[sc_key] - r["sc_orig"]) / abs(r["sc_orig"])
                  if r["sc_orig"] != 0 else 0 for r in group]
        rel_ppl = [abs(r[ppl_key] - r["ppl_orig"]) / abs(r["ppl_orig"])
                   if r["ppl_orig"] != 0 else 0 for r in group]

        sc_orig = [r["sc_orig"] for r in group]
        sc_var = [r[sc_key] for r in group]
        log_ppl_orig = [math.log(r["ppl_orig"]) for r in group if r["ppl_orig"] > 0]
        log_ppl_var = [math.log(r[ppl_key]) for r in group if r[ppl_key] > 0]

        stats[key] = {
            "med_sc": median(rel_sc) * 100,
            "med_ppl": median(rel_ppl) * 100,
            "corr_sc": pearson(sc_orig, sc_var),
            "corr_log_ppl": pearson(log_ppl_orig, log_ppl_var),
            "n": len(group),
        }

        print(f"\n{group_name} (n={len(group)}) vs {label}:")
        print(f"  SC  median rel delta: {stats[key]['med_sc']:.1f}%  corr: {stats[key]['corr_sc']:.3f}")
        print(f"  PPL median rel delta: {stats[key]['med_ppl']:.1f}%  log corr: {stats[key]['corr_log_ppl']:.3f}")

# ============================================================
#  Figure: 2 rows x 3 cols
#  Row 1: Rewritten (scatter correct, scatter incorrect, bar chart)
#  Row 2: Expansion (scatter correct, scatter incorrect, bar chart)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    "Exp 1: SC vs PPL Stability — Split by Answer Correctness\n"
    "(Fixed answer, only prompt changes. SC stable ~5% for both groups; PPL unstable for incorrect)",
    fontsize=13, fontweight="bold",
)

for row_idx, (label, sc_key, ppl_key) in enumerate(variants):

    # --- Left: SC scatter, color by correctness ---
    ax = axes[row_idx][0]
    sc_orig_c = [r["sc_orig"] for r in correct_results]
    sc_var_c = [r[sc_key] for r in correct_results]
    sc_orig_i = [r["sc_orig"] for r in incorrect_results]
    sc_var_i = [r[sc_key] for r in incorrect_results]

    corr_c = pearson(sc_orig_c, sc_var_c)
    corr_i = pearson(sc_orig_i, sc_var_i)

    ax.scatter(sc_orig_i, sc_var_i, alpha=0.25, s=15, color=ORANGE, label=f"Incorrect (r={corr_i:.3f})")
    ax.scatter(sc_orig_c, sc_var_c, alpha=0.4, s=15, color=BLUE, label=f"Correct (r={corr_c:.3f})")
    all_sc = sc_orig_c + sc_orig_i + sc_var_c + sc_var_i
    lims = [min(all_sc), max(all_sc)]
    ax.plot(lims, lims, "--k", linewidth=1.5, alpha=0.5)
    ax.set_xlabel("SC (original)", fontsize=10)
    ax.set_ylabel(f"SC ({label.lower()})", fontsize=10)
    ax.set_title(f"SC Stability — Original vs {label}\n(both groups tight on diagonal)", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")

    # --- Middle: log(PPL) scatter, color by correctness ---
    ax = axes[row_idx][1]
    lp_orig_c = [math.log(r["ppl_orig"]) for r in correct_results if r["ppl_orig"] > 0]
    lp_var_c = [math.log(r[ppl_key]) for r in correct_results if r[ppl_key] > 0]
    lp_orig_i = [math.log(r["ppl_orig"]) for r in incorrect_results if r["ppl_orig"] > 0]
    lp_var_i = [math.log(r[ppl_key]) for r in incorrect_results if r[ppl_key] > 0]

    corr_c = pearson(lp_orig_c, lp_var_c)
    corr_i = pearson(lp_orig_i, lp_var_i)

    ax.scatter(lp_orig_i, lp_var_i, alpha=0.25, s=15, color=ORANGE, label=f"Incorrect (r={corr_i:.3f})")
    ax.scatter(lp_orig_c, lp_var_c, alpha=0.4, s=15, color=BLUE, label=f"Correct (r={corr_c:.3f})")
    all_lp = lp_orig_c + lp_orig_i + lp_var_c + lp_var_i
    lims = [min(all_lp), max(all_lp)]
    ax.plot(lims, lims, "--k", linewidth=1.5, alpha=0.5)
    ax.set_xlabel("log(PPL) (original)", fontsize=10)
    ax.set_ylabel(f"log(PPL) ({label.lower()})", fontsize=10)
    ax.set_title(f"log(PPL) Stability — Original vs {label}\n(incorrect group far more scattered)", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")

    # --- Right: Grouped bar chart ---
    ax = axes[row_idx][2]
    key_c = f"Correct_{label}"
    key_i = f"Incorrect_{label}"

    x = np.arange(2)
    width = 0.3
    vals_sc = [stats[key_c]["med_sc"], stats[key_i]["med_sc"]]
    vals_ppl = [stats[key_c]["med_ppl"], stats[key_i]["med_ppl"]]

    bars1 = ax.bar(x - width / 2, vals_sc, width, label="SC", color=BLUE, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, vals_ppl, width, label="PPL", color=ORANGE, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars1, vals_sc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold", color=BLUE)
    for bar, val in zip(bars2, vals_ppl):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold", color=ORANGE)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Correct\n(n={stats[key_c]['n']})", f"Incorrect\n(n={stats[key_i]['n']})"], fontsize=10)
    ax.set_ylabel("Median relative change (%)", fontsize=10)
    ax.set_title(f"Median |Δ| / original — {label}\n(lower = more stable)", fontsize=11)
    ax.set_ylim(0, max(max(vals_sc), max(vals_ppl)) * 1.3)
    ax.legend(fontsize=10)

fig.tight_layout()
out = os.path.join(FIGURE_DIR, "exp1_stability_by_correctness.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")
