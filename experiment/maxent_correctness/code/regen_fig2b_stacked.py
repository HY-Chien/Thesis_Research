"""
Regenerate fig2b_stacked_bar_accuracy_error.png with a clean layout.

Fix: the title is two lines and the legend was anchored at (0.5, 1.02) just
above the axes, so the legend text overlapped the title's second line.
Here the legend is moved BELOW the plot, fully separated from the title.

Reuses the data-loading / binning helpers from the original analysis script.
"""

import collections
import json
import re
import string
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

RESULTS_DIR = Path("/home/S113065528/Research/main_methodology/results")
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "2WikiMultiHopQA": "2wiki_1000_msample.jsonl",
    "HotpotQA": "hotpotqa_1000_msample.jsonl",
    "MuSiQue": "musique_1000_msample.jsonl",
    "PopQA": "popqa_1000_msample.jsonl",
}
SIGNAL = "mean_max_ent"


def normalize_answer(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def compute_f1(pred, gold):
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    common = collections.Counter(pred_toks) & collections.Counter(gold_toks)
    num = sum(common.values())
    if not pred_toks or not gold_toks:
        return int(pred_toks == gold_toks)
    if num == 0:
        return 0
    p = num / len(pred_toks)
    r = num / len(gold_toks)
    return 2 * p * r / (p + r)


def check_correct(prediction, ground_truths):
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    pred_n = normalize_answer(prediction)
    for gt in ground_truths:
        if pred_n == normalize_answer(gt):
            return True
        if compute_f1(prediction, gt) >= 0.5:
            return True
    return False


def load_dataset(path):
    ent, correct = [], []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if SIGNAL not in r:
                continue
            ent.append(float(r[SIGNAL]))
            correct.append(1 if check_correct(r["majority"], r["gold"]) else 0)
    return np.asarray(ent), np.asarray(correct)


def load_all():
    ent_all, cor_all = [], []
    for name, fname in DATASETS.items():
        ent, cor = load_dataset(RESULTS_DIR / fname)
        ent_all.append(ent)
        cor_all.append(cor)
    return np.concatenate(ent_all), np.concatenate(cor_all)


def equal_freq_bins(ent, cor, n_bins=8):
    order = np.argsort(ent)
    ent_s, cor_s = ent[order], cor[order]
    edges = np.linspace(0, len(ent_s), n_bins + 1).astype(int)
    centers, accs = [], []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo:
            continue
        chunk_ent = ent_s[lo:hi]
        chunk_cor = cor_s[lo:hi]
        k, n = int(chunk_cor.sum()), len(chunk_cor)
        centers.append(float(np.median(chunk_ent)))
        accs.append(100 * k / n)
    return np.array(centers), np.array(accs)


def fig_bar_contrast(ent, cor, n_bins=8):
    centers, accs = equal_freq_bins(ent, cor, n_bins=n_bins)
    err = 100 - accs
    rho, _ = sp_stats.spearmanr(ent, 1 - cor)

    x = np.arange(len(centers))
    w = 0.65

    fig, ax = plt.subplots(figsize=(10, 6.5))

    ax.bar(x, accs, w, color="#27ae60", label="Accuracy (correct %)",
           edgecolor="white", linewidth=1.5, zorder=3)
    ax.bar(x, err, w, bottom=accs, color="#c0392b", label="Error (incorrect %)",
           edgecolor="white", linewidth=1.5, zorder=3)

    for i in range(len(centers)):
        if accs[i] > 5:
            ax.text(x[i], accs[i] / 2, f"{accs[i]:.0f}%", ha="center",
                    va="center", color="white", fontweight="bold",
                    fontsize=11, zorder=4)
        if err[i] > 5:
            ax.text(x[i], accs[i] + err[i] / 2, f"{err[i]:.0f}%", ha="center",
                    va="center", color="white", fontweight="bold",
                    fontsize=11, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c:.2f}" for c in centers])
    ax.set_xlabel("MAX_ENTROPY bin (median value, low → high)", fontsize=13)
    ax.set_ylabel("Proportion (%)", fontsize=13)
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title("As MAX_ENTROPY rises: Error proportion replaces Accuracy\n"
                 f"(pooled, n = {len(ent)}, equal-frequency bins, "
                 f"Spearman(ent, err) = +{rho:.3f})",
                 fontsize=14, fontweight="bold", pad=12)

    # Legend moved BELOW the plot so it no longer collides with the title.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
              fontsize=12, frameon=False)

    ax.grid(alpha=0.3, axis="y", linestyle="--")

    fig.tight_layout()
    out = FIG_DIR / "fig2b_stacked_bar_accuracy_error.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    ent, cor = load_all()
    out = fig_bar_contrast(ent, cor)
    print("saved:", out)
