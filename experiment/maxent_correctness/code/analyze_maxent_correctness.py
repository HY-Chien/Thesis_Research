"""
MAX_ENTROPY vs. correctness analysis.

Question
--------
Does the model's MAX token entropy (per question, averaged over self-consistency
samples: the `mean_max_ent` field) correlate with whether the answer is correct?

Data
----
Self-consistency sampling results (`*_msample.jsonl`) produced by the SEER pipeline.
Each row has:
    - mean_max_ent : per-question MAX_ENTROPY signal (mean over SC samples of the
                     max per-token entropy of the sampled answer)
    - majority     : self-consistency majority answer
    - gold         : gold answer

Correctness is computed with the same normalized EM / token-F1>=0.5 rule used
elsewhere in the self_certainty experiments.

Key empirical finding (see figures / printed AUC):
    LOW  max_entropy  -> the model tends to be CORRECT  (it "knows")
    HIGH max_entropy  -> the model tends to be WRONG    (it "doesn't know")
i.e. max_entropy is POSITIVELY correlated with the error (1 - correct),
and NEGATIVELY correlated with accuracy. Both views are plotted.

Reuses the correctness/AUC logic from
    experiment/self_certainty/scripts/exp4_sc_gated_rag.py
    experiment/self_certainty/scripts/exp5b_correct_vs_incorrect_stats.py
"""

import collections
import json
import re
import string
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
RESULTS_DIR = Path("/home/S113065528/Research/main_methodology/results")
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "2WikiMultiHopQA": "2wiki_1000_msample.jsonl",
    "HotpotQA": "hotpotqa_1000_msample.jsonl",
    "MuSiQue": "musique_1000_msample.jsonl",
    "PopQA": "popqa_1000_msample.jsonl",
}

SIGNAL = "mean_max_ent"  # the MAX_ENTROPY signal


# ----------------------------------------------------------------------------
# Answer evaluation (identical rule to exp4_sc_gated_rag.py)
# ----------------------------------------------------------------------------
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


def auc(correct_scores, incorrect_scores, higher_is_confident=True):
    """P(a random correct sample scores 'more confident' than a random wrong one).
    For entropy, lower = more confident, so higher_is_confident=False."""
    if not correct_scores or not incorrect_scores:
        return float("nan")
    c = np.asarray(correct_scores)
    i = np.asarray(incorrect_scores)
    wins = 0.0
    for x in c:
        if higher_is_confident:
            wins += np.sum(x > i) + 0.5 * np.sum(x == i)
        else:
            wins += np.sum(x < i) + 0.5 * np.sum(x == i)
    return wins / (len(c) * len(i))


# ----------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------
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
    data = {}
    for name, fname in DATASETS.items():
        ent, cor = load_dataset(RESULTS_DIR / fname)
        data[name] = (ent, cor)
    ent_all = np.concatenate([data[d][0] for d in data])
    cor_all = np.concatenate([data[d][1] for d in data])
    data["ALL"] = (ent_all, cor_all)
    return data


# ----------------------------------------------------------------------------
# Binned accuracy with Wilson CI
# ----------------------------------------------------------------------------
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return center - half, center + half


def equal_freq_bins(ent, cor, n_bins=10):
    """Equal-frequency (quantile) bins -> each point has equal weight."""
    order = np.argsort(ent)
    ent_s, cor_s = ent[order], cor[order]
    edges = np.linspace(0, len(ent_s), n_bins + 1).astype(int)
    centers, accs, los, his, ns = [], [], [], [], []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo:
            continue
        chunk_ent = ent_s[lo:hi]
        chunk_cor = cor_s[lo:hi]
        k, n = int(chunk_cor.sum()), len(chunk_cor)
        centers.append(float(np.median(chunk_ent)))
        accs.append(100 * k / n)
        cl, ch = wilson_ci(k, n)
        los.append(100 * cl)
        his.append(100 * ch)
        ns.append(n)
    return (np.array(centers), np.array(accs), np.array(los),
            np.array(his), np.array(ns))


# ----------------------------------------------------------------------------
# Stats helpers
# ----------------------------------------------------------------------------
def correlations(ent, cor):
    # point-biserial r between entropy and correctness (expect negative)
    r_pb, p_pb = sp_stats.pointbiserialr(cor, ent)
    # Spearman between entropy and ERROR (expect positive)
    err = 1 - cor
    rho, p_rho = sp_stats.spearmanr(ent, err)
    a = auc(ent[cor == 1].tolist(), ent[cor == 0].tolist(),
            higher_is_confident=False)
    return r_pb, p_pb, rho, p_rho, a


# ----------------------------------------------------------------------------
# Figure 1 (MAIN): pooled accuracy vs max_entropy  — clean, large, monotone
# ----------------------------------------------------------------------------
def fig_main(data):
    ent, cor = data["ALL"]
    centers, accs, los, his, ns = equal_freq_bins(ent, cor, n_bins=10)
    r_pb, p_pb, rho, p_rho, a = correlations(ent, cor)

    fig, ax = plt.subplots(figsize=(9, 6))
    yerr = np.vstack([accs - los, his - accs])
    ax.errorbar(centers, accs, yerr=yerr, fmt="o-", color="#1f4e79",
                lw=2.5, ms=9, capsize=4, mfc="#1f4e79", mec="white",
                label="Accuracy per max-entropy bin (95% Wilson CI)", zorder=3)

    # linear trend line over bin points
    m, b = np.polyfit(centers, accs, 1)
    xs = np.linspace(centers.min(), centers.max(), 100)
    ax.plot(xs, m * xs + b, "--", color="#c0392b", lw=2,
            label=f"Linear fit (slope = {m:.1f} acc%/nat)", zorder=2)

    ax.set_xlabel("MAX_ENTROPY  (per-question max token entropy)", fontsize=13)
    ax.set_ylabel("Exact-Match Accuracy  (%)", fontsize=13)
    ax.set_title("Lower MAX_ENTROPY $\\rightarrow$ the model is more likely correct\n"
                 "(pooled over 4 datasets, n = %d, equal-frequency bins)" % len(ent),
                 fontsize=14, fontweight="bold")
    txt = (f"Spearman(MAX_ENTROPY, error) = +{rho:.3f}  (p={p_rho:.1e})\n"
           f"Point-biserial(MAX_ENTROPY, correct) = {r_pb:.3f}\n"
           f"AUROC (low entropy predicts correct) = {a:.3f}")
    ax.text(0.97, 0.95, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=11, bbox=dict(boxstyle="round", fc="#fdf6e3", ec="#999"))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "fig1_main_maxent_vs_accuracy.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
# Figure 2: the POSITIVE-correlation view — P(incorrect) rises with entropy
# ----------------------------------------------------------------------------
def fig_positive(data):
    ent, cor = data["ALL"]
    centers, accs, los, his, ns = equal_freq_bins(ent, cor, n_bins=10)
    err = 100 - accs
    elo, ehi = 100 - his, 100 - los
    rho, p_rho = sp_stats.spearmanr(ent, 1 - cor)

    fig, ax = plt.subplots(figsize=(9, 6))
    yerr = np.vstack([err - elo, ehi - err])
    ax.errorbar(centers, err, yerr=yerr, fmt="s-", color="#922b21",
                lw=2.5, ms=9, capsize=4, mec="white",
                label="Error rate per max-entropy bin (95% CI)", zorder=3)
    m, b = np.polyfit(centers, err, 1)
    xs = np.linspace(centers.min(), centers.max(), 100)
    ax.plot(xs, m * xs + b, "--", color="#1f4e79", lw=2,
            label=f"Linear fit (slope = +{m:.1f} err%/nat)")
    ax.set_xlabel("MAX_ENTROPY  (per-question max token entropy)", fontsize=13)
    ax.set_ylabel("Error Rate  (%)", fontsize=13)
    ax.set_title("MAX_ENTROPY is positively correlated with answering INCORRECTLY\n"
                 f"Spearman r = +{rho:.3f}  (pooled, n = {len(ent)})",
                 fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "fig2_maxent_vs_error_positive.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
# Figure 3: distribution of max_entropy, correct vs incorrect
# ----------------------------------------------------------------------------
def fig_distribution(data):
    ent, cor = data["ALL"]
    c_ent = ent[cor == 1]
    i_ent = ent[cor == 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    bins = np.linspace(0, np.percentile(ent, 99), 35)
    ax1.hist(c_ent, bins=bins, density=True, alpha=0.6, color="#27ae60",
             label=f"Correct (n={len(c_ent)})")
    ax1.hist(i_ent, bins=bins, density=True, alpha=0.6, color="#c0392b",
             label=f"Incorrect (n={len(i_ent)})")
    ax1.axvline(c_ent.mean(), color="#27ae60", ls="--", lw=2)
    ax1.axvline(i_ent.mean(), color="#c0392b", ls="--", lw=2)
    ax1.set_xlabel("MAX_ENTROPY", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Correct answers cluster at LOW max-entropy", fontsize=13,
                  fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    parts = ax2.violinplot([c_ent, i_ent], showmeans=True, showmedians=True)
    for pc, col in zip(parts["bodies"], ["#27ae60", "#c0392b"]):
        pc.set_facecolor(col)
        pc.set_alpha(0.6)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["Correct", "Incorrect"], fontsize=12)
    ax2.set_ylabel("MAX_ENTROPY", fontsize=12)
    ax2.set_title(f"mean: {c_ent.mean():.3f}  vs  {i_ent.mean():.3f}",
                  fontsize=13, fontweight="bold")
    ax2.grid(alpha=0.3)

    fig.suptitle("Distribution of MAX_ENTROPY by correctness (pooled, n=%d)" % len(ent),
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = FIG_DIR / "fig3_distribution_correct_vs_incorrect.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
# Figure 2b (BAR): grouped bars per entropy bin — accuracy DOWN vs error UP
# ----------------------------------------------------------------------------
def fig_bar_contrast(data, n_bins=8):
    ent, cor = data["ALL"]
    centers, accs, los, his, ns = equal_freq_bins(ent, cor, n_bins=n_bins)
    err = 100 - accs
    rho, _ = sp_stats.spearmanr(ent, 1 - cor)

    x = np.arange(n_bins)
    w = 0.42
    fig, ax = plt.subplots(figsize=(11, 6.5))
    b1 = ax.bar(x - w / 2, accs, w, color="#27ae60", label="Accuracy (correct %)",
                edgecolor="white", zorder=3)
    b2 = ax.bar(x + w / 2, err, w, color="#c0392b", label="Error (incorrect %)",
                edgecolor="white", zorder=3)
    # trend lines on top of the bars to make "down vs up" explicit
    ax.plot(x - w / 2, accs, "o-", color="#145a32", lw=2.2, ms=6, zorder=4)
    ax.plot(x + w / 2, err, "s-", color="#7b241c", lw=2.2, ms=6, zorder=4)
    ax.annotate("", xy=(x[-1] - w / 2, accs[-1] + 4), xytext=(x[0] - w / 2, accs[0] + 4),
                arrowprops=dict(arrowstyle="->", color="#145a32", lw=2))
    ax.text(x.mean() - w / 2, max(accs) + 7, "accuracy ↓", color="#145a32",
            fontsize=12, fontweight="bold", ha="center")
    ax.annotate("", xy=(x[-1] + w / 2, err[-1] + 4), xytext=(x[0] + w / 2, err[0] + 4),
                arrowprops=dict(arrowstyle="->", color="#7b241c", lw=2))
    ax.text(x.mean() + w / 2, max(err) + 7, "error ↑", color="#7b241c",
            fontsize=12, fontweight="bold", ha="center")

    for rect in list(b1) + list(b2):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 0.8, f"{h:.0f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c:.2f}" for c in centers])
    ax.set_xlabel("MAX_ENTROPY bin (median value, low → high)", fontsize=13)
    ax.set_ylabel("Percentage within bin (%)", fontsize=13)
    ax.set_ylim(0, 100)
    ax.set_title("As MAX_ENTROPY rises: accuracy goes DOWN, error goes UP\n"
                 f"(pooled, n = {len(ent)}, equal-frequency bins, "
                 f"Spearman(ent,err) = +{rho:.3f})",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="center left", fontsize=12, framealpha=0.95)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = FIG_DIR / "fig2b_bar_accuracy_down_error_up.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
# Figure 4: per-dataset 2x2 grid of accuracy vs entropy bins
# ----------------------------------------------------------------------------
def fig_per_dataset(data):
    names = [d for d in DATASETS]  # 4 datasets
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, name in zip(axes.flat, names):
        ent, cor = data[name]
        nb = 8
        centers, accs, los, his, ns = equal_freq_bins(ent, cor, n_bins=nb)
        r_pb, p_pb, rho, p_rho, a = correlations(ent, cor)
        yerr = np.vstack([accs - los, his - accs])
        ax.errorbar(centers, accs, yerr=yerr, fmt="o-", color="#1f4e79",
                    lw=2, ms=7, capsize=3, mec="white")
        m, b = np.polyfit(centers, accs, 1)
        xs = np.linspace(centers.min(), centers.max(), 50)
        ax.plot(xs, m * xs + b, "--", color="#c0392b", lw=1.8)
        ax.set_title(f"{name}   (acc={100*cor.mean():.1f}%, AUROC={a:.3f})",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("MAX_ENTROPY", fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.grid(alpha=0.3)
    fig.suptitle("MAX_ENTROPY vs accuracy per dataset (equal-frequency bins, 95% CI)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    out = FIG_DIR / "fig4_per_dataset_grid.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
def write_summary(data):
    lines = ["# MAX_ENTROPY vs Correctness — summary\n",
             "Signal: `mean_max_ent` from `*_msample.jsonl` (self-consistency).",
             "Correctness: normalized EM / token-F1>=0.5 of SC majority vs gold.\n",
             "| Dataset | n | Acc% | mean ent (correct) | mean ent (wrong) | "
             "Spearman(ent,err) | point-biserial(ent,correct) | AUROC |",
             "|---|---|---|---|---|---|---|---|"]
    for name in list(DATASETS) + ["ALL"]:
        ent, cor = data[name]
        r_pb, p_pb, rho, p_rho, a = correlations(ent, cor)
        ce = ent[cor == 1].mean()
        ie = ent[cor == 0].mean()
        lines.append(f"| {name} | {len(ent)} | {100*cor.mean():.1f} | "
                     f"{ce:.3f} | {ie:.3f} | +{rho:.3f} | {r_pb:.3f} | {a:.3f} |")
    out = FIG_DIR.parent / "RESULTS.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def main():
    data = load_all()
    print("Loaded:", {k: len(v[0]) for k, v in data.items()})
    outs = [
        fig_main(data),
        fig_positive(data),
        fig_bar_contrast(data),
        fig_distribution(data),
        fig_per_dataset(data),
        write_summary(data),
    ]
    for o in outs:
        print("wrote", o)


if __name__ == "__main__":
    main()
