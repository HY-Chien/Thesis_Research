"""Generate thesis-quality figures for Chapter 3, Section 3.4 (tail-leakage signal).

Outputs four vector PDFs into figures/:
  fig_renyi_reweight_auroc.pdf   - Renyi escort reweighting + 2Wiki AUROC bars
  fig_same_entropy_risk.pdf      - corrected "same entropy, different risk" example
  fig_leakage_monotonicity.pdf   - binary entropy vs -log m as leakage scores
  fig_tail_leakage_error_bins.pdf- error rate across tail-leakage score bins
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paul Tol "vibrant" hues (CVD-safe): blue, orange, green, red, gray
BLUE, ORANGE, GREEN, RED, GRAY = "#4477AA", "#EE7733", "#228833", "#CC3311", "#BBBBBB"
INK = "#333333"

plt.rcParams.update({
    "font.size": 8.5,
    "axes.edgecolor": "#999999",
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "figure.dpi": 200,
})

FIGDIR = "figures"

# Corrected example distributions (speaker-script numbers, entropy ~1.83 both)
HEAD_A = [0.38, 0.26, 0.11, 0.08, 0.07]
TAIL_A = [0.10 / 12] * 12
HEAD_B = [0.52, 0.13, 0.05, 0.05, 0.05]
TAIL_B = [0.03] * 5 + [0.05 / 7] * 7
HEAD_LABELS = ["PO", "po", "r3", "r4", "r5"]


def entropy(p):
    p = np.asarray(p)
    return float(-(p * np.log(p)).sum())


def style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.5)


# ---------------------------------------------------------------- figure 1
def fig_renyi():
    base = np.array(HEAD_A + TAIL_A)
    fig, axes = plt.subplots(
        1, 4, figsize=(6.6, 1.9), gridspec_kw={"width_ratios": [1, 1, 1, 1.15]}
    )

    settings = [(2.0, r"$\alpha=2$ (head-sensitive)"),
                (1.0, r"$\alpha\rightarrow 1$ (Shannon)"),
                (0.5, r"$\alpha=0.5$ (tail-sensitive)")]
    x = np.arange(len(base))
    for ax, (alpha, title) in zip(axes[:3], settings):
        q = base ** alpha
        q = q / q.sum()
        colors = [ORANGE] * 5 + [BLUE] * 12
        ax.bar(x, q, width=0.82, color=colors, edgecolor="none")
        ax.set_title(title, fontsize=8.5, color=INK, pad=4)
        ax.set_ylim(0, 0.65)
        ax.set_xticks([2, 10.5])
        ax.set_xticklabels(["head", "tail"], fontsize=8)
        ax.set_yticks([0, 0.2, 0.4, 0.6])
        style_axis(ax)
        if ax is axes[0]:
            ax.set_ylabel("escort probability", fontsize=8.5)
        else:
            ax.set_yticklabels([])

    ax = axes[3]
    aurocs = [0.651, 0.671, 0.716]
    labels = [r"$\alpha=2$", r"$\alpha\rightarrow 1$", r"$\alpha=0.5$"]
    colors = [GRAY, GRAY, GREEN]
    bars = ax.bar(np.arange(3), aurocs, width=0.62, color=colors, edgecolor="none")
    for b, v in zip(bars, aurocs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0.60, 0.74)
    ax.set_yticks([0.60, 0.65, 0.70])
    ax.set_ylabel("error-detection AUROC", fontsize=8.5)
    style_axis(ax)

    fig.tight_layout(w_pad=1.4)
    fig.savefig(f"{FIGDIR}/fig_renyi_reweight_auroc.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_same_entropy():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.2), sharey=True)

    for ax, head, tail, name in (
        (axes[0], HEAD_A, TAIL_A, "A"),
        (axes[1], HEAD_B, TAIL_B, "B"),
    ):
        probs = head + tail
        n = len(probs)
        x = np.arange(n)
        colors = [ORANGE] * 5 + [BLUE] * len(tail)
        ax.bar(x, probs, width=0.82, color=colors, edgecolor="none")
        m = sum(head)
        h = entropy(probs)
        ax.set_title(
            f"Distribution {name} · tail mass {1 - m:.0%}",
            fontsize=9.5, color=INK, pad=5,
        )
        ax.text(
            0.97, 0.72,
            f"top-5 mass = {m:.2f}\nentropy = {h:.2f}\nleakage score = {-np.log(m):.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.45", fc="#F5F5F5", ec="#CCCCCC", lw=0.7),
        )
        ax.set_xticks(list(range(5)) + [5 + (len(tail) - 1) / 2])
        ax.set_xticklabels(HEAD_LABELS + [f"tail (6–{5 + len(tail)})"],
                           fontsize=7.5)
        ax.set_xlim(-0.7, 16.7)
        ax.set_ylim(0, 0.56)
        style_axis(ax)

    axes[0].set_ylabel("next-token probability", fontsize=9)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(f"{FIGDIR}/fig_same_entropy_risk.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_monotonicity():
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.1))
    ell = np.linspace(1e-4, 0.999, 600)
    m = 1 - ell

    ax = axes[0]
    hb = -(m * np.log(m) + ell * np.log(ell))
    ax.plot(ell, hb, color=RED, lw=1.8)
    ax.axvline(0.5, color="#999999", lw=0.8, ls=(0, (4, 3)))
    ax.fill_between(ell, hb, where=ell >= 0.5, color=RED, alpha=0.10, lw=0)
    ax.annotate("peak at $\\ell=0.5$", xy=(0.5, np.log(2)), xytext=(0.60, 0.74),
                fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="-", color="#999999", lw=0.8))
    ax.text(0.78, 0.22, "more leakage,\nlower score", fontsize=8.5,
            color=RED, ha="center")
    ax.set_title("binary entropy\n" + r"$-m_t\log m_t-(1-m_t)\log(1-m_t)$", fontsize=9.0, color=INK)
    ax.set_ylim(0, 0.85)

    ax = axes[1]
    keep = ell <= 0.96
    ax.plot(ell[keep], -np.log(m[keep]), color=GREEN, lw=1.8)
    ax.text(0.60, 2.35, "strictly increasing", fontsize=8.5, color=GREEN)
    ax.set_title(r"Tail Leakage $-\log m_t$", fontsize=9.5, color=INK)
    ax.set_ylim(0, 3.4)

    for ax in axes:
        ax.set_xlabel(r"leaked mass $\ell = 1 - m_t$", fontsize=9)
        ax.set_xlim(0, 1)
        style_axis(ax)
    axes[0].set_ylabel("score", fontsize=9)

    fig.tight_layout(w_pad=2.2)
    fig.savefig(f"{FIGDIR}/fig_leakage_monotonicity.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def fig_error_bins():
    bins = ["0.0002", "0.0012", "0.0044", "0.0119",
            "0.0295", "0.0660", "0.173", "0.523"]
    acc = np.array([67, 38, 26, 24, 14, 12, 9, 3])
    err = 100 - acc

    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    x = np.arange(len(bins))
    ax.bar(x, acc, width=0.74, color=GREEN, edgecolor="white", linewidth=1.2,
           label="correct (%)")
    ax.bar(x, err, width=0.74, bottom=acc, color=RED, edgecolor="white",
           linewidth=1.2, label="incorrect (%)")
    for i in range(len(bins)):
        if acc[i] >= 8:
            ax.text(i, acc[i] / 2, f"{acc[i]}", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
        else:
            ax.text(i, acc[i] + 4.5, f"{acc[i]}", ha="center", va="center",
                    fontsize=8, color=GREEN, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.22", fc="white",
                              ec="none"))
        ax.text(i, acc[i] + err[i] / 2, f"{err[i]}", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(bins, fontsize=8.5)
    ax.set_xlabel(r"Tail Leakage $\sigma^{\mathrm{tl}}$"
                  " (equal-frequency bins, low → high)", fontsize=9)
    ax.set_ylabel("proportion (%)", fontsize=9)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2,
              fontsize=8, frameon=False)
    style_axis(ax)

    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig_tail_leakage_error_bins.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_renyi()
    fig_same_entropy()
    fig_monotonicity()
    fig_error_bins()
    print("entropy A:", entropy(np.array(HEAD_A + TAIL_A)))
    print("entropy B:", entropy(np.array(HEAD_B + TAIL_B)))
    print("done")
