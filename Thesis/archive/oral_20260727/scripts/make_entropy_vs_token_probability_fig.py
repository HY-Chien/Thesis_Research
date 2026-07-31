from __future__ import annotations

from math import log
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"


NAVY = "#10233f"
TEAL = "#0f9f7a"
RED = "#ef4444"
ORANGE = "#f59e0b"
BLUE = "#2b6cb0"
GRAY = "#64748b"
LIGHT = "#f3f6fb"
LINE = "#d8e0eb"
BLACK = "#0f172a"


def entropy(probs: list[float]) -> float:
    return -sum(p * log(p) for p in probs if p > 0)


def add_round_box(ax, xy, width, height, fc="white", ec=LINE, lw=1.4, radius=0.035):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(box)
    return box


def add_text(ax, x, y, text, size=18, weight="normal", color=BLACK, ha="left", va="top"):
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        family="DejaVu Sans",
    )


def draw_distribution(ax, x, y, w, h, title, probs, entropy_value, mode):
    add_round_box(ax, (x, y), w, h, fc="white", ec=LINE, lw=1.2, radius=0.02)
    add_text(ax, x + 0.025, y + h - 0.04, title, size=17, weight="bold", color=NAVY)

    chart_x = x + 0.055
    chart_y = y + 0.145
    chart_w = w - 0.11
    chart_h = h - 0.245
    n = len(probs)
    gap = chart_w / n * 0.28
    bar_w = chart_w / n * 0.72
    max_p = 0.60

    ax.plot(
        [chart_x - 0.01, chart_x + chart_w + 0.01],
        [chart_y, chart_y],
        transform=ax.transAxes,
        color=LINE,
        lw=1.2,
    )

    for i, p_i in enumerate(probs):
        bx = chart_x + i * (bar_w + gap)
        bh = chart_h * (p_i / max_p)
        color = TEAL if i == 0 else (ORANGE if i == 1 else "#93a4bd")
        ax.add_patch(
            Rectangle(
                (bx, chart_y),
                bar_w,
                bh,
                transform=ax.transAxes,
                color=color,
                alpha=0.96 if i < 2 else 0.68,
            )
        )
        add_text(
            ax,
            bx + bar_w / 2,
            chart_y - 0.018,
            f"{p_i:.2f}",
            size=10.5,
            weight="bold" if i == 0 else "normal",
            color=BLACK if i == 0 else GRAY,
            ha="center",
            va="top",
        )

    if mode == "broad":
        pill_color = RED
        pill_text = f"H = {entropy_value:.2f} (High)"
    else:
        pill_color = TEAL
        pill_text = f"H = {entropy_value:.2f} (Lower)"

    pill_w = min(0.24, w - 0.12)
    ax.add_patch(
        FancyBboxPatch(
            (x + (w - pill_w) / 2, y + 0.04),
            pill_w,
            0.052,
            boxstyle="round,pad=0.008,rounding_size=0.02",
            linewidth=0,
            facecolor=pill_color,
            transform=ax.transAxes,
            clip_on=False,
        )
    )
    add_text(
        ax,
        x + w / 2,
        y + 0.066,
        pill_text,
        size=11.6,
        color="white",
        weight="bold",
        ha="center",
        va="center",
    )

def make_full_slide():
    broad = [0.55, 0.1125, 0.1125, 0.1125, 0.1125]
    concentrated = [0.55, 0.42, 0.01, 0.01, 0.01]
    h_broad = entropy(broad)
    h_conc = entropy(concentrated)

    fig = plt.figure(figsize=(16, 9), dpi=180)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    fig.patch.set_facecolor("white")

    add_text(
        ax,
        0.055,
        0.91,
        "Token probability sees one generated choice; entropy sees the whole next-token distribution.",
        size=23,
        color=GRAY,
        va="top",
    )

    add_round_box(ax, (0.065, 0.71), 0.37, 0.13, fc="white", ec=LINE, lw=1.3, radius=0.022)
    add_text(ax, 0.092, 0.795, "One-token probability", size=19, weight="bold", color=NAVY, va="top")
    add_text(ax, 0.092, 0.752, "Only checks p(generated token)", size=14.5, color=GRAY, va="top")
    add_text(ax, 0.332, 0.775, "0.55", size=24, weight="bold", color=TEAL, ha="center", va="center")

    add_round_box(ax, (0.565, 0.71), 0.37, 0.13, fc="white", ec=LINE, lw=1.3, radius=0.022)
    add_text(ax, 0.592, 0.795, "Entropy", size=19, weight="bold", color=NAVY, va="top")
    add_text(ax, 0.592, 0.752, "Summarizes all candidate tokens", size=14.5, color=GRAY, va="top")
    add_text(ax, 0.785, 0.775, r"$H = -\sum_i p_i \log p_i$", size=21, weight="bold", color=TEAL, ha="center", va="center")

    ax.annotate(
        "",
        xy=(0.535, 0.775),
        xytext=(0.457, 0.775),
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="->", lw=2.5, color=GRAY),
    )

    draw_distribution(ax, 0.055, 0.14, 0.43, 0.46, "Scenario A: Broad uncertainty", broad, h_broad, "broad")
    draw_distribution(ax, 0.515, 0.14, 0.43, 0.46, "Scenario B: Concentrated alternatives", concentrated, h_conc, "concentrated")

    slide_png = FIG_DIR / "slide_entropy_vs_token_probability.png"
    slide_pdf = FIG_DIR / "slide_entropy_vs_token_probability.pdf"
    fig.savefig(slide_png, facecolor="white")
    fig.savefig(slide_pdf, facecolor="white")
    plt.close(fig)


def make_standalone_figure():
    broad = [0.55, 0.1125, 0.1125, 0.1125, 0.1125]
    concentrated = [0.55, 0.42, 0.01, 0.01, 0.01]
    h_broad = entropy(broad)
    h_conc = entropy(concentrated)

    fig = plt.figure(figsize=(12, 5.4), dpi=180)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    fig.patch.set_facecolor("white")

    add_text(ax, 0.035, 0.94, "Single-token probability cannot see the full uncertainty structure", size=22, weight="bold", color=NAVY)
    add_text(ax, 0.035, 0.875, "Both examples generate the same token with p = 0.55, but entropy separates broad confusion from a narrow alternative.", size=13.5, color=GRAY)

    draw_distribution(ax, 0.045, 0.16, 0.43, 0.58, "Broad uncertainty", broad, h_broad, "broad")
    draw_distribution(ax, 0.525, 0.16, 0.43, 0.58, "Concentrated alternatives", concentrated, h_conc, "concentrated")

    fig_png = FIG_DIR / "fig_entropy_vs_token_probability.png"
    fig_pdf = FIG_DIR / "fig_entropy_vs_token_probability.pdf"
    fig.savefig(fig_png, bbox_inches="tight", facecolor="white")
    fig.savefig(fig_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    make_full_slide()
    make_standalone_figure()
    print(f"Wrote {FIG_DIR / 'slide_entropy_vs_token_probability.png'}")
    print(f"Wrote {FIG_DIR / 'fig_entropy_vs_token_probability.png'}")


if __name__ == "__main__":
    main()
