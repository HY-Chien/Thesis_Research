#!/usr/bin/env python3
"""Build an editable slide showing that equal entropy can hide different risk sources."""

from __future__ import annotations

from math import log
from pathlib import Path

from make_tl_sum_2slides import (
    COLORS,
    SlideCanvas,
    new_presentation,
    save_pdf,
    validate,
)


OUT_DIR = Path(__file__).resolve().parent
PPTX_OUT = OUT_DIR / "entropy_signal_vs_noise_slide.pptx"
PDF_OUT = OUT_DIR / "entropy_signal_vs_noise_slide.pdf"
PNG_OUT = OUT_DIR / "entropy_signal_vs_noise_slide.png"

NAVY = "1D4967"
NAVY_DARK = "173B54"
NAVY_LIGHT = "DCE7ED"
TAB_FILL = "EEF1F3"
TOP5 = "D88922"
TAIL = "4C86C6"
GREEN = "137455"
GREEN_LIGHT = "DCEFE7"
RED = "C9543E"
RED_LIGHT = "F2D8CF"


def shannon_entropy(probabilities: list[float]) -> float:
    return -sum(p * log(p) for p in probabilities if p > 0)


def deck_header(
    c: SlideCanvas,
    title: str,
    section: str,
    slide_no: str,
    title_size: float = 29.5,
) -> None:
    c.rect(10.88, 0.0, 2.45, 0.58, TAB_FILL)
    c.text(section, 10.98, 0.02, 2.18, 0.48, 18.0, bold=True, align="center")
    c.text(
        title,
        0.55,
        0.23,
        10.15,
        0.64,
        title_size,
        color=NAVY,
        bold=True,
    )
    c.line(0.55, 1.02, 11.76, NAVY, 0.012)
    c.line(0.55, 7.10, 11.88, NAVY, 0.012)
    c.ellipse(12.54, 6.43, 1.02, 1.02, "AEC1CC")
    c.text(
        slide_no,
        12.72,
        7.12,
        0.39,
        0.22,
        11.5,
        color=NAVY,
        align="right",
    )


def draw_rank_distribution_panel(
    c: SlideCanvas,
    x: float,
    title: str,
    head_probabilities: list[float],
    tail_probabilities: list[float],
    header_fill: str,
    header_color: str,
) -> None:
    """Draw a head-to-tail rank histogram in the visual style of the reference."""
    width = 5.90
    probabilities = [*head_probabilities, *tail_probabilities]
    entropy = shannon_entropy(probabilities)
    tail_mass = sum(tail_probabilities)
    active_tail = [p for p in tail_probabilities if p > 0]

    assert len(head_probabilities) == 5
    assert abs(sum(probabilities) - 1.0) < 1e-8
    assert min(head_probabilities) >= max(tail_probabilities)

    # Soft shadow and rounded white card.
    c.rect(x + 0.06, 1.70, width, 4.25, "D9E1E5", radius=0.15)
    c.rect(x, 1.62, width, 4.25, "FFFFFF", COLORS["panel_line"], 0.8, 0.15)
    c.rect(x + 0.20, 1.82, width - 0.40, 0.43, header_fill, radius=0.06)
    c.text(
        title,
        x + 0.30,
        1.79,
        width - 0.60,
        0.48,
        14.2,
        color=header_color,
        bold=True,
        family="sans",
        align="center",
    )

    c.text(
        "Same leaders:  PO = 0.38  ·  po = 0.26",
        x + 0.34,
        2.37,
        3.68,
        0.28,
        10.3,
        color=COLORS["muted"],
        family="sans",
    )
    c.rect(x + 4.16, 2.31, 1.30, 0.39, NAVY_LIGHT, radius=0.05)
    c.text(
        f"H ≈ {entropy:.2f}",
        x + 4.21,
        2.29,
        1.20,
        0.43,
        12.1,
        color=NAVY_DARK,
        bold=True,
        family="sans",
        align="center",
    )

    chart_x = x + 0.52
    chart_base = 4.91
    chart_height = 1.62
    max_probability = 0.40
    head_width = 0.36
    head_gap = 0.11
    head_colors = ["D98213", "E59622", "E8A33A", "ECB154", "F0BE70"]
    token_labels = ["PO", "po", "r3", "r4", "r5"]

    # Head bars: orange, descending by token rank.
    bar_x = chart_x
    for index, (probability, fill) in enumerate(zip(head_probabilities, head_colors)):
        height = probability / max_probability * chart_height
        c.rect(bar_x, chart_base - height, head_width, height, fill, radius=0.025)
        c.text(
            f"{probability:.2f}",
            bar_x - 0.08,
            chart_base - height - 0.27,
            head_width + 0.16,
            0.23,
            8.3,
            color=TOP5,
            bold=index < 2,
            family="sans",
            align="center",
        )
        c.text(
            token_labels[index],
            bar_x - 0.05,
            4.96,
            head_width + 0.10,
            0.22,
            8.3,
            color=COLORS["ink"],
            bold=index < 2,
            family="sans",
            align="center",
        )
        bar_x += head_width + head_gap

    # Tail bars: blue and intentionally thin, retaining all 12 vocabulary slots.
    tail_x = chart_x + 2.61
    tail_width = 0.12
    tail_gap = 0.065
    for probability in tail_probabilities:
        if probability > 0:
            height = max(0.025, probability / max_probability * chart_height)
            c.rect(tail_x, chart_base - height, tail_width, height, TAIL, radius=0.015)
        tail_x += tail_width + tail_gap

    # Shared baseline and the simple head/tail labels from the visual reference.
    c.line(chart_x - 0.03, chart_base, 4.82, "B8C7D0", 0.020)
    c.text(
        "head · ranks 1–5",
        chart_x + 0.15,
        5.20,
        1.92,
        0.26,
        10.0,
        color=TOP5,
        bold=True,
        family="sans",
        align="center",
    )
    c.text(
        "tail · ranks 6+",
        chart_x + 3.02,
        5.20,
        1.52,
        0.26,
        10.0,
        color=TAIL,
        bold=True,
        family="sans",
        align="center",
    )

    c.rect(x + 3.01, 3.35, 2.26, 0.57, "EDF5FB", radius=0.06)
    per_token = active_tail[0] if active_tail else 0.0
    c.text(
        f"tail probability = {tail_mass:.0%}\n{len(active_tail)} tokens × {per_token:.2%}",
        x + 3.08,
        3.34,
        2.12,
        0.59,
        10.1,
        color=TAIL,
        bold=True,
        family="sans",
        align="center",
    )

    result_fill = GREEN_LIGHT if tail_mass < 0.15 else RED_LIGHT
    result_color = GREEN if tail_mass < 0.15 else RED
    result_text = (
        "10% outside the top-5"
        if tail_mass < 0.15
        else "20% outside the top-5 (2×)"
    )
    c.rect(x + 1.40, 5.54, 3.10, 0.25, result_fill, radius=0.05)
    c.text(
        result_text,
        x + 1.45,
        5.51,
        3.00,
        0.32,
        10.3,
        color=result_color,
        bold=True,
        family="sans",
        align="center",
    )


def add_slide(prs) -> SlideCanvas:
    # Both examples use the same 17-token toy vocabulary and identical top two
    # probabilities. Seven negligible entries in the second example are shown
    # as zero so the support size stays fixed. Entropy matches to two decimals.
    head_variation = [0.38, 0.26, 0.11, 0.08, 0.07]
    head_variation_tail = [0.10 / 12] * 12
    tail_leakage = [0.38, 0.26, 0.06, 0.05, 0.05]
    tail_leakage_tail = [0.04] * 5 + [0.0] * 7
    entropy_a = shannon_entropy([*head_variation, *head_variation_tail])
    entropy_b = shannon_entropy([*tail_leakage, *tail_leakage_tail])
    assert round(entropy_a, 2) == round(entropy_b, 2) == 1.83

    c = SlideCanvas(prs)
    deck_header(c, "Same Entropy, Different Risk", "Methodology", "2")
    c.text(
        "The leading predictions and Shannon entropy are identical in both cases.",
        0.65,
        1.08,
        12.00,
        0.30,
        14.2,
        color=COLORS["muted"],
        italic=True,
    )

    draw_rank_distribution_panel(
        c,
        0.72,
        "A · 10% Probability in the Tail",
        head_variation,
        head_variation_tail,
        GREEN_LIGHT,
        GREEN,
    )
    draw_rank_distribution_panel(
        c,
        6.72,
        "B · 20% Probability in the Tail",
        tail_leakage,
        tail_leakage_tail,
        RED_LIGHT,
        RED,
    )

    c.rect(0.82, 6.20, 11.68, 0.55, NAVY_LIGHT, radius=0.06)
    c.text(
        "Same entropy, different tail risk: probability outside the top-5 doubles from 10% to 20%.",
        1.08,
        6.22,
        11.16,
        0.50,
        14.2,
        color=NAVY_DARK,
        bold=True,
        align="center",
    )
    return c


def main() -> None:
    prs = new_presentation()
    canvas = add_slide(prs)
    validate(prs, 1)
    prs.save(PPTX_OUT)
    save_pdf([canvas], PDF_OUT)
    canvas.image.save(PNG_OUT, "PNG", optimize=True)
    for path in (PPTX_OUT, PDF_OUT, PNG_OUT):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
