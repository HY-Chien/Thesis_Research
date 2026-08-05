#!/usr/bin/env python3
"""Build an editable teaching slide for top-k retained mass and tail leakage."""

from __future__ import annotations

from pathlib import Path

from make_tl_sum_2slides import (
    COLORS,
    SlideCanvas,
    new_presentation,
    save_pdf,
    validate,
)


OUT_DIR = Path(__file__).resolve().parent
PPTX_OUT = OUT_DIR / "tail_signal_motivation_slide.pptx"
PDF_OUT = OUT_DIR / "tail_signal_motivation_slide.pdf"
PNG_OUT = OUT_DIR / "tail_signal_motivation_slide.png"

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


def definition_box(
    c: SlideCanvas,
    x: float,
    label: str,
    detail: str,
    accent: str,
) -> None:
    c.rect(x, 1.48, 3.42, 0.66, "FFFFFF", COLORS["panel_line"], 0.8, 0.06)
    c.rect(x + 0.16, 1.62, 0.86, 0.38, accent, radius=0.05)
    c.text(
        label,
        x + 0.16,
        1.59,
        0.86,
        0.43,
        13.0,
        color="FFFFFF",
        bold=True,
        family="sans",
        align="center",
    )
    c.text(
        detail,
        x + 1.13,
        1.54,
        2.12,
        0.54,
        10.5,
        color=COLORS["muted"],
        family="sans",
        align="center",
    )


def candidate_chip(
    c: SlideCanvas,
    x: float,
    y: float,
    token: str,
    probability: float,
    fill: str,
) -> None:
    c.rect(x, y, 0.88, 0.58, fill, radius=0.06)
    c.text(
        f"{token}\n{probability:.2f}",
        x + 0.03,
        y + 0.02,
        0.82,
        0.53,
        10.2,
        color="FFFFFF",
        bold=True,
        family="sans",
        align="center",
    )


def draw_case(
    c: SlideCanvas,
    x: float,
    title: str,
    subtitle: str,
    probabilities: list[float],
    tail_mass: float,
    result: str,
    result_fill: str,
    result_color: str,
) -> None:
    width = 5.72
    c.rect(x, 2.38, width, 3.62, "FFFFFF", COLORS["panel_line"], 0.8, 0.07)
    c.rect(x + 0.18, 2.55, width - 0.36, 0.38, result_fill, radius=0.05)
    c.text(
        title,
        x + 0.24,
        2.52,
        width - 0.48,
        0.44,
        14.2,
        color=result_color,
        bold=True,
        family="sans",
        align="center",
    )
    c.text(
        subtitle,
        x + 0.27,
        3.00,
        width - 0.54,
        0.28,
        10.4,
        color=COLORS["muted"],
        family="sans",
        align="center",
    )

    c.rect(x + 1.28, 3.36, 3.16, 0.42, NAVY_LIGHT, radius=0.06)
    c.text(
        'Top-1:  P("PO") = 0.60',
        x + 1.31,
        3.33,
        3.10,
        0.48,
        15.0,
        color=NAVY_DARK,
        bold=True,
        family="sans",
        align="center",
    )

    c.text(
        "Five most likely next-token candidates",
        x + 0.43,
        3.89,
        4.86,
        0.24,
        9.8,
        color=COLORS["muted"],
        family="sans",
        align="center",
    )
    tokens = ["PO", "po", "Po", "P", "pO"]
    fills = [NAVY, "668DA5", TOP5, TOP5, TOP5]
    start_x = x + 0.49
    for index, (token, probability, fill) in enumerate(zip(tokens, probabilities, fills)):
        candidate_chip(c, start_x + index * 0.96, 4.17, token, probability, fill)

    retained_mass = sum(probabilities)
    bar_x = x + 0.50
    bar_y = 4.96
    bar_width = 4.72
    c.text(
        "Where the full 100% probability mass goes",
        x + 0.44,
        4.72,
        4.84,
        0.22,
        9.4,
        color=COLORS["muted"],
        family="sans",
        align="center",
    )
    c.rect(bar_x, bar_y, bar_width, 0.34, COLORS["track"], radius=0.05)
    retained_width = bar_width * retained_mass
    c.rect(bar_x, bar_y, retained_width, 0.34, TOP5, radius=0.05)
    c.rect(
        bar_x + retained_width,
        bar_y,
        bar_width * tail_mass,
        0.34,
        TAIL,
        radius=0.05,
    )
    c.text(
        f"top-5 retained  {retained_mass:.0%}",
        bar_x,
        5.32,
        2.72,
        0.25,
        10.4,
        color=TOP5,
        bold=True,
        family="sans",
    )
    c.text(
        f"tail  {tail_mass:.0%}",
        bar_x + 3.02,
        5.32,
        1.70,
        0.25,
        10.4,
        color=TAIL,
        bold=True,
        family="sans",
        align="right",
    )
    c.rect(x + 1.03, 5.63, 3.66, 0.25, result_fill, radius=0.05)
    c.text(
        result,
        x + 1.06,
        5.59,
        3.60,
        0.33,
        10.8,
        color=result_color,
        bold=True,
        family="sans",
        align="center",
    )


def add_slide(prs) -> SlideCanvas:
    c = SlideCanvas(prs)
    deck_header(c, "Same Strong Top Prediction, Different Confidence", "Methodology", "1", 28.5)
    c.text(
        'Illustrative example: "PO" has probability 0.60 in both cases. The warning is probability escaping behind it.',
        0.65,
        1.08,
        12.00,
        0.30,
        14.0,
        color=COLORS["muted"],
        italic=True,
    )

    definition_box(c, 0.72, "TOP-5", "the five highest-probability next tokens", TOP5)
    c.text("→", 4.19, 1.60, 0.34, 0.38, 18.0, color=NAVY, bold=True, align="center")
    definition_box(c, 4.55, "m₅", "their probability mass added together", NAVY)
    c.text("→", 8.02, 1.60, 0.34, 0.38, 18.0, color=NAVY, bold=True, align="center")
    definition_box(c, 8.38, "TAIL", "every token ranked 6 through |V|", TAIL)

    draw_case(
        c,
        0.72,
        "A · Mostly harmless case variation",
        'The head is filled with plausible forms such as "PO" and "po".',
        [0.60, 0.22, 0.07, 0.05, 0.03],
        0.03,
        "Only 3% escapes → higher confidence",
        GREEN_LIGHT,
        GREEN,
    )
    draw_case(
        c,
        6.78,
        "B · Strong top-1, but probability leaks",
        "The top token is unchanged, yet much more mass moves outside the candidate set.",
        [0.60, 0.08, 0.04, 0.03, 0.03],
        0.22,
        "22% escapes → lower confidence",
        RED_LIGHT,
        RED,
    )

    c.rect(0.82, 6.16, 11.68, 0.72, NAVY_LIGHT, radius=0.06)
    c.text(
        "Why k = 5?",
        1.04,
        6.25,
        1.47,
        0.50,
        15.0,
        color=NAVY_DARK,
        bold=True,
        family="sans",
        align="center",
    )
    metrics = [
        (2.65, "k=2", ".741", "AEB8BE", "FFFFFF"),
        (3.78, "k=5", ".749", GREEN, "FFFFFF"),
        (4.91, "k=10", ".750", "6F9C8B", "FFFFFF"),
    ]
    for x, label, value, fill, text_color in metrics:
        c.rect(x, 6.27, 0.98, 0.43, fill, radius=0.05)
        c.text(
            f"{label}  {value}",
            x + 0.03,
            6.24,
            0.92,
            0.49,
            9.6,
            color=text_color,
            bold=True,
            family="sans",
            align="center",
        )
    c.text(
        "AUROC: k=5 is the smallest cutoff on the performance plateau—not a mathematical constant.",
        6.08,
        6.23,
        6.12,
        0.52,
        11.2,
        color=NAVY_DARK,
        bold=True,
        family="sans",
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
