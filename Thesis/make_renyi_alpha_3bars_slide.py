#!/usr/bin/env python3
"""Build a standalone, editable 16:9 slide for the Rényi-alpha scan."""

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
PPTX_OUT = OUT_DIR / "renyi_alpha_3bars_slide.pptx"
PDF_OUT = OUT_DIR / "renyi_alpha_3bars_slide.pdf"
PNG_OUT = OUT_DIR / "renyi_alpha_3bars_slide.png"


def add_slide(prs) -> SlideCanvas:
    c = SlideCanvas(prs)
    c.header("Lower Rényi α Improves Error Separation", "Analysis", "48", 30.0)
    c.text(
        "same model and data  ·  only α changes  ·  lower α gives low-probability tokens more influence",
        0.66,
        1.12,
        12.0,
        0.34,
        14.5,
        color=COLORS["muted"],
        italic=True,
    )

    c.rect(0.72, 1.55, 11.90, 4.84, COLORS["white"], COLORS["panel_line"], 0.8, 0.10)
    c.rect(0.98, 1.82, 4.12, 0.31, COLORS["panel_header"], radius=0.06)
    c.text(
        "Rényi entropy: α controls head-vs.-tail emphasis",
        1.00,
        1.80,
        4.08,
        0.34,
        11.0,
        color=COLORS["muted"],
        family="sans",
        align="center",
    )
    c.text(
        "2Wiki  ·  AUROC for predicting error ↑",
        8.60,
        1.84,
        3.46,
        0.30,
        12.2,
        color=COLORS["muted"],
        family="sans",
        align="right",
    )

    baseline_y = 5.20
    chart_top_y = 2.38
    chart_height = baseline_y - chart_top_y
    c.line(1.15, baseline_y, 11.03, COLORS["panel_line"], 0.014)
    c.text(
        "chance = 0.50",
        1.18,
        baseline_y - 0.26,
        1.45,
        0.22,
        10.0,
        color=COLORS["muted"],
        family="sans",
    )

    bars = [
        (2.06, 0.651, "BAC4CA", "α = 2", "p² suppresses small p", False),
        (5.30, 0.671, "7892A1", "α → 1", "Shannon limit", False),
        (8.54, 0.716, COLORS["green"], "α = 0.5", "√p magnifies small p", True),
    ]
    bar_w = 1.42
    for x, value, fill, alpha_label, meaning, highlight in bars:
        height = (value - 0.50) / 0.25 * chart_height
        y = baseline_y - height
        c.rect(x, y, bar_w, height, fill, radius=0.10)
        c.text(
            f"{value:.3f}",
            x - 0.10,
            y - 0.43,
            bar_w + 0.20,
            0.37,
            19.0,
            bold=True,
            family="sans",
            align="center",
        )
        c.text(
            alpha_label,
            x - 0.13,
            5.34,
            bar_w + 0.26,
            0.33,
            15.0,
            bold=True,
            family="sans",
            align="center",
        )
        c.text(
            meaning,
            x - 0.35,
            5.74,
            bar_w + 0.70,
            0.29,
            11.3,
            color=COLORS["green_dark"] if highlight else COLORS["muted"],
            bold=highlight,
            family="sans",
            align="center",
        )

    c.rect(0.92, 6.55, 11.50, 0.42, COLORS["highlight"], radius=0.07)
    c.text(
        "Takeaway: lower α makes rare-token evidence count more—and separates errors more effectively.",
        1.10,
        6.54,
        11.12,
        0.44,
        15.2,
        color=COLORS["green_dark"],
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
