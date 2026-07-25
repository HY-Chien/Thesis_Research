#!/usr/bin/env python3
"""Build an editable motivation slide showing that tail emphasis improves AUROC."""

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


def add_slide(prs) -> SlideCanvas:
    c = SlideCanvas(prs)
    c.header(
        "Lower α Reveals an Error Signal in the Tail",
        "Methodology",
        "1",
        28.5,
    )
    c.text(
        "The same token distribution is viewed with different emphasis; tail emphasis gives better error detection.",
        0.66,
        1.12,
        11.8,
        0.34,
        14.5,
        color=COLORS["muted"],
        italic=True,
    )

    # Left: empirical result.
    c.rect(0.72, 1.55, 4.52, 4.84, COLORS["white"], COLORS["panel_line"], 0.8, 0.10)
    c.rect(0.98, 1.82, 4.00, 0.31, COLORS["panel_header"], radius=0.06)
    c.text(
        "2Wiki error-detection AUROC ↑",
        1.02,
        1.80,
        3.92,
        0.34,
        11.2,
        color=COLORS["muted"],
        family="sans",
        align="center",
    )

    baseline_y = 5.18
    chart_top_y = 2.48
    chart_height = baseline_y - chart_top_y
    c.line(1.02, baseline_y, 4.94, COLORS["panel_line"], 0.014)
    c.text(
        "chance = .50",
        0.78,
        baseline_y - 0.25,
        0.74,
        0.22,
        8.5,
        color=COLORS["muted"],
        family="sans",
    )

    bars = [
        (1.24, 0.651, "BAC4CA", "α = 2", "head emphasis", False),
        (2.62, 0.671, "7892A1", "α = 1", "original", False),
        (4.00, 0.716, COLORS["green"], "α = 0.5", "tail emphasis", True),
    ]
    bar_w = 0.76
    for x, value, fill, label, meaning, highlight in bars:
        height = (value - 0.50) / 0.25 * chart_height
        y = baseline_y - height
        c.rect(x, y, bar_w, height, fill, radius=0.09)
        c.text(
            f"{value:.3f}",
            x - 0.12,
            y - 0.40,
            bar_w + 0.24,
            0.34,
            16.5,
            bold=True,
            family="sans",
            align="center",
        )
        c.text(
            label,
            x - 0.20,
            5.30,
            bar_w + 0.40,
            0.32,
            13.2,
            bold=True,
            family="sans",
            align="center",
        )
        c.text(
            meaning,
            x - 0.30,
            5.69,
            bar_w + 0.60,
            0.29,
            9.6,
            color=COLORS["green_dark"] if highlight else COLORS["muted"],
            bold=highlight,
            family="sans",
            align="center",
        )

    # Right: the same distribution after applying different powers. All
    # elements remain native PowerPoint shapes/text.
    c.rect(5.45, 1.55, 7.17, 4.84, COLORS["white"], COLORS["panel_line"], 0.8, 0.10)
    c.text(
        "Same probabilities, different emphasis",
        5.78,
        1.78,
        6.50,
        0.30,
        13.2,
        bold=True,
        family="sans",
        align="center",
    )
    c.text(
        "Displayed weight:  w_α(v) = p(v)^α / Σ_u p(u)^α",
        5.78,
        2.12,
        6.50,
        0.30,
        12.0,
        color=COLORS["muted"],
        family="math",
        align="center",
    )

    probabilities = [0.32, 0.21, 0.14, 0.09, 0.07, 0.05, 0.035, 0.027, 0.021, 0.017, 0.012, 0.008]
    mini_panels = [
        (5.72, 2.0, "α = 2", "head dominates"),
        (7.91, 1.0, "α = 1", "original distribution"),
        (10.10, 0.5, "α = 0.5", "tail becomes visible"),
    ]
    mini_w = 1.86
    mini_base = 5.23
    max_height = 1.92
    common_y_max = 0.60
    bar_width = 0.092
    gap = 0.035
    for panel_x, alpha, alpha_label, interpretation in mini_panels:
        c.rect(panel_x, 2.57, mini_w, 3.33, "F7F8F8", COLORS["panel_line"], 0.55, 0.06)
        transformed = [value ** alpha for value in probabilities]
        total = sum(transformed)
        weights = [value / total for value in transformed]
        c.text(alpha_label, panel_x + 0.08, 2.70, mini_w - 0.16, 0.28, 13.5, bold=True, family="sans", align="center")
        c.text(interpretation, panel_x + 0.06, 2.99, mini_w - 0.12, 0.24, 9.3,
               color=COLORS["green_dark"] if alpha < 1 else COLORS["muted"],
               bold=alpha < 1, family="sans", align="center")
        c.line(panel_x + 0.14, mini_base, panel_x + mini_w - 0.14, COLORS["panel_line"], 0.009)
        bar_x = panel_x + 0.16
        for index, weight in enumerate(weights):
            if index == 5:
                bar_x += 0.07
            height = weight / common_y_max * max_height
            fill = "E49325" if index < 5 else "4C86C6"
            c.rect(bar_x, mini_base - height, bar_width, height, fill)
            bar_x += bar_width + gap
        c.text("top-5", panel_x + 0.12, 5.37, 0.70, 0.22, 8.7, color="C97812", bold=True, family="sans", align="center")
        c.text("tail", panel_x + 1.02, 5.37, 0.62, 0.22, 8.7, color="3D75B3", bold=True, family="sans", align="center")
    c.text("Tokens ranked by original probability  →", 6.64, 5.72, 4.80, 0.24, 9.8,
           color=COLORS["muted"], family="sans", align="center")

    c.rect(0.92, 6.55, 11.50, 0.42, COLORS["highlight"], radius=0.07)
    c.text(
        "α < 1 makes low-probability tokens matter more; its higher AUROC points us toward the tail.",
        1.08,
        6.54,
        11.18,
        0.44,
        14.7,
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
