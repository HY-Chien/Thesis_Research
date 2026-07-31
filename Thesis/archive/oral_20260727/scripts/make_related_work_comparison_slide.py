#!/usr/bin/env python3
"""Build an editable, concise related-work comparison slide."""

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
PPTX_OUT = OUT_DIR / "related_work_comparison_slide.pptx"
PDF_OUT = OUT_DIR / "related_work_comparison_slide.pdf"
PNG_OUT = OUT_DIR / "related_work_comparison_slide.png"

NAVY = "1D4967"
NAVY_DARK = "173B54"
NAVY_LIGHT = "DCE7ED"
TAB_FILL = "EEF1F3"
RULE = "5F6A70"
SEEK_FILL = "E6F1EC"
GREEN = "3D8B3D"
AMBER = "C47A00"
RED = "D43D3D"


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


def add_slide(prs) -> SlideCanvas:
    c = SlideCanvas(prs)
    deck_header(c, "SEEK in the Adaptive RAG Design Space", "Related Work", "24")
    c.text(
        "A concise comparison of where retrieval is decided, what signal it uses, and how interpretable that decision is.",
        0.65,
        1.08,
        12.00,
        0.30,
        13.7,
        color=COLORS["muted"],
        italic=True,
    )

    x0 = 0.70
    y0 = 1.56
    header_h = 0.68
    row_h = 0.82
    widths = [2.42, 2.64, 3.62, 3.22]
    headers = [
        "Method",
        "Decision\nunit",
        "Retrieval\nsignal",
        "Decision\ntransparency",
    ]
    rows = [
        ("Adaptive-RAG", "Question", "Complexity classifier", "Medium"),
        ("FLARE", "Token / Sentence", "Token probability", "High"),
        ("DeepRAG", "Sub-question", "Learned policy", "Low"),
        ("SEEK (Ours)", "Sub-question", "Tail leakage + verifier", "High"),
    ]
    unit_colors = [RED, AMBER, GREEN, GREEN]
    transparency_colors = [AMBER, GREEN, RED, GREEN]
    table_width = sum(widths)

    # Minimal booktabs-style table: white header, generous spacing, horizontal rules.
    c.line(x0, y0, table_width, RULE, 0.014)
    cursor_x = x0
    for header, width in zip(headers, widths):
        c.text(
            header,
            cursor_x + 0.05,
            y0 + 0.05,
            width - 0.10,
            header_h - 0.10,
            13.2,
            bold=True,
            align="center",
        )
        cursor_x += width
    c.line(x0, y0 + header_h, table_width, RULE, 0.010)

    for row_index, row in enumerate(rows):
        y = y0 + header_h + row_index * row_h
        if row_index == len(rows) - 1:
            c.rect(x0, y + 0.02, table_width, row_h - 0.04, SEEK_FILL)

        cursor_x = x0
        for column_index, (value, width) in enumerate(zip(row, widths)):
            color = COLORS["ink"]
            bold = column_index == 0 or row_index == len(rows) - 1
            align = "left" if column_index == 0 else "center"
            margin_x = 0.18 if column_index == 0 else 0.06
            if column_index == 1:
                color = unit_colors[row_index]
            elif column_index == 2 and row_index == len(rows) - 1:
                color = GREEN
            elif column_index == 3:
                color = transparency_colors[row_index]
                bold = True
            c.text(
                value,
                cursor_x + margin_x,
                y + 0.05,
                width - 2 * margin_x,
                row_h - 0.10,
                13.5 if column_index else 14.0,
                color=color,
                bold=bold,
                align=align,
            )
            cursor_x += width
        c.line(x0, y + row_h, table_width, RULE, 0.008)

    c.rect(0.92, 5.69, 11.50, 0.72, NAVY_LIGHT, radius=0.06)
    c.text(
        "SEEK is the only method here that combines sub-question decisions with an explicit, interpretable confidence signal.",
        1.15,
        5.70,
        11.04,
        0.68,
        16.0,
        color=NAVY_DARK,
        bold=True,
        align="center",
    )
    c.text(
        "Granularity tells us where the decision happens; transparency tells us whether that decision can be inspected and explained.",
        1.00,
        6.53,
        11.32,
        0.30,
        10.3,
        color=COLORS["muted"],
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
