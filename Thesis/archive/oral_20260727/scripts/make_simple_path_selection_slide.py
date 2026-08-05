#!/usr/bin/env python3
"""Build a thesis-style simplified version of Table 5.7 for presentation."""

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
PPTX_OUT = OUT_DIR / "simple_path_selection_slide.pptx"
PDF_OUT = OUT_DIR / "simple_path_selection_slide.pdf"
PNG_OUT = OUT_DIR / "simple_path_selection_slide.png"

NAVY = "0B3972"
NAVY_DARK = "173B54"
PALE_BLUE = "EDF3F7"
PANEL_LINE = "B6C8D8"
MUTED = "52606D"
RULE = "353B40"

COLORS.update(
    {
        "background": "FFFFFF",
        "white": "FFFFFF",
        "title": NAVY,
        "ink": "111111",
        "muted": MUTED,
        "line": "7B9AAE",
        "panel_line": PANEL_LINE,
        "panel_header": "EEF1F3",
        "ice": PALE_BLUE,
        "ice_deep": "DCE7ED",
    }
)


def add_slide(prs) -> SlideCanvas:
    c = SlideCanvas(prs)
    c.header("Verifier Path Selection", "Analysis", "51", 29.0)
    c.text(
        "Thesis-formatted result  ·  accuracy-first tail-leakage gate  τ = 0.002305",
        0.66,
        1.10,
        11.95,
        0.30,
        13.4,
        color=MUTED,
        italic=True,
        family="sans",
    )

    c.rect(0.62, 1.49, 12.05, 4.92, "FFFFFF", PANEL_LINE, 0.9, 0.08)
    c.rect(0.80, 1.66, 1.96, 0.27, PALE_BLUE, radius=0.05)
    c.text(
        "Thesis Table 5.7 · simplified",
        0.84,
        1.65,
        1.88,
        0.29,
        8.8,
        color=NAVY_DARK,
        bold=True,
        family="sans",
        align="center",
    )
    c.text(
        "Final Verifier Choices by Dataset",
        2.75,
        1.68,
        7.83,
        0.33,
        15.5,
        bold=True,
        align="center",
    )
    c.text(
        "Selection counts and percentages across 1,000 questions per dataset",
        2.53,
        2.00,
        8.27,
        0.25,
        10.4,
        color=MUTED,
        italic=True,
        align="center",
    )

    x0 = 1.18
    y0 = 2.39
    header_h = 0.70
    row_h = 0.55
    widths = [3.05, 3.76, 4.18]
    table_width = sum(widths)
    headers = ["Dataset", "Direct chosen\nn (%)", "Decomposition path\nn (%)"]
    rows = [
        ("HotpotQA", "171 (17.1%)", "829 (82.9%)"),
        ("2Wiki", "25 (2.5%)", "975 (97.5%)"),
        ("MuSiQue", "121 (12.1%)", "879 (87.9%)"),
        ("PopQA", "10 (1.0%)", "990 (99.0%)"),
        ("Pooled", "327 (8.2%)", "3,673 (91.8%)"),
    ]

    # Native editable shapes styled like a LaTeX booktabs table.
    c.line(x0, y0, table_width, RULE, 0.016)
    cursor_x = x0
    for column_index, (header, width) in enumerate(zip(headers, widths)):
        c.text(
            header,
            cursor_x + 0.06,
            y0 + 0.04,
            width - 0.12,
            header_h - 0.08,
            13.1,
            bold=True,
            align="left" if column_index == 0 else "center",
        )
        cursor_x += width
    c.line(x0, y0 + header_h, table_width, RULE, 0.010)

    for row_index, row in enumerate(rows):
        y = y0 + header_h + row_index * row_h
        is_pooled = row_index == len(rows) - 1
        if is_pooled:
            c.line(x0, y, table_width, RULE, 0.010)
        cursor_x = x0
        for column_index, (value, width) in enumerate(zip(row, widths)):
            c.text(
                value,
                cursor_x + 0.08,
                y + 0.03,
                width - 0.16,
                row_h - 0.06,
                13.0,
                bold=False,
                align="left" if column_index == 0 else "center",
            )
            cursor_x += width
    table_bottom = y0 + header_h + len(rows) * row_h
    c.line(x0, table_bottom, table_width, RULE, 0.016)

    c.text(
        "Selection percentages describe routing behavior, not candidate correctness.",
        2.48,
        5.96,
        8.36,
        0.24,
        9.7,
        color=MUTED,
        italic=True,
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
