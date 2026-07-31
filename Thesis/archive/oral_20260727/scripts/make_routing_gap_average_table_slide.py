#!/usr/bin/env python3
"""Build a thesis-style table for routing gap with a macro-average row."""

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
PPTX_OUT = OUT_DIR / "routing_gap_average_table_slide.pptx"
PDF_OUT = OUT_DIR / "routing_gap_average_table_slide.pdf"
PNG_OUT = OUT_DIR / "routing_gap_average_table_slide.png"

NAVY = "0B3972"
NAVY_DARK = "173B54"
PALE_BLUE = "EDF3F7"
PANEL_LINE = "B6C8D8"
MUTED = "52606D"
RULE = "353B40"
GREEN = "137455"

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
        "green_dark": GREEN,
    }
)


def add_slide(prs) -> SlideCanvas:
    c = SlideCanvas(prs)
    c.header("Direct-vs-Decompose Verifier Gap", "Analysis", "52", 28.0)
    c.text(
        "Thesis-formatted summary  ·  exact match at the accuracy-first operating point",
        0.66,
        1.10,
        11.95,
        0.30,
        13.4,
        color=MUTED,
        italic=True,
        family="sans",
    )

    c.rect(0.62, 1.49, 12.05, 4.82, "FFFFFF", PANEL_LINE, 0.9, 0.08)
    c.rect(0.80, 1.66, 2.00, 0.27, PALE_BLUE, radius=0.05)
    c.text(
        "Thesis Figure 5.5 · tabular view",
        0.84,
        1.65,
        1.92,
        0.29,
        8.6,
        color=NAVY_DARK,
        bold=True,
        family="sans",
        align="center",
    )
    c.text(
        "Final Verifier and Oracle Headroom",
        2.98,
        1.68,
        7.36,
        0.33,
        15.5,
        bold=True,
        align="center",
    )
    c.text(
        "Exact Match (EM); higher is better",
        3.78,
        2.00,
        5.76,
        0.25,
        10.4,
        color=MUTED,
        italic=True,
        align="center",
    )

    x0 = 0.94
    y0 = 2.37
    header_h = 0.72
    row_h = 0.53
    widths = [2.16, 2.24, 2.35, 2.16, 2.18]
    table_width = sum(widths)
    headers = [
        "Dataset",
        "Best pure\npath",
        "SEEK final\nverifier",
        "Oracle\nselector",
        "Oracle −\nVerifier",
    ]
    rows = [
        ("HotpotQA", "36.0", "37.8", "43.8", "+6.0"),
        ("2Wiki", "49.7", "49.3", "53.3", "+4.0"),
        ("MuSiQue", "26.7", "26.5", "29.4", "+2.9"),
        ("PopQA", "54.0", "53.9", "57.2", "+3.3"),
        ("Macro Average", "41.60", "41.88", "45.93", "+4.05"),
    ]

    c.line(x0, y0, table_width, RULE, 0.016)
    cursor_x = x0
    for column_index, (header, width) in enumerate(zip(headers, widths)):
        c.text(
            header,
            cursor_x + 0.04,
            y0 + 0.04,
            width - 0.08,
            header_h - 0.08,
            12.6,
            bold=True,
            align="left" if column_index == 0 else "center",
        )
        cursor_x += width
    c.line(x0, y0 + header_h, table_width, RULE, 0.010)

    for row_index, row in enumerate(rows):
        y = y0 + header_h + row_index * row_h
        is_average = row_index == len(rows) - 1
        if is_average:
            c.line(x0, y, table_width, RULE, 0.010)
        cursor_x = x0
        for column_index, (value, width) in enumerate(zip(row, widths)):
            color = NAVY_DARK if is_average else COLORS["ink"]
            if is_average and column_index == 4:
                color = GREEN
            c.text(
                value,
                cursor_x + 0.07,
                y + 0.03,
                width - 0.14,
                row_h - 0.06,
                12.7,
                color=color,
                bold=is_average,
                align="left" if column_index == 0 else "center",
            )
            cursor_x += width
    table_bottom = y0 + header_h + len(rows) * row_h
    c.line(x0, table_bottom, table_width, RULE, 0.016)

    c.text(
        "Best pure path = the better of always-direct and always-decompose within each dataset.",
        2.04,
        5.88,
        9.24,
        0.22,
        9.5,
        color=MUTED,
        italic=True,
        family="sans",
        align="center",
    )

    c.rect(0.82, 6.43, 11.72, 0.44, PALE_BLUE, radius=0.06)
    c.ellipse(1.02, 6.53, 0.23, 0.23, GREEN)
    c.text(
        "Macro average: SEEK final verifier is +0.28 EM above the best pure path; oracle headroom remains +4.05 EM.",
        1.38,
        6.42,
        10.86,
        0.45,
        12.4,
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
