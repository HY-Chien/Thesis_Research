#!/usr/bin/env python3
"""Build an editable slide with identical top-two probabilities and different tail mass."""

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
PPTX_OUT = OUT_DIR / "same_leaders_different_tail_slide.pptx"
PDF_OUT = OUT_DIR / "same_leaders_different_tail_slide.pdf"
PNG_OUT = OUT_DIR / "same_leaders_different_tail_slide.png"

NAVY = "1D4967"
NAVY_DARK = "173B54"
NAVY_LIGHT = "DCE7ED"
TAB_FILL = "EEF1F3"
PO_COLOR = "214F6B"
PO_LOWER_COLOR = "6B93AA"
TOP_REST = "D88922"
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
    title_size: float = 28.5,
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


def fixed_leader_chip(
    c: SlideCanvas,
    x: float,
    token: str,
    probability: float,
    fill: str,
) -> None:
    c.rect(x, 1.53, 1.20, 0.50, fill, radius=0.06)
    c.text(
        f"{token}  {probability:.2f}",
        x + 0.04,
        1.50,
        1.12,
        0.56,
        12.0,
        color="FFFFFF",
        bold=True,
        family="sans",
        align="center",
    )


def draw_mass_bar(
    c: SlideCanvas,
    x: float,
    y: float,
    width: float,
    third_to_fifth_mass: float,
    tail_mass: float,
) -> None:
    parts = [
        (0.38, PO_COLOR),
        (0.26, PO_LOWER_COLOR),
        (third_to_fifth_mass, TOP_REST),
        (tail_mass, TAIL),
    ]
    cursor_x = x
    for mass, fill in parts:
        part_width = width * mass
        c.rect(cursor_x, y, part_width, 0.48, fill, radius=0.04)
        cursor_x += part_width

    # Labels are placed separately so that the 3% tail remains legible.
    c.text("PO  38%", x + 0.08, y + 0.04, width * 0.38 - 0.16, 0.39, 11.0,
           color="FFFFFF", bold=True, family="sans", align="center")
    c.text("po  26%", x + width * 0.38 + 0.05, y + 0.04, width * 0.26 - 0.10, 0.39, 11.0,
           color="FFFFFF", bold=True, family="sans", align="center")
    if third_to_fifth_mass >= 0.20:
        c.text(
            f"ranks 3–5  {third_to_fifth_mass:.0%}",
            x + width * 0.64 + 0.04,
            y + 0.04,
            width * third_to_fifth_mass - 0.08,
            0.39,
            9.7,
            color="FFFFFF",
            bold=True,
            family="sans",
            align="center",
        )
    else:
        c.text(
            f"3–5\n{third_to_fifth_mass:.0%}",
            x + width * 0.64 + 0.02,
            y + 0.01,
            width * third_to_fifth_mass - 0.04,
            0.45,
            8.5,
            color="FFFFFF",
            bold=True,
            family="sans",
            align="center",
        )
    if tail_mass >= 0.10:
        c.text(
            f"tail  {tail_mass:.0%}",
            x + width * (1 - tail_mass) + 0.04,
            y + 0.04,
            width * tail_mass - 0.08,
            0.39,
            10.4,
            color="FFFFFF",
            bold=True,
            family="sans",
            align="center",
        )


def draw_case(
    c: SlideCanvas,
    x: float,
    title: str,
    ranks_3_to_5: list[float],
    tail_probabilities: list[float],
    header_fill: str,
    header_color: str,
    result: str,
) -> None:
    width = 5.72
    third_to_fifth_mass = sum(ranks_3_to_5)
    tail_mass = sum(tail_probabilities)
    retained_mass = 0.64 + third_to_fifth_mass
    full_distribution = [0.38, 0.26, *ranks_3_to_5, *tail_probabilities]

    assert len(ranks_3_to_5) == 3
    assert abs(sum(full_distribution) - 1.0) < 1e-8
    assert min(ranks_3_to_5) >= max(tail_probabilities)

    c.rect(x, 2.24, width, 3.72, "FFFFFF", COLORS["panel_line"], 0.8, 0.07)
    c.rect(x + 0.18, 2.42, width - 0.36, 0.40, header_fill, radius=0.05)
    c.text(
        title,
        x + 0.25,
        2.39,
        width - 0.50,
        0.46,
        14.0,
        color=header_color,
        bold=True,
        family="sans",
        align="center",
    )

    c.text(
        "The remaining 36% is allocated to:",
        x + 0.40,
        2.94,
        width - 0.80,
        0.27,
        10.6,
        color=COLORS["muted"],
        family="sans",
        align="center",
    )

    # Ranks 3–5 remain individually visible so the probability transfer is explicit.
    c.rect(x + 0.70, 3.32, 2.02, 0.84, "F8F4EC", COLORS["panel_line"], 0.7, 0.06)
    c.text(
        "Ranks 3–5",
        x + 0.78,
        3.37,
        1.86,
        0.24,
        10.0,
        color=TOP_REST,
        bold=True,
        family="sans",
        align="center",
    )
    c.text(
        "  +  ".join(f"{p:.2f}" for p in ranks_3_to_5),
        x + 0.78,
        3.64,
        1.86,
        0.31,
        12.3,
        color=TOP_REST,
        bold=True,
        family="sans",
        align="center",
    )

    c.text("→", x + 2.78, 3.48, 0.38, 0.40, 18.0, color=NAVY, bold=True, align="center")

    c.rect(x + 3.16, 3.32, 1.88, 0.84, "EEF4FA", COLORS["panel_line"], 0.7, 0.06)
    c.text(
        "Tail mass",
        x + 3.24,
        3.37,
        1.72,
        0.24,
        10.0,
        color=TAIL,
        bold=True,
        family="sans",
        align="center",
    )
    c.text(
        f"{tail_mass:.0%}",
        x + 3.24,
        3.61,
        1.72,
        0.34,
        17.0,
        color=TAIL,
        bold=True,
        family="sans",
        align="center",
    )

    c.text(
        "Full probability mass",
        x + 0.50,
        4.34,
        4.72,
        0.24,
        9.6,
        color=COLORS["muted"],
        family="sans",
        align="center",
    )
    draw_mass_bar(c, x + 0.50, 4.63, 4.72, third_to_fifth_mass, tail_mass)

    c.text(
        f"top-5 retained mass  {retained_mass:.0%}",
        x + 0.50,
        5.17,
        2.80,
        0.26,
        10.5,
        color=TOP_REST,
        bold=True,
        family="sans",
    )
    c.text(
        f"tail leakage  {tail_mass:.0%}",
        x + 3.02,
        5.17,
        2.20,
        0.26,
        10.5,
        color=TAIL,
        bold=True,
        family="sans",
        align="right",
    )
    c.rect(x + 1.10, 5.51, 3.52, 0.28, header_fill, radius=0.05)
    c.text(
        result,
        x + 1.13,
        5.47,
        3.46,
        0.36,
        10.7,
        color=header_color,
        bold=True,
        family="sans",
        align="center",
    )


def add_slide(prs) -> SlideCanvas:
    c = SlideCanvas(prs)
    deck_header(c, "Same Leading Candidates, Different Tail Leakage", "Methodology", "1")
    c.text(
        "The two leading probabilities are identical; only the remaining mass moves from ranks 3–5 into the tail.",
        0.65,
        1.08,
        12.00,
        0.30,
        14.0,
        color=COLORS["muted"],
        italic=True,
    )

    c.rect(3.43, 1.42, 6.46, 0.68, NAVY_LIGHT, radius=0.06)
    c.text(
        "Identical in both examples",
        3.63,
        1.48,
        2.14,
        0.48,
        12.2,
        color=NAVY_DARK,
        bold=True,
        family="sans",
        align="center",
    )
    fixed_leader_chip(c, 5.93, "PO", 0.38, PO_COLOR)
    fixed_leader_chip(c, 7.25, "po", 0.26, PO_LOWER_COLOR)
    c.text(
        "top-2 mass = 64%",
        8.55,
        1.49,
        1.15,
        0.46,
        10.7,
        color=NAVY_DARK,
        bold=True,
        family="sans",
        align="center",
    )

    draw_case(
        c,
        0.72,
        "A · Remaining mass stays near the head",
        [0.15, 0.10, 0.08],
        [0.005] * 6,
        GREEN_LIGHT,
        GREEN,
        "97% remains in the top-5",
    )
    draw_case(
        c,
        6.78,
        "B · Remaining mass escapes to the tail",
        [0.05, 0.05, 0.04],
        [0.22 / 6] * 6,
        RED_LIGHT,
        RED,
        "Tail leakage rises from 3% to 22%",
    )

    c.rect(0.82, 6.17, 11.68, 0.70, NAVY_LIGHT, radius=0.06)
    c.text(
        "PO and po look equally strong in both cases; 19 percentage points moving into the tail reveals the confidence drop.",
        1.08,
        6.20,
        11.16,
        0.40,
        14.0,
        color=NAVY_DARK,
        bold=True,
        align="center",
    )
    c.text(
        "Illustrative distributions; each row sums to 1.00.",
        4.23,
        6.57,
        4.86,
        0.22,
        9.2,
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
