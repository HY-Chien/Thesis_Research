#!/usr/bin/env python3
"""Create the revised SEEK research-gap-to-solution slide."""

from pathlib import Path

import make_seek_rq_solution_slide as b


OUT_DIR = Path(__file__).resolve().parent
PPTX_PATH = OUT_DIR / "SEEK_Motivation_Research_Question_Solution_v2.pptx"
PNG_PATH = OUT_DIR / "SEEK_Motivation_Research_Question_Solution_v2_preview.png"


def add_method_card(slide, x, title, subtitle, icon):
    b.add_rounded_rect(
        slide, x, 2.16, 2.30, 1.02, fill=b.PALE_BLUE, line=b.LIGHT_BORDER
    )
    b.add_circle_badge(
        slide,
        icon,
        x + 0.15,
        2.43,
        0.46,
        fill=b.WHITE,
        line=b.BLUE,
        text_color=b.BLUE,
        size=14,
    )
    b.add_text(
        slide,
        title,
        x + 0.73,
        2.23,
        1.43,
        0.45,
        size=11.5,
        color=b.NAVY,
        bold=True,
        align=b.PP_ALIGN.CENTER,
        margin=0,
    )
    b.add_text(
        slide,
        subtitle,
        x + 0.72,
        2.70,
        1.45,
        0.28,
        size=10.3,
        color=b.MUTED,
        align=b.PP_ALIGN.CENTER,
        margin=0,
    )


def add_solution_card(slide, number, y, title, subtitle):
    x, w, h = 7.48, 4.80, 0.94
    b.add_rounded_rect(
        slide, x, y, w, h, fill=b.PALE_BLUE, line=b.LIGHT_BORDER
    )
    b.add_circle_badge(
        slide,
        str(number),
        x + 0.18,
        y + 0.23,
        0.48,
        fill=b.WHITE,
        line=b.BLUE,
        text_color=b.BLUE,
        size=14,
    )
    b.add_text(
        slide,
        title,
        x + 0.82,
        y + 0.10,
        3.62,
        0.34,
        size=15.5,
        color=b.NAVY,
        bold=True,
        margin=0,
    )
    b.add_text(
        slide,
        subtitle,
        x + 0.82,
        y + 0.48,
        3.62,
        0.27,
        size=10.8,
        color=b.MUTED,
        margin=0,
    )


def add_header(slide):
    b.add_text(
        slide,
        "Motivation",
        0.40,
        0.16,
        2.15,
        0.52,
        size=28,
        color=b.TEAL,
        font=b.SERIF,
        bold=True,
        margin=0,
    )
    top_rule = b.add_connector(slide, 1.02, 0.82, 12.30, 0.82, width=1.0)
    top_rule.line.color.rgb = b.rgb(b.LINE)
    tab = slide.shapes.add_shape(
        b.MSO_SHAPE.RECTANGLE,
        b.Inches(10.86),
        b.Inches(0.00),
        b.Inches(2.47),
        b.Inches(0.62),
    )
    b.set_shape_fill(tab, b.TAB_GRAY)
    b.set_shape_line(tab, None)
    b.add_text(
        slide,
        "Introduction",
        10.96,
        0.10,
        2.24,
        0.40,
        size=18,
        color="000000",
        font=b.SERIF,
        bold=True,
        align=b.PP_ALIGN.CENTER,
        margin=0,
    )


def add_footer(slide):
    bottom_rule = b.add_connector(slide, 1.03, 7.02, 12.30, 7.02, width=1.0)
    bottom_rule.line.color.rgb = b.rgb(b.LINE)
    corner = slide.shapes.add_shape(
        b.MSO_SHAPE.PIE_WEDGE,
        b.Inches(11.99),
        b.Inches(6.68),
        b.Inches(0.58),
        b.Inches(0.58),
    )
    b.set_shape_fill(corner, b.CORNER)
    b.set_shape_line(corner, None)
    corner.rotation = 180


def build_pptx():
    prs = b.Presentation()
    prs.slide_width = b.Inches(b.SLIDE_W)
    prs.slide_height = b.Inches(b.SLIDE_H)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.background.fill
    background.solid()
    background.fore_color.rgb = b.rgb(b.WHITE)

    add_header(slide)

    left_x, panel_y, panel_w, panel_h = 0.68, 1.20, 5.55, 5.15
    right_x = 7.10
    b.add_panel(slide, left_x, panel_y, panel_w, panel_h)
    b.add_panel(slide, right_x, panel_y, panel_w, panel_h)

    arrow = slide.shapes.add_shape(
        b.MSO_SHAPE.RIGHT_ARROW,
        b.Inches(6.38),
        b.Inches(3.45),
        b.Inches(0.45),
        b.Inches(0.43),
    )
    b.set_shape_fill(arrow, b.BLUE)
    b.set_shape_line(arrow, None)

    b.add_circle_badge(slide, "1", 0.98, 1.47, 0.53, size=15)
    b.add_text(
        slide,
        "RESEARCH QUESTION",
        1.67,
        1.43,
        3.95,
        0.55,
        size=18,
        color=b.NAVY,
        font=b.SERIF,
        bold=True,
        margin=0,
    )
    b.add_circle_badge(slide, "2", 7.40, 1.47, 0.53, size=15)
    b.add_text(
        slide,
        "SOLUTION — SEEK",
        8.09,
        1.43,
        3.95,
        0.55,
        size=18,
        color=b.NAVY,
        font=b.SERIF,
        bold=True,
        margin=0,
    )

    # Reconnect to the two adaptive-RAG approaches introduced immediately before.
    add_method_card(
        slide,
        1.02,
        "TOKEN-LEVEL\nCONFIDENCE",
        "Local uncertainty",
        "T",
    )
    add_method_card(
        slide,
        3.54,
        "QUESTION-LEVEL\nCLASSIFICATION",
        "One route per question",
        "Q",
    )
    b.add_connector(slide, 2.17, 3.18, 2.17, 3.40)
    b.add_connector(slide, 4.69, 3.18, 4.69, 3.40)
    b.add_connector(slide, 2.17, 3.40, 4.69, 3.40)
    b.add_connector(slide, 3.43, 3.40, 3.43, 3.58)

    b.add_rounded_rect(
        slide,
        1.61,
        3.58,
        3.64,
        0.57,
        fill=b.NAVY,
        line=None,
    )
    b.add_text(
        slide,
        "SUB-QUESTION-LEVEL DECISION",
        1.61,
        3.58,
        3.64,
        0.57,
        size=13.5,
        color=b.WHITE,
        bold=True,
        align=b.PP_ALIGN.CENTER,
        margin=0,
    )

    b.add_rounded_rect(
        slide,
        1.04,
        4.43,
        4.82,
        1.27,
        fill=b.PALE_BLUE,
        line=b.BLUE,
        line_width=1.3,
    )
    b.add_text(
        slide,
        "Can adaptive RAG make interpretable\n"
        "retrieval decisions at each sub-question\n"
        "without learning a separate retrieval policy?",
        1.27,
        4.52,
        4.36,
        1.04,
        size=14.8,
        color=b.TEXT,
        font=b.SERIF,
        bold=True,
        align=b.PP_ALIGN.CENTER,
        margin=0,
    )
    b.add_rounded_rect(
        slide,
        1.73,
        5.86,
        3.40,
        0.36,
        fill=b.PALE_GRAY,
        line=None,
    )
    b.add_text(
        slide,
        "Decomposition remains reversible.",
        1.73,
        5.86,
        3.40,
        0.36,
        size=11.2,
        color=b.MUTED,
        italic=True,
        align=b.PP_ALIGN.CENTER,
        margin=0,
    )

    # SEEK maps the token signal to reasoning-step decisions and keeps routing inspectable.
    solution_rows = [
        (
            1,
            2.16,
            "Sub-question self-certainty",
            "Token probabilities → step-level signal",
        ),
        (
            2,
            3.35,
            "Evidence-gated retrieval",
            "Retrieve only when a sub-question is uncertain",
        ),
        (
            3,
            4.54,
            "Dual-path verification",
            "Preserve original and decomposed candidates",
        ),
    ]
    for number, y, title, subtitle in solution_rows:
        add_solution_card(slide, number, y, title, subtitle)
    b.add_connector(slide, 9.88, 3.10, 9.88, 3.35)
    b.add_connector(slide, 9.88, 4.29, 9.88, 4.54)

    b.add_rounded_rect(
        slide,
        8.12,
        5.78,
        3.55,
        0.42,
        fill=b.PALE_GRAY,
        line=None,
    )
    b.add_text(
        slide,
        "No separate retrieval-policy training",
        8.12,
        5.78,
        3.55,
        0.42,
        size=11.5,
        color=b.TEAL,
        bold=True,
        align=b.PP_ALIGN.CENTER,
        margin=0,
    )

    add_footer(slide)
    prs.core_properties.title = "SEEK Motivation — Research Question and Solution v2"
    prs.core_properties.subject = "Thesis defense introduction slide"
    prs.core_properties.author = "SEEK"
    prs.save(PPTX_PATH)


def render_preview():
    image = b.Image.new("RGB", (b.PX_W, b.PX_H), b.hex_color(b.WHITE))
    draw = b.ImageDraw.Draw(image)

    b.text_left(
        draw,
        "Motivation",
        0.40,
        0.15,
        size=28,
        color=b.TEAL,
        serif=True,
        bold=True,
    )
    draw.line(
        (b.px(1.02), b.px(0.82), b.px(12.30), b.px(0.82)),
        fill=b.hex_color(b.LINE),
        width=2,
    )
    draw.rectangle(
        (b.px(10.86), 0, b.px(13.33), b.px(0.62)),
        fill=b.hex_color(b.TAB_GRAY),
    )
    b.text_center(
        draw,
        "Introduction",
        10.96,
        0.10,
        2.24,
        0.40,
        size=18,
        color="000000",
        serif=True,
        bold=True,
    )

    for x in (0.68, 7.10):
        b.rr(draw, x + 0.045, 1.255, 5.55, 5.15, b.SHADOW)
        b.rr(draw, x, 1.20, 5.55, 5.15, b.WHITE, b.LIGHT_BORDER, radius=0.14)

    ax, ay, aw, ah = b.px(6.38), b.px(3.45), b.px(0.45), b.px(0.43)
    draw.polygon(
        [
            (ax, ay + ah * 0.25),
            (ax + aw * 0.58, ay + ah * 0.25),
            (ax + aw * 0.58, ay),
            (ax + aw, ay + ah / 2),
            (ax + aw * 0.58, ay + ah),
            (ax + aw * 0.58, ay + ah * 0.75),
            (ax, ay + ah * 0.75),
        ],
        fill=b.hex_color(b.BLUE),
    )

    b.circle(draw, "1", 0.98, 1.47, 0.53, fill=b.NAVY)
    b.text_left(
        draw,
        "RESEARCH QUESTION",
        1.67,
        1.52,
        size=18,
        color=b.NAVY,
        serif=True,
        bold=True,
    )
    b.circle(draw, "2", 7.40, 1.47, 0.53, fill=b.NAVY)
    b.text_left(
        draw,
        "SOLUTION — SEEK",
        8.09,
        1.52,
        size=18,
        color=b.NAVY,
        serif=True,
        bold=True,
    )

    method_rows = [
        (1.02, "TOKEN-LEVEL\nCONFIDENCE", "Local uncertainty", "T"),
        (3.54, "QUESTION-LEVEL\nCLASSIFICATION", "One route per question", "Q"),
    ]
    for x, title, subtitle, icon in method_rows:
        b.rr(draw, x, 2.16, 2.30, 1.02, b.PALE_BLUE, b.LIGHT_BORDER)
        b.circle(
            draw,
            icon,
            x + 0.15,
            2.43,
            0.46,
            fill=b.WHITE,
            outline=b.BLUE,
            text_color=b.BLUE,
            size=14,
        )
        b.text_center(
            draw,
            title,
            x + 0.73,
            2.23,
            1.43,
            0.45,
            size=11.5,
            color=b.NAVY,
            bold=True,
        )
        b.text_center(
            draw,
            subtitle,
            x + 0.72,
            2.70,
            1.45,
            0.28,
            size=10.3,
            color=b.MUTED,
        )
    draw.line(
        (b.px(2.17), b.px(3.18), b.px(2.17), b.px(3.40)),
        fill=b.hex_color(b.LINE),
        width=2,
    )
    draw.line(
        (b.px(4.69), b.px(3.18), b.px(4.69), b.px(3.40)),
        fill=b.hex_color(b.LINE),
        width=2,
    )
    draw.line(
        (b.px(2.17), b.px(3.40), b.px(4.69), b.px(3.40)),
        fill=b.hex_color(b.LINE),
        width=2,
    )
    draw.line(
        (b.px(3.43), b.px(3.40), b.px(3.43), b.px(3.58)),
        fill=b.hex_color(b.LINE),
        width=2,
    )
    b.rr(draw, 1.61, 3.58, 3.64, 0.57, b.NAVY, radius=0.11)
    b.text_center(
        draw,
        "SUB-QUESTION-LEVEL DECISION",
        1.61,
        3.58,
        3.64,
        0.57,
        size=13.5,
        color=b.WHITE,
        bold=True,
    )

    b.rr(draw, 1.04, 4.43, 4.82, 1.27, b.PALE_BLUE, b.BLUE, width=1.3)
    b.text_center(
        draw,
        "Can adaptive RAG make interpretable\n"
        "retrieval decisions at each sub-question\n"
        "without learning a separate retrieval policy?",
        1.27,
        4.52,
        4.36,
        1.04,
        size=14.8,
        serif=True,
        bold=True,
    )
    b.rr(draw, 1.73, 5.86, 3.40, 0.36, b.PALE_GRAY)
    b.text_center(
        draw,
        "Decomposition remains reversible.",
        1.73,
        5.86,
        3.40,
        0.36,
        size=11.2,
        color=b.MUTED,
    )

    solution_rows = [
        (1, 2.16, "Sub-question self-certainty", "Token probabilities → step-level signal"),
        (2, 3.35, "Evidence-gated retrieval", "Retrieve only when a sub-question is uncertain"),
        (3, 4.54, "Dual-path verification", "Preserve original and decomposed candidates"),
    ]
    for number, y, title, subtitle in solution_rows:
        b.rr(draw, 7.48, y, 4.80, 0.94, b.PALE_BLUE, b.LIGHT_BORDER)
        b.circle(
            draw,
            str(number),
            7.66,
            y + 0.23,
            0.48,
            fill=b.WHITE,
            outline=b.BLUE,
            text_color=b.BLUE,
            size=14,
        )
        b.text_left(
            draw,
            title,
            8.30,
            y + 0.17,
            size=15.5,
            color=b.NAVY,
            bold=True,
        )
        b.text_left(
            draw,
            subtitle,
            8.30,
            y + 0.54,
            size=10.8,
            color=b.MUTED,
        )
    draw.line(
        (b.px(9.88), b.px(3.10), b.px(9.88), b.px(3.35)),
        fill=b.hex_color(b.LINE),
        width=2,
    )
    draw.line(
        (b.px(9.88), b.px(4.29), b.px(9.88), b.px(4.54)),
        fill=b.hex_color(b.LINE),
        width=2,
    )

    b.rr(draw, 8.12, 5.78, 3.55, 0.42, b.PALE_GRAY)
    b.text_center(
        draw,
        "No separate retrieval-policy training",
        8.12,
        5.78,
        3.55,
        0.42,
        size=11.5,
        color=b.TEAL,
        bold=True,
    )
    draw.line(
        (b.px(1.03), b.px(7.02), b.px(12.30), b.px(7.02)),
        fill=b.hex_color(b.LINE),
        width=2,
    )
    draw.pieslice(
        (b.px(11.99), b.px(6.68), b.px(12.57), b.px(7.26)),
        start=180,
        end=270,
        fill=b.hex_color(b.CORNER),
    )
    image.save(PNG_PATH, quality=95)


if __name__ == "__main__":
    build_pptx()
    render_preview()
    print(PPTX_PATH)
    print(PNG_PATH)
