#!/usr/bin/env python3
"""Create slides 50--52 from the thesis-formatted analysis tables and figure."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PACKAGE_DIR = Path("/home/S113065528/.codex_tmp/python_pptx")
sys.path.insert(0, str(PACKAGE_DIR))

from PIL import Image
from pptx.util import Inches

from make_tl_sum_2slides import COLORS, SlideCanvas, new_presentation, save_pdf


OUT_DIR = Path(__file__).resolve().parent
ASSET_DIR = OUT_DIR / "replacement_slides_assets"
THESIS_PDF = OUT_DIR / "main.pdf"
PPTX_OUT = OUT_DIR / "analysis_tables_tau002305_replacement.pptx"
PDF_OUT = OUT_DIR / "analysis_tables_tau002305_replacement.pdf"
PNG_OUT = [
    OUT_DIR / "analysis_tables_tau002305_slide50.png",
    OUT_DIR / "analysis_tables_tau002305_slide51.png",
    OUT_DIR / "analysis_tables_tau002305_slide52.png",
]

PAGE_WIDTH_PT = 595.276
PAGE_HEIGHT_PT = 841.890
RENDER_DPI = 300

NAVY = "0B3972"
NAVY_DARK = "173B54"
PALE_BLUE = "EDF3F7"
PANEL_LINE = "B6C8D8"
MUTED = "52606D"
GREEN = "137455"


COLORS.update(
    {
        "background": "FFFFFF",
        "white": "FFFFFF",
        "title": NAVY,
        "ink": "111827",
        "muted": MUTED,
        "line": "7B9AAE",
        "panel_line": PANEL_LINE,
        "panel_header": "EEF1F3",
        "ice": PALE_BLUE,
        "ice_deep": "DCE7ED",
        "green_dark": GREEN,
    }
)


def render_thesis_pages() -> dict[int, Path]:
    """Render only the physical PDF pages that contain the requested excerpts."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    prefix = ASSET_DIR / "thesis_page_300dpi"
    subprocess.run(
        [
            "pdftocairo",
            "-png",
            "-r",
            str(RENDER_DPI),
            "-f",
            "42",
            "-l",
            "42",
            str(THESIS_PDF),
            str(prefix),
        ],
        check=True,
    )
    subprocess.run(
        [
            "pdftocairo",
            "-png",
            "-r",
            str(RENDER_DPI),
            "-f",
            "45",
            "-l",
            "46",
            str(THESIS_PDF),
            str(prefix),
        ],
        check=True,
    )
    return {page: ASSET_DIR / f"thesis_page_300dpi-{page}.png" for page in (42, 45, 46)}


def crop_pdf_points(
    source: Path,
    target: Path,
    box_pt: tuple[float, float, float, float],
) -> Path:
    """Crop a rendered PDF page using coordinates in the original PDF point space."""
    with Image.open(source) as image:
        sx = image.width / PAGE_WIDTH_PT
        sy = image.height / PAGE_HEIGHT_PT
        x0, y0, x1, y1 = box_pt
        crop_box = (
            round(x0 * sx),
            round(y0 * sy),
            round(x1 * sx),
            round(y1 * sy),
        )
        crop = image.crop(crop_box).convert("RGB")
        crop.save(target, "PNG", optimize=True)
    return target


def make_excerpts(pages: dict[int, Path]) -> dict[str, Path]:
    """Extract complete captions plus the associated table or figure."""
    return {
        "overlap": crop_pdf_points(
            pages[42],
            ASSET_DIR / "thesis_table_5_8_tau002305.png",
            (90, 418, 510, 626),
        ),
        "selection": crop_pdf_points(
            pages[42],
            ASSET_DIR / "thesis_table_5_7_tau002305.png",
            (90, 74, 510, 298),
        ),
        # Use the exact high-resolution source embedded as Thesis Figure 5.5.
        # A page-level crop makes its axis labels too small for projection.
        "oracle": OUT_DIR / "figures" / "fig_routing_gap_oracle.png",
    }


def add_picture_fit(
    canvas: SlideCanvas,
    path: Path,
    x: float,
    y: float,
    w: float,
    h: float,
) -> None:
    """Place the same aspect-preserving image in the editable PPTX and PNG preview."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        ratio = image.width / image.height
        target_ratio = w / h
        if ratio >= target_ratio:
            placed_w = w
            placed_h = w / ratio
        else:
            placed_h = h
            placed_w = h * ratio
        placed_x = x + (w - placed_w) / 2
        placed_y = y + (h - placed_h) / 2
        canvas.slide.shapes.add_picture(
            str(path),
            Inches(placed_x),
            Inches(placed_y),
            width=Inches(placed_w),
            height=Inches(placed_h),
        )
        preview = image.resize(
            (canvas.px(placed_w), canvas.px(placed_h)),
            Image.Resampling.LANCZOS,
        )
        canvas.image.paste(preview, (canvas.px(placed_x), canvas.px(placed_y)))


def add_excerpt_slide(
    prs,
    title: str,
    slide_no: str,
    excerpt: Path,
    source_label: str,
    takeaway: str,
) -> SlideCanvas:
    canvas = SlideCanvas(prs)
    canvas.header(title, "Analysis", slide_no, 27.5)
    canvas.text(
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

    canvas.rect(0.62, 1.49, 12.05, 4.78, "FFFFFF", PANEL_LINE, 0.9, 0.08)
    canvas.rect(0.80, 1.66, 1.52, 0.27, PALE_BLUE, radius=0.05)
    canvas.text(
        source_label,
        0.84,
        1.65,
        1.44,
        0.29,
        9.2,
        color=NAVY_DARK,
        bold=True,
        family="sans",
        align="center",
    )
    add_picture_fit(canvas, excerpt, 0.87, 1.84, 11.55, 4.15)

    canvas.rect(0.82, 6.42, 11.72, 0.45, PALE_BLUE, radius=0.06)
    canvas.ellipse(1.02, 6.52, 0.23, 0.23, GREEN)
    canvas.text(
        takeaway,
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
    return canvas


def validate_deck(prs, expected_slides: int) -> None:
    assert len(prs.slides) == expected_slides
    for index, slide in enumerate(prs.slides, start=1):
        assert len(slide.shapes) >= 15, f"slide {index}: unexpectedly few shapes"
        for shape in slide.shapes:
            assert shape.left + shape.width > 0
            assert shape.top + shape.height > 0


def main() -> None:
    pages = render_thesis_pages()
    excerpts = make_excerpts(pages)

    prs = new_presentation()
    canvases = [
        add_excerpt_slide(
            prs,
            "Correct Answer Distribution Across Two Paths",
            "50",
            excerpts["overlap"],
            "Thesis Table 5.8",
            "Decomposition-only wins exceed direct-only wins on every dataset.",
        ),
        add_excerpt_slide(
            prs,
            "Verifier Path Selection",
            "51",
            excerpts["selection"],
            "Thesis Table 5.7",
            "The verifier chooses decomposition 91.8% of the time while retaining a direct fallback.",
        ),
        add_excerpt_slide(
            prs,
            "Direct-vs-Decompose Oracle Gap",
            "52",
            excerpts["oracle"],
            "Thesis Figure 5.5",
            "A perfect path selector still offers +4.05 EM on average; HotpotQA has the largest gap (+6.0).",
        ),
    ]

    validate_deck(prs, 3)
    prs.save(PPTX_OUT)
    save_pdf(canvases, PDF_OUT)
    for canvas, png_path in zip(canvases, PNG_OUT):
        canvas.image.save(png_path, "PNG", optimize=True)

    for path in [PPTX_OUT, PDF_OUT, *PNG_OUT, *excerpts.values()]:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
