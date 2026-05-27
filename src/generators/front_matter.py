"""Front-matter PDF helpers shared across book pipelines.

The title page and the copyright/tips page are the only interior pages whose
layout is identical for every book type. Both the coloring pipeline and the
puzzle pipeline import the helpers below; each owns its own ``tips`` text
(coloring: marker bleed warnings; puzzle: solver instructions).
"""

from __future__ import annotations

from pathlib import Path

import reportlab
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from src.utils.kdp_specs import InteriorLayout

FONT_REGULAR = "Vera"
FONT_BOLD = "VeraBd"
_FONT_DIR = Path(reportlab.__file__).resolve().parent / "fonts"

TITLE_SIZE = 32
AUTHOR_SIZE = 16
BODY_SIZE = 11

_fonts_registered = False


def register_fonts() -> None:
    """Register the bundled Vera TrueType fonts (idempotent)."""
    global _fonts_registered
    if _fonts_registered:
        return
    pdfmetrics.registerFont(TTFont(FONT_REGULAR, str(_FONT_DIR / "Vera.ttf")))
    pdfmetrics.registerFont(TTFont(FONT_BOLD, str(_FONT_DIR / "VeraBd.ttf")))
    _fonts_registered = True


def _wrap_text(text: str, font: str, size: float, max_width: float) -> list[str]:
    """Greedy word-wrap `text` to lines no wider than `max_width`."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and pdfmetrics.stringWidth(candidate, font, size) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def draw_title_page(
    pdf: canvas.Canvas, layout: InteriorLayout, title: str, author: str
) -> None:
    """Draw the centred title page — title (Vera Bold 32) + author (Vera 16)."""
    center_x = layout.page_width / 2
    title_lines = _wrap_text(title, FONT_BOLD, TITLE_SIZE, layout.text_safe_width)
    line_height = TITLE_SIZE * 1.2
    y = layout.page_height / 2 + line_height * len(title_lines) / 2
    pdf.setFont(FONT_BOLD, TITLE_SIZE)
    for line in title_lines:
        pdf.drawCentredString(center_x, y, line)
        y -= line_height
    pdf.setFont(FONT_REGULAR, AUTHOR_SIZE)
    pdf.drawCentredString(center_x, y - AUTHOR_SIZE * 1.6, author)


def draw_copyright_page(
    pdf: canvas.Canvas,
    layout: InteriorLayout,
    author: str,
    year: int,
    tips: tuple[str, ...],
) -> None:
    """Draw the copyright line plus a caller-supplied tips block.

    The tips tuple is rendered verbatim, line-by-line, after a blank spacer.
    Coloring books pass marker-bleed tips; puzzle books pass solver tips.
    """
    # Front-matter text uses the wider text-safe inset (0.5"), not the 0.25"
    # image margin — KDP flags text nearer the trim edge on a bleed PDF.
    margin_x = (layout.page_width - layout.text_safe_width) / 2
    y = (layout.page_height + layout.text_safe_height) / 2 - BODY_SIZE
    pdf.setFont(FONT_REGULAR, BODY_SIZE)
    lines: list[str] = [f"Copyright © {year} {author}. All rights reserved.", ""]
    lines.extend(tips)
    for line in lines:
        pdf.drawString(margin_x, y, line)
        y -= BODY_SIZE * 1.8
