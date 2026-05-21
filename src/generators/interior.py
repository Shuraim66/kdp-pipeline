"""Interior PDF assembly — title page, copyright page, one design per page.

The PDF is sized with bleed on all four sides; every page is the same size.
Design images are converted to grayscale and embedded losslessly, centred in
the safe area. Text uses Bitstream Vera (bundled with reportlab) so all fonts
are embedded — a KDP requirement.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import reportlab
from PIL import Image as PILImage
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from src.config.schema import NicheConfig
from src.db.models import Book
from src.utils.kdp_specs import InteriorLayout, interior_layout
from src.utils.logging import logger

_FONT_REGULAR = "Vera"
_FONT_BOLD = "VeraBd"
_FONT_DIR = Path(reportlab.__file__).resolve().parent / "fonts"

_TITLE_SIZE = 32
_AUTHOR_SIZE = 16
_BODY_SIZE = 11

_COPYRIGHT_TIPS = (
    "Tips for coloring:",
    "• Test markers on the last page first — some may bleed through.",
    "• Pages are single-sided to prevent bleed-through.",
    "• Use colored pencils, markers, gel pens, or watercolor pencils.",
    "• Take your time and enjoy the process.",
)

_fonts_registered = False


def _register_fonts() -> None:
    """Register the bundled Vera TrueType fonts (idempotent)."""
    global _fonts_registered
    if _fonts_registered:
        return
    pdfmetrics.registerFont(TTFont(_FONT_REGULAR, str(_FONT_DIR / "Vera.ttf")))
    pdfmetrics.registerFont(TTFont(_FONT_BOLD, str(_FONT_DIR / "VeraBd.ttf")))
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


def _fit(
    pixel_width: int, pixel_height: int, max_width: float, max_height: float
) -> tuple[float, float]:
    """Scale ``pixel_width x pixel_height`` to fit a box, preserving aspect."""
    scale = min(max_width / pixel_width, max_height / pixel_height)
    return pixel_width * scale, pixel_height * scale


def _draw_title_page(pdf: canvas.Canvas, layout: InteriorLayout, title: str, author: str) -> None:
    center_x = layout.page_width / 2
    # Wrap to the text-safe width so a long title can't reach the trim edge.
    title_lines = _wrap_text(title, _FONT_BOLD, _TITLE_SIZE, layout.text_safe_width)
    line_height = _TITLE_SIZE * 1.2
    y = layout.page_height / 2 + line_height * len(title_lines) / 2
    pdf.setFont(_FONT_BOLD, _TITLE_SIZE)
    for line in title_lines:
        pdf.drawCentredString(center_x, y, line)
        y -= line_height
    pdf.setFont(_FONT_REGULAR, _AUTHOR_SIZE)
    pdf.drawCentredString(center_x, y - _AUTHOR_SIZE * 1.6, author)


def _draw_copyright_page(
    pdf: canvas.Canvas, layout: InteriorLayout, author: str, year: int
) -> None:
    # Front-matter text uses the wider text-safe inset (0.5"), not the 0.25"
    # image margin — KDP flags text nearer the trim edge on a bleed PDF.
    margin_x = (layout.page_width - layout.text_safe_width) / 2
    y = (layout.page_height + layout.text_safe_height) / 2 - _BODY_SIZE
    pdf.setFont(_FONT_REGULAR, _BODY_SIZE)
    lines = [f"Copyright © {year} {author}. All rights reserved.", ""]
    lines.extend(_COPYRIGHT_TIPS)
    for line in lines:
        pdf.drawString(margin_x, y, line)
        y -= _BODY_SIZE * 1.8


def _draw_design_page(pdf: canvas.Canvas, layout: InteriorLayout, image_path: Path) -> None:
    with PILImage.open(image_path) as handle:
        grayscale = handle.convert("L")  # independent of the file handle
    reader = ImageReader(grayscale)
    width, height = grayscale.size
    draw_width, draw_height = _fit(width, height, layout.safe_width, layout.safe_height)
    x = (layout.page_width - draw_width) / 2
    y = (layout.page_height - draw_height) / 2
    pdf.drawImage(reader, x, y, width=draw_width, height=draw_height)


def build_interior_pdf(
    book: Book,
    config: NicheConfig,
    *,
    filtered_images: list[Path],
    output_path: Path,
    author: str,
    year: int | None = None,
) -> Path:
    """Assemble the interior PDF: title page, copyright page, then designs."""
    if not filtered_images:
        raise ValueError("no filtered images to assemble into an interior PDF")
    _register_fonts()
    layout = interior_layout(config.book.trim_size)
    title = book.title or config.metadata.title_seed
    year = year if year is not None else datetime.now(UTC).year
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # initialFontName=Vera keeps reportlab from seeding every page with the
    # non-embedded standard Helvetica — so every font in the PDF is embedded.
    pdf = canvas.Canvas(
        str(output_path),
        pagesize=(layout.page_width, layout.page_height),
        initialFontName=_FONT_REGULAR,
    )
    pdf.setTitle(title)
    pdf.setAuthor(author)

    _draw_title_page(pdf, layout, title, author)
    pdf.showPage()
    _draw_copyright_page(pdf, layout, author, year)
    pdf.showPage()
    for image_path in filtered_images:
        _draw_design_page(pdf, layout, image_path)
        pdf.showPage()
    pdf.save()

    logger.info(
        "interior PDF written: {} ({} pages)",
        output_path,
        len(filtered_images) + 2,
    )
    return output_path
