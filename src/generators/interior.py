"""Interior PDF assembly — title page, copyright page, one design per page.

The PDF is sized with bleed on all four sides; every page is the same size.
Design images are converted to grayscale and embedded losslessly, centred in
the safe area. Text uses Bitstream Vera (bundled with reportlab) so all fonts
are embedded — a KDP requirement.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from src.config.schema import NicheConfig
from src.db.models import Book
from src.generators.front_matter import (
    FONT_REGULAR,
    draw_copyright_page,
    draw_title_page,
    register_fonts,
)
from src.utils.kdp_specs import InteriorLayout, interior_layout
from src.utils.logging import logger

_COPYRIGHT_TIPS: tuple[str, ...] = (
    "Tips for coloring:",
    "• Test markers on the last page first — some may bleed through.",
    "• Pages are single-sided to prevent bleed-through.",
    "• Use colored pencils, markers, gel pens, or watercolor pencils.",
    "• Take your time and enjoy the process.",
)


def _fit(
    pixel_width: int, pixel_height: int, max_width: float, max_height: float
) -> tuple[float, float]:
    """Scale ``pixel_width x pixel_height`` to fit a box, preserving aspect."""
    scale = min(max_width / pixel_width, max_height / pixel_height)
    return pixel_width * scale, pixel_height * scale


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
    register_fonts()
    layout = interior_layout(config.book.trim_size)
    title = book.title or config.metadata.title_seed
    year = year if year is not None else datetime.now(UTC).year
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # initialFontName=Vera keeps reportlab from seeding every page with the
    # non-embedded standard Helvetica — so every font in the PDF is embedded.
    pdf = canvas.Canvas(
        str(output_path),
        pagesize=(layout.page_width, layout.page_height),
        initialFontName=FONT_REGULAR,
    )
    pdf.setTitle(title)
    pdf.setAuthor(author)

    draw_title_page(pdf, layout, title, author)
    pdf.showPage()
    draw_copyright_page(pdf, layout, author, year, _COPYRIGHT_TIPS)
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
