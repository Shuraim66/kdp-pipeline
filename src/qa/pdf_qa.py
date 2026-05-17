"""Interior-PDF QA — verify an assembled PDF meets KDP print requirements.

Checks the page count, that every page is the expected bleed size, that
embedded design images clear 300 DPI, that the file is under KDP's size
ceiling, that the PDF opens cleanly, and that every font is embedded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from src.utils.kdp_specs import POINTS_PER_INCH, InteriorLayout

_MAX_FILE_SIZE_MB = 150.0
_MIN_IMAGE_DPI = 300.0
_DIMENSION_TOLERANCE_PT = 1.0
_FONT_FILE_KEYS = ("/FontFile", "/FontFile2", "/FontFile3")


@dataclass(frozen=True, slots=True)
class PDFQAResult:
    """The outcome of `check_interior_pdf`."""

    page_count: int
    expected_pages: int
    dimensions_ok: bool
    min_image_dpi: float
    file_size_mb: float
    fonts_embedded: bool
    opens_cleanly: bool
    issues: list[str]

    @property
    def passed(self) -> bool:
        """Whether the PDF cleared every QA check."""
        return not self.issues


def _image_pixel_widths(page: Any) -> list[int]:
    """The pixel widths of every image XObject referenced by a page."""
    widths: list[int] = []
    resources = page.get("/Resources")
    xobjects = resources.get("/XObject") if resources is not None else None
    if xobjects is None:
        return widths
    for ref in xobjects.values():
        obj = ref.get_object()
        if obj.get("/Subtype") == "/Image" and "/Width" in obj:
            widths.append(int(obj["/Width"]))
    return widths


def _font_is_embedded(font: Any) -> bool:
    """Whether a font dictionary embeds its font program."""
    descriptor = font.get("/FontDescriptor")
    if descriptor is None:
        descendants = font.get("/DescendantFonts")
        if descendants is None:
            return False
        return all(_font_is_embedded(d.get_object()) for d in descendants)
    descriptor = descriptor.get_object()
    return any(key in descriptor for key in _FONT_FILE_KEYS)


def _all_fonts_embedded(reader: PdfReader) -> bool:
    for page in reader.pages:
        resources = page.get("/Resources")
        fonts = resources.get("/Font") if resources is not None else None
        if fonts is None:
            continue
        for ref in fonts.values():
            if not _font_is_embedded(ref.get_object()):
                return False
    return True


def _min_image_dpi(reader: PdfReader, layout: InteriorLayout) -> float:
    """Lowest effective DPI across embedded design images.

    Design images are embedded across the safe area, so DPI is the image's
    pixel width over the safe width in inches. Returns infinity when the PDF
    embeds no images.
    """
    display_inches = layout.safe_width / POINTS_PER_INCH
    widths = [w for page in reader.pages for w in _image_pixel_widths(page)]
    if not widths:
        return float("inf")
    return min(width / display_inches for width in widths)


def check_interior_pdf(path: Path, *, expected_pages: int, layout: InteriorLayout) -> PDFQAResult:
    """Run every interior-PDF QA check and collect the issues found."""
    issues: list[str] = []
    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb >= _MAX_FILE_SIZE_MB:
        issues.append(f"file size {file_size_mb:.1f} MB >= {_MAX_FILE_SIZE_MB} MB limit")

    try:
        reader = PdfReader(str(path))
        page_count = len(reader.pages)
    except Exception as exc:
        issues.append(f"PDF failed to open: {exc}")
        return PDFQAResult(
            page_count=0,
            expected_pages=expected_pages,
            dimensions_ok=False,
            min_image_dpi=0.0,
            file_size_mb=file_size_mb,
            fonts_embedded=False,
            opens_cleanly=False,
            issues=issues,
        )

    if page_count != expected_pages:
        issues.append(f"page count {page_count} != expected {expected_pages}")

    dimensions_ok = True
    for index, page in enumerate(reader.pages, start=1):
        box = page.mediabox
        width, height = float(box.width), float(box.height)
        if (
            abs(width - layout.page_width) > _DIMENSION_TOLERANCE_PT
            or abs(height - layout.page_height) > _DIMENSION_TOLERANCE_PT
        ):
            dimensions_ok = False
            issues.append(
                f"page {index} is {width:.0f}x{height:.0f}pt, expected "
                f"{layout.page_width:.0f}x{layout.page_height:.0f}pt"
            )
            break

    min_dpi = _min_image_dpi(reader, layout)
    if min_dpi < _MIN_IMAGE_DPI:
        issues.append(f"min image DPI {min_dpi:.0f} < {_MIN_IMAGE_DPI:.0f}")

    fonts_embedded = _all_fonts_embedded(reader)
    if not fonts_embedded:
        issues.append("not every font is embedded")

    return PDFQAResult(
        page_count=page_count,
        expected_pages=expected_pages,
        dimensions_ok=dimensions_ok,
        min_image_dpi=min_dpi,
        file_size_mb=file_size_mb,
        fonts_embedded=fonts_embedded,
        opens_cleanly=True,
        issues=issues,
    )
