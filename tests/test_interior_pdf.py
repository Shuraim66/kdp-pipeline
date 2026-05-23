"""Tests for interior PDF assembly and interior-PDF QA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pdfplumber
import pytest
from PIL import Image
from pypdf import PdfReader
from src.generators.interior import build_interior_pdf
from src.qa.pdf_qa import check_interior_pdf
from src.utils.kdp_specs import interior_layout

# 2550px across an 8.0" safe area is 318 DPI — comfortably above the 300 floor.
_IMAGE_PX = 2550


def _design_png(path: Path) -> Path:
    """A white page with a bold black bar — stands in for a coloring design."""
    array = np.full((_IMAGE_PX, _IMAGE_PX, 3), 255, dtype=np.uint8)
    array[_IMAGE_PX // 2 - 40 : _IMAGE_PX // 2 + 40, 100 : _IMAGE_PX - 100] = 0
    Image.fromarray(array, mode="RGB").save(path)
    return path


def _designs(tmp_path: Path, count: int) -> list[Path]:
    return [_design_png(tmp_path / f"design_{i}.png") for i in range(count)]


def test_build_interior_pdf_has_frontmatter_and_designs(
    tmp_path: Path, make_niche_config, make_book
) -> None:
    config = make_niche_config()
    book = make_book(slug="t_v1", title="A Test Coloring Book")
    pdf_path = tmp_path / "pdf" / "interior.pdf"

    build_interior_pdf(
        book,
        config,
        filtered_images=_designs(tmp_path, 3),
        output_path=pdf_path,
        author="A. Tester",
    )

    assert pdf_path.is_file()
    # title page + copyright page + 3 design pages.
    assert len(PdfReader(str(pdf_path)).pages) == 5


def test_check_interior_pdf_passes_for_valid_pdf(
    tmp_path: Path, make_niche_config, make_book
) -> None:
    config = make_niche_config()
    book = make_book(slug="t_v1", title="A Test Coloring Book")
    pdf_path = tmp_path / "pdf" / "interior.pdf"
    build_interior_pdf(
        book,
        config,
        filtered_images=_designs(tmp_path, 3),
        output_path=pdf_path,
        author="A. Tester",
    )
    layout = interior_layout(config.book.trim_size)

    result = check_interior_pdf(pdf_path, expected_pages=5, layout=layout)

    assert result.passed, result.issues
    assert result.page_count == 5
    assert result.dimensions_ok
    assert result.fonts_embedded
    assert result.opens_cleanly
    assert result.min_image_dpi >= 300


def test_check_interior_pdf_flags_wrong_page_count(
    tmp_path: Path, make_niche_config, make_book
) -> None:
    config = make_niche_config()
    book = make_book(slug="t_v1", title="A Test Coloring Book")
    pdf_path = tmp_path / "pdf" / "interior.pdf"
    build_interior_pdf(
        book,
        config,
        filtered_images=_designs(tmp_path, 2),
        output_path=pdf_path,
        author="A. Tester",
    )
    layout = interior_layout(config.book.trim_size)

    result = check_interior_pdf(pdf_path, expected_pages=99, layout=layout)

    assert not result.passed
    assert any("page count" in issue for issue in result.issues)


# For an 8.5x8.5" bleed PDF: 630pt page with 9pt bleed → trim edges sit at
# (9, 9, 621, 621). KDP's Print Previewer flags text closer than 0.375" to a
# trim edge on a bleed PDF; we hold to the safer 0.5" = 36pt target.
_TRIM_INSET_PT = 36.0
_TRIM_GLYPH_TOL_PT = 1.0  # absorbs sub-point glyph-metric noise


def _assert_text_clears_trim(page: Any, page_size_pt: float, bleed_pt: float) -> None:
    """Every char on the pdfplumber page must sit >= 36pt from every trim edge."""
    trim_lo = bleed_pt
    trim_hi = page_size_pt - bleed_pt
    chars = page.chars
    assert chars, f"page {page.page_number} has no extractable text"
    for char in chars:
        glyph = repr(char["text"])
        left_clear = char["x0"] - trim_lo
        right_clear = trim_hi - char["x1"]
        bot_clear = char["y0"] - trim_lo
        top_clear = trim_hi - char["y1"]
        assert left_clear >= _TRIM_INSET_PT - _TRIM_GLYPH_TOL_PT, (
            f"page {page.page_number} {glyph}: left clearance {left_clear:.2f}pt "
            f"< required {_TRIM_INSET_PT}pt from trim {trim_lo}"
        )
        assert right_clear >= _TRIM_INSET_PT - _TRIM_GLYPH_TOL_PT, (
            f"page {page.page_number} {glyph}: right clearance {right_clear:.2f}pt "
            f"< required {_TRIM_INSET_PT}pt from trim {trim_hi}"
        )
        assert bot_clear >= _TRIM_INSET_PT - _TRIM_GLYPH_TOL_PT, (
            f"page {page.page_number} {glyph}: bottom clearance {bot_clear:.2f}pt "
            f"< required {_TRIM_INSET_PT}pt from trim {trim_lo}"
        )
        assert top_clear >= _TRIM_INSET_PT - _TRIM_GLYPH_TOL_PT, (
            f"page {page.page_number} {glyph}: top clearance {top_clear:.2f}pt "
            f"< required {_TRIM_INSET_PT}pt from trim {trim_hi}"
        )


def test_copyright_page_text_stays_within_text_safe_margins(
    tmp_path: Path, make_niche_config, make_book
) -> None:
    # Regression: KDP's Print Previewer flagged page 2 — its left-aligned
    # copyright/tips text sat only 0.25" from the trim edge. Front-matter text
    # must clear the trim by 0.5" (36pt) on all four sides.
    config = make_niche_config()
    book = make_book(slug="t_v1", title="A Test Coloring Book")
    pdf_path = tmp_path / "pdf" / "interior.pdf"
    build_interior_pdf(
        book,
        config,
        filtered_images=_designs(tmp_path, 3),
        output_path=pdf_path,
        author="A. Tester",
    )
    layout = interior_layout(config.book.trim_size)

    with pdfplumber.open(str(pdf_path)) as pdf:
        page = pdf.pages[1]  # page 2 — the copyright / tips page
        assert "Copyright" in (page.extract_text() or "")
        bleed_pt = (layout.page_height - layout.trim_height) / 2
        _assert_text_clears_trim(page, layout.page_height, bleed_pt)


def test_title_page_text_stays_within_text_safe_margins(
    tmp_path: Path, make_niche_config, make_book
) -> None:
    # Recurrence-proofing for page 1: a long future title that wraps must still
    # clear the same 0.5" trim margin as the copyright page.
    config = make_niche_config()
    book = make_book(slug="t_v1", title="A Test Coloring Book")
    pdf_path = tmp_path / "pdf" / "interior.pdf"
    build_interior_pdf(
        book,
        config,
        filtered_images=_designs(tmp_path, 3),
        output_path=pdf_path,
        author="A. Tester",
    )
    layout = interior_layout(config.book.trim_size)

    with pdfplumber.open(str(pdf_path)) as pdf:
        page = pdf.pages[0]  # page 1 — the title page
        assert "Test Coloring Book" in (page.extract_text() or "")
        bleed_pt = (layout.page_height - layout.trim_height) / 2
        _assert_text_clears_trim(page, layout.page_height, bleed_pt)


def test_build_interior_pdf_rejects_an_empty_image_list(
    tmp_path: Path, make_niche_config, make_book
) -> None:
    config = make_niche_config()
    book = make_book(slug="t_v1")
    with pytest.raises(ValueError, match="no filtered images"):
        build_interior_pdf(
            book,
            config,
            filtered_images=[],
            output_path=tmp_path / "interior.pdf",
            author="A. Tester",
        )
