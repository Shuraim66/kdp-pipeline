"""Tests for interior PDF assembly and interior-PDF QA."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
