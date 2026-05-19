"""Tests for cover math, the font loader, and cover composition."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from src.generators.cover import build_cover, cover_layout
from src.qa.pdf_qa import check_cover_pdf
from src.utils.fonts import CoverFonts, load_cover_fonts
from src.utils.kdp_specs import POINTS_PER_INCH, compute_cover_dimensions


def _hero_png(path: Path) -> Path:
    """A plain colored square standing in for the Fal-generated hero."""
    array = np.full((600, 600, 3), (240, 180, 90), dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)
    return path


# --- cover dimensions --------------------------------------------------------


def test_compute_cover_dimensions_matches_kdp_example() -> None:
    # Spec example: a 52-page 8.5x8.5 book on white paper.
    dims = compute_cover_dimensions(52, 8.5, 8.5)
    assert dims.total_width_px == 5210
    assert dims.total_height_px == 2625
    assert dims.total_height_in == pytest.approx(8.75)
    assert dims.total_width_in == pytest.approx(17.367, abs=0.01)
    assert dims.spine_width_in == pytest.approx(0.117, abs=0.001)


def test_compute_cover_dimensions_rejects_unknown_paper() -> None:
    with pytest.raises(ValueError, match="paper"):
        compute_cover_dimensions(52, 8.5, 8.5, paper="papyrus")


# --- font loader -------------------------------------------------------------


def test_load_cover_fonts_resolves_to_real_files() -> None:
    fonts = load_cover_fonts()
    assert isinstance(fonts, CoverFonts)
    # Whether the real fonts or the Vera fallback, every path must be usable.
    assert fonts.title.is_file()
    assert fonts.title_black.is_file()
    assert fonts.title_regular.is_file()
    assert fonts.body.is_file()
    assert fonts.body_bold.is_file()


# --- cover layout ------------------------------------------------------------


def test_cover_layout_columns_are_contiguous() -> None:
    layout = cover_layout(52, 8.5, 8.5, dpi=300)
    assert layout.back[1] == layout.spine[0]
    assert layout.spine[1] == layout.front[0]
    assert layout.front[1] <= layout.width


# --- cover composition -------------------------------------------------------


def test_build_cover_produces_pdf_and_preview(
    tmp_path: Path, make_niche_config, make_book
) -> None:
    config = make_niche_config()
    book = make_book(slug="cov_v1", title="A Cover Test Book")
    hero = _hero_png(tmp_path / "hero.png")

    pdf = build_cover(
        book,
        config,
        hero_path=hero,
        output_dir=tmp_path,
        interior_page_count=config.book.page_count + 2,
        dpi=72,
    )

    assert pdf.name == "cover.pdf"
    assert pdf.is_file()
    assert (tmp_path / "cov_v1" / "cover" / "cover.png").is_file()


def test_built_cover_passes_qa(tmp_path: Path, make_niche_config, make_book) -> None:
    config = make_niche_config()
    book = make_book(slug="cov_v1", title="A Cover Test Book")
    hero = _hero_png(tmp_path / "hero.png")
    interior_pages = config.book.page_count + 2

    pdf = build_cover(
        book,
        config,
        hero_path=hero,
        output_dir=tmp_path,
        interior_page_count=interior_pages,
        dpi=72,
    )

    dims = compute_cover_dimensions(interior_pages, 8.5, 8.5)
    result = check_cover_pdf(
        pdf,
        expected_width_pt=dims.total_width_in * POINTS_PER_INCH,
        expected_height_pt=dims.total_height_in * POINTS_PER_INCH,
    )
    assert result.passed, result.issues
    assert result.page_count == 1
    assert result.dimensions_ok
