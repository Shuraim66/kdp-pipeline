"""Tests for cover math, the font loader, and cover composition."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from src.generators.cover import _barcode_zone_px, _hex_rgb, build_cover, cover_layout
from src.qa.pdf_qa import check_cover_pdf
from src.utils.fonts import CoverFonts, load_cover_fonts
from src.utils.kdp_specs import POINTS_PER_INCH, compute_cover_dimensions


def _hero_png(path: Path) -> Path:
    """A plain colored square standing in for the Fal-generated hero."""
    array = np.full((600, 600, 3), (240, 180, 90), dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)
    return path


def _thumb_png(path: Path) -> Path:
    """A black-bordered white square — a stand-in for a filtered design page."""
    array = np.full((400, 400, 3), 255, dtype=np.uint8)
    array[20:380, 20:30] = 0  # left bar
    array[20:380, 370:380] = 0  # right bar
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


def test_build_cover_produces_pdf_and_preview(tmp_path: Path, make_niche_config, make_book) -> None:
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


def test_imprint_palette_swap_changes_plate_cream(
    tmp_path: Path, make_niche_config, make_book
) -> None:
    # Default niche → cottagecore_earth palette (#f7f1e3 cream); switching to
    # the pawpress imprint → warm_friendly palette (#fff8e7 cream). The plate
    # is rendered in both, so at least one pixel of each cream must appear in
    # the respective rendered cover (and the other niche's cream must not).
    book = make_book(slug="cov_v1", title="A Cover Test Book")
    hero = _hero_png(tmp_path / "hero.png")
    cottagecore_cream = np.array([247, 241, 227], dtype=np.int16)
    warm_friendly_cream = np.array([255, 248, 231], dtype=np.int16)

    def _plate_pixel_counts(image_path: Path) -> tuple[int, int]:
        arr = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.int16)
        return (
            int((np.abs(arr - cottagecore_cream).sum(axis=2) == 0).sum()),
            int((np.abs(arr - warm_friendly_cream).sum(axis=2) == 0).sum()),
        )

    default_config = make_niche_config()
    build_cover(
        book,
        default_config,
        hero_path=hero,
        output_dir=tmp_path / "qhp",
        interior_page_count=default_config.book.page_count + 2,
        dpi=72,
    )
    cot, warm = _plate_pixel_counts(tmp_path / "qhp" / "cov_v1" / "cover" / "cover.png")
    assert cot > 0, "default imprint should render the cottagecore_earth plate cream"
    assert warm == 0, "default imprint must not render warm_friendly cream"

    pawpress_config = default_config.model_copy(update={"imprint": "pawpress"})
    build_cover(
        book,
        pawpress_config,
        hero_path=hero,
        output_dir=tmp_path / "pp",
        interior_page_count=pawpress_config.book.page_count + 2,
        dpi=72,
    )
    cot2, warm2 = _plate_pixel_counts(tmp_path / "pp" / "cov_v1" / "cover" / "cover.png")
    assert warm2 > 0, "pawpress imprint should render the warm_friendly plate cream"
    assert cot2 == 0, "pawpress imprint must not render cottagecore_earth cream"


def test_back_cover_leaves_barcode_zone_clear(tmp_path: Path, make_niche_config, make_book) -> None:
    # KDP overlays an EAN barcode on the back panel's bottom-right corner of
    # the spread (spine-side bottom). No artwork — including the page-preview
    # thumbnail row, which most plausibly intrudes — may overlap that rect.
    base = make_niche_config()
    config = base.model_copy(
        update={"cover": base.cover.model_copy(update={"thumbnails": [0, 1, 2]})}
    )
    book = make_book(slug="cov_v1", title="A Cover Test Book")
    hero = _hero_png(tmp_path / "hero.png")
    filtered_dir = tmp_path / "cov_v1" / "images" / "filtered"
    filtered_dir.mkdir(parents=True)
    for i in range(3):
        _thumb_png(filtered_dir / f"{i:03d}.png")

    interior_pages = config.book.page_count + 2
    build_cover(
        book,
        config,
        hero_path=hero,
        output_dir=tmp_path,
        interior_page_count=interior_pages,
        dpi=72,
    )

    layout = cover_layout(interior_pages, 8.5, 8.5, dpi=72)
    x0, y0, x1, y1 = _barcode_zone_px(layout)
    composed = np.asarray(
        Image.open(tmp_path / "cov_v1" / "cover" / "cover.png").convert("RGB"),
        dtype=np.int16,
    )
    background = np.array(_hex_rgb(config.cover.background_color), dtype=np.int16)
    deviation = np.abs(composed[y0:y1, x0:x1] - background).sum(axis=2)
    intrusion = int((deviation > 30).sum())
    assert intrusion == 0, (
        f"{intrusion} drawn pixel(s) intrude into the barcode zone ({x0},{y0},{x1},{y1})"
    )


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
