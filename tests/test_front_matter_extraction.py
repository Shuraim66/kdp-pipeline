"""Tests for the shared front-matter helpers (extracted from interior.py).

The extraction must keep coloring interior PDFs unchanged AND make the helpers
callable standalone so the puzzle pipeline can reuse them.
"""

from __future__ import annotations

from pathlib import Path

import pdfplumber
from pypdf import PdfReader
from reportlab.pdfgen import canvas
from src.generators.front_matter import (
    FONT_REGULAR,
    draw_copyright_page,
    draw_title_page,
    register_fonts,
)
from src.utils.kdp_specs import interior_layout


def test_register_fonts_is_idempotent() -> None:
    register_fonts()
    register_fonts()  # second call must not raise


def test_draw_title_page_writes_title_and_author(tmp_path: Path) -> None:
    register_fonts()
    layout = interior_layout("8.5x8.5")
    pdf_path = tmp_path / "title_only.pdf"
    pdf = canvas.Canvas(
        str(pdf_path),
        pagesize=(layout.page_width, layout.page_height),
        initialFontName=FONT_REGULAR,
    )
    draw_title_page(pdf, layout, "Standalone Title", "Standalone Author")
    pdf.showPage()
    pdf.save()

    with pdfplumber.open(str(pdf_path)) as opened:
        text = opened.pages[0].extract_text() or ""
    assert "Standalone Title" in text
    assert "Standalone Author" in text


def test_draw_copyright_page_renders_custom_tips(tmp_path: Path) -> None:
    """The shared helper takes tips as a required arg — verify a maze-style block renders."""
    register_fonts()
    layout = interior_layout("8.5x11")
    pdf_path = tmp_path / "copyright_only.pdf"
    pdf = canvas.Canvas(
        str(pdf_path),
        pagesize=(layout.page_width, layout.page_height),
        initialFontName=FONT_REGULAR,
    )
    maze_tips = (
        "Tips for solving:",
        "• Use a pencil so you can erase wrong turns.",
        "• Solutions are at the back if you get stuck.",
    )
    draw_copyright_page(pdf, layout, "Maze Author", 2026, maze_tips)
    pdf.showPage()
    pdf.save()

    with pdfplumber.open(str(pdf_path)) as opened:
        text = opened.pages[0].extract_text() or ""
    assert "Copyright © 2026 Maze Author" in text
    assert "Tips for solving:" in text
    assert "erase wrong turns" in text
    # And the coloring tip MUST NOT appear — we passed maze tips only.
    assert "markers" not in text.lower()


def test_extracted_helpers_produce_same_two_page_count(tmp_path: Path) -> None:
    """A title page + a copyright page is exactly 2 pages — sanity-check the showPage cadence."""
    register_fonts()
    layout = interior_layout("8.5x8.5")
    pdf_path = tmp_path / "two_pages.pdf"
    pdf = canvas.Canvas(
        str(pdf_path),
        pagesize=(layout.page_width, layout.page_height),
        initialFontName=FONT_REGULAR,
    )
    draw_title_page(pdf, layout, "Title", "Author")
    pdf.showPage()
    draw_copyright_page(pdf, layout, "Author", 2026, ("Tip line",))
    pdf.showPage()
    pdf.save()

    assert len(PdfReader(str(pdf_path)).pages) == 2
