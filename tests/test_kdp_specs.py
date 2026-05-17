"""Tests for KDP interior-page geometry."""

from __future__ import annotations

import pytest
from src.utils.kdp_specs import interior_layout, parse_trim_size


def test_parse_trim_size() -> None:
    assert parse_trim_size("8.5x8.5") == (8.5, 8.5)
    assert parse_trim_size("8.5x11") == (8.5, 11.0)


def test_interior_layout_square_book() -> None:
    layout = interior_layout("8.5x8.5")
    # 8.5" trim + 0.125" bleed each side -> 8.75" page.
    assert layout.page_width == pytest.approx(8.75 * 72)
    assert layout.page_height == pytest.approx(8.75 * 72)
    assert layout.trim_width == pytest.approx(8.5 * 72)
    # 8.5" trim - 0.25" safe margin each side -> 8.0" live area.
    assert layout.safe_width == pytest.approx(8.0 * 72)
    assert layout.safe_height == pytest.approx(8.0 * 72)


def test_interior_layout_tall_book() -> None:
    layout = interior_layout("8.5x11")
    assert layout.page_width == pytest.approx(8.75 * 72)
    assert layout.page_height == pytest.approx(11.25 * 72)
    assert layout.safe_width == pytest.approx(8.0 * 72)
    assert layout.safe_height == pytest.approx(10.5 * 72)
