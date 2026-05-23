"""Tests for the imprint configuration loader."""

from __future__ import annotations

import pytest
from src.config.imprint import ImprintConfig, ImprintNotFoundError, load_imprint


def test_quiet_hours_press_imprint_loads_with_default_palette() -> None:
    imprint = load_imprint("quiet_hours_press")
    assert isinstance(imprint, ImprintConfig)
    assert imprint.name == "Quiet Hours Press"
    assert imprint.publisher_field == "Quiet Hours Press"
    assert imprint.visual_style.cover_palette_name == "cottagecore_earth"


def test_pawpress_imprint_loads_with_warm_friendly_palette() -> None:
    imprint = load_imprint("pawpress")
    assert imprint.name == "PawPress"
    assert imprint.publisher_field == "PawPress Books"
    assert imprint.visual_style.cover_palette_name == "warm_friendly"
    assert imprint.visual_style.cover_font_primary == "Fraunces"


def test_unknown_imprint_raises_clear_error() -> None:
    with pytest.raises(ImprintNotFoundError, match="no imprint config"):
        load_imprint("does_not_exist_v999")
