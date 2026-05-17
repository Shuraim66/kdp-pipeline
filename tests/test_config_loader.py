"""Tests for niche config loading and strict validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from src.config.loader import compute_config_hash, load_niche_config
from src.config.schema import NicheConfig

_NICHES_DIR = Path(__file__).resolve().parent.parent / "niches"


def _valid_config() -> dict[str, Any]:
    """A minimal valid niche config: 12 subjects x 2 = 24 pages."""
    return {
        "slug": "test_niche_v1",
        "niche": "testing",
        "book": {
            "trim_size": "8.5x8.5",
            "page_count": 24,
            "price_usd": 9.99,
            "target_audience": "testers",
        },
        "style": {
            "art_style": "bold and easy line art",
            "line_weight": "very thick",
            "negative_prompts": "shading, gray, color",
        },
        "subjects": [f"test subject {i}" for i in range(12)],
        "variations_per_subject": 2,
        "composition_modifiers": ["centered composition", "diagonal composition"],
        "metadata": {
            "title_seed": "Test Coloring Book",
            "subtitle_seed": "24 Test Designs",
            "author": "Test Press",
            "keywords_seed": ["test coloring book", "test gifts"],
            "categories": ["Books > A > B", "Books > C > D"],
        },
        "cover": {
            "background_color": "#1a3a52",
            "accent_color": "#f4d35e",
            "text_color": "#ffffff",
            "hero_subject": "a test hero illustration",
            "hero_style": "flat vector",
            "font_family": "Bebas Neue",
        },
        "generation": {
            "model": "fal-ai/flux/schnell",
            "image_dimensions": [2550, 2550],
            "num_inference_steps": 4,
            "guidance_scale": 0.0,
            "fixed_seed": None,
        },
        "qa": {
            "min_white_pct": 90.0,
            "max_gray_pct": 3.0,
            "required_white_margin_px": 75,
            "required_dimensions": [2550, 2550],
            "max_retries_per_slot": 3,
        },
    }


def test_valid_config_passes() -> None:
    config = NicheConfig.model_validate(_valid_config())
    assert config.slug == "test_niche_v1"
    assert config.book.page_count == 24
    assert config.generation.image_dimensions == (2550, 2550)


def test_real_nurses_yaml_loads() -> None:
    config = load_niche_config(_NICHES_DIR / "nurses_v1.yaml")
    assert config.slug == "nurses_bold_easy_v1"
    produced = len(config.subjects) * config.variations_per_subject
    assert produced == config.book.page_count == 50


def test_schema_reference_is_valid() -> None:
    config = load_niche_config(_NICHES_DIR / "_schema.yaml")
    assert config.slug == "example_niche_v1"


def test_bad_slug_rejected() -> None:
    data = _valid_config()
    data["slug"] = "Bad-Slug!"
    with pytest.raises(ValidationError):
        NicheConfig.model_validate(data)


def test_subject_count_mismatch_rejected() -> None:
    data = _valid_config()
    data["subjects"] = data["subjects"][:5]  # 5 x 2 = 10, not 24
    with pytest.raises(ValidationError):
        NicheConfig.model_validate(data)


def test_bad_hex_color_rejected() -> None:
    data = _valid_config()
    data["cover"]["background_color"] = "navy"
    with pytest.raises(ValidationError):
        NicheConfig.model_validate(data)


def test_wrong_category_count_rejected() -> None:
    data = _valid_config()
    data["metadata"]["categories"] = ["Books > Only > One"]
    with pytest.raises(ValidationError):
        NicheConfig.model_validate(data)


def test_unknown_key_rejected() -> None:
    data = _valid_config()
    data["bogus_key"] = "unexpected"
    with pytest.raises(ValidationError):
        NicheConfig.model_validate(data)


def test_config_hash_is_deterministic() -> None:
    a = NicheConfig.model_validate(_valid_config())
    b = NicheConfig.model_validate(_valid_config())
    assert compute_config_hash(a) == compute_config_hash(b)


def test_config_hash_changes_with_content() -> None:
    a = NicheConfig.model_validate(_valid_config())
    data = _valid_config()
    data["niche"] = "a different niche"
    b = NicheConfig.model_validate(data)
    assert compute_config_hash(a) != compute_config_hash(b)
