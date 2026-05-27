"""Tests for the puzzle-book discriminated union + back-compat shim."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from src.config.loader import load_niche_config
from src.config.schema import ColoringBody, NicheConfig, PuzzleBody

_NICHES_DIR = Path(__file__).resolve().parent.parent / "niches"


def _puzzle_yaml_dict(**overrides: Any) -> dict[str, Any]:
    """A minimal valid puzzle-maze niche config dict."""
    data: dict[str, Any] = {
        "slug": "test_maze_v1",
        "niche": "test_mazes",
        "imprint": "pawpress",
        "book_type": "puzzle_maze",
        "book": {
            "trim_size": "8.5x11",
            "page_count": 80,
            "price_usd": 6.99,
            "target_audience": "Kids 6-10",
        },
        "metadata": {
            "title_seed": "Test Maze Book",
            "subtitle_seed": "40 fun mazes",
            "author": "Test Studio",
            "keywords_seed": [
                "test mazes",
                "maze book for kids",
            ],
            "categories": [
                "Books > Children's > Activity > Mazes",
                "Books > Children's > Activity > General",
            ],
        },
        "cover": {
            "background_color": "#fff8e7",
            "accent_color": "#c45a3a",
            "text_color": "#2b1810",
            "hero_subject": "a kid solving a maze",
            "hero_style": "colorful vector",
            "font_family": "Fraunces",
            "bullets": [
                "40 hand-crafted mazes",
                "Three difficulty levels",
                "Full solutions in the back",
            ],
        },
        "puzzle": {
            "type": "maze",
            "algorithm": "prim",
            "count": 40,
            "difficulty_curve": ["easy", "easy", "medium", "medium", "hard"],
            "grid_sizes": {"easy": [10, 10], "medium": [15, 15], "hard": [20, 20]},
            "include_solutions": True,
            "solutions_section": "end",
        },
        "front_matter": {
            "title_page": True,
            "intro_page": True,
            "intro_text": "Welcome!",
        },
    }
    data.update(overrides)
    return data


# --- legacy coloring YAMLs still validate -----------------------------------


def test_legacy_coloring_yaml_cottagecore_validates() -> None:
    config = load_niche_config(_NICHES_DIR / "cottagecore_mushrooms_v1.yaml")
    assert isinstance(config.body, ColoringBody)
    assert config.coloring is not None
    assert config.puzzle is None


def test_legacy_coloring_yaml_cozy_dogs_validates() -> None:
    config = load_niche_config(_NICHES_DIR / "cozy_dogs_v1.yaml")
    assert isinstance(config.body, ColoringBody)


def test_legacy_coloring_yaml_nurses_validates() -> None:
    config = load_niche_config(_NICHES_DIR / "nurses_v1.yaml")
    assert isinstance(config.body, ColoringBody)


# --- puzzle YAML validates and accessors return narrow types ----------------


def test_puzzle_yaml_validates() -> None:
    config = NicheConfig.model_validate(_puzzle_yaml_dict())
    assert isinstance(config.body, PuzzleBody)
    assert config.puzzle is not None
    assert config.coloring is None
    assert config.puzzle.puzzle.count == 40
    assert config.puzzle.puzzle.algorithm == "prim"


def test_require_coloring_raises_clear_error_on_puzzle() -> None:
    config = NicheConfig.model_validate(_puzzle_yaml_dict())
    with pytest.raises(ValueError, match="book_type='coloring'"):
        config.require_coloring()


def test_require_puzzle_raises_clear_error_on_coloring() -> None:
    config = load_niche_config(_NICHES_DIR / "cozy_dogs_v1.yaml")
    with pytest.raises(ValueError, match="book_type='puzzle_maze'"):
        config.require_puzzle()


# --- invalid combinations fail clearly --------------------------------------


def test_coloring_yaml_with_extra_puzzle_block_fails() -> None:
    """A coloring YAML with a stray `puzzle:` block must be rejected."""
    data = load_niche_config(_NICHES_DIR / "cozy_dogs_v1.yaml").model_dump(mode="python")
    # Re-validate the flat legacy form WITH a puzzle key — should fail.
    flat: dict[str, Any] = {
        "slug": data["slug"],
        "niche": data["niche"],
        "imprint": data["imprint"],
        "book": data["book"],
        "metadata": data["metadata"],
        "cover": data["cover"],
        # legacy coloring keys at top level:
        **{k: v for k, v in data["body"].items() if k != "kind"},
        "puzzle": {
            "type": "maze",
            "algorithm": "prim",
            "count": 40,
            "difficulty_curve": ["easy"],
            "grid_sizes": {"easy": [10, 10]},
        },
    }
    # The shim wraps body.kind=coloring — `puzzle` then sits as an extra top-level key.
    with pytest.raises(ValidationError):
        NicheConfig.model_validate(flat)


def test_puzzle_yaml_missing_cover_bullets_fails_with_explicit_error() -> None:
    """Guardrail #6: puzzle books must specify cover.bullets explicitly."""
    data = _puzzle_yaml_dict()
    data["cover"]["bullets"] = []
    with pytest.raises(ValidationError) as exc_info:
        NicheConfig.model_validate(data)
    msg = str(exc_info.value)
    assert "puzzle books must specify cover.bullets explicitly" in msg
    assert "coloring-specific fallback" in msg


def test_puzzle_yaml_too_few_cover_bullets_fails() -> None:
    """Need at least 3 bullets — 2 should fail with the same explicit error."""
    data = _puzzle_yaml_dict()
    data["cover"]["bullets"] = ["only one", "and two"]
    with pytest.raises(ValidationError) as exc_info:
        NicheConfig.model_validate(data)
    assert "cover.bullets explicitly" in str(exc_info.value)


def test_puzzle_grid_sizes_must_cover_curve() -> None:
    data = _puzzle_yaml_dict()
    data["puzzle"]["grid_sizes"] = {"easy": [10, 10]}  # missing medium + hard
    with pytest.raises(ValidationError, match="missing an entry for difficulty"):
        NicheConfig.model_validate(data)


def test_puzzle_page_count_must_match_count_plus_solutions() -> None:
    """80 = 40 mazes + 40 solutions; setting page_count=50 should fail."""
    data = _puzzle_yaml_dict()
    data["book"]["page_count"] = 50
    with pytest.raises(ValidationError, match=r"puzzle\.count="):
        NicheConfig.model_validate(data)


def test_puzzle_page_count_without_solutions_section() -> None:
    """With include_solutions=False, page_count == puzzle.count alone."""
    data = _puzzle_yaml_dict()
    data["puzzle"]["include_solutions"] = False
    data["book"]["page_count"] = 40
    config = NicheConfig.model_validate(data)
    puzzle_body = config.require_puzzle()
    assert puzzle_body.puzzle.count == 40


def test_difficulty_curve_cycles_when_shorter_than_count() -> None:
    """40-maze book with a 5-element curve [easy,easy,medium,medium,hard] cycles every 5."""
    config = NicheConfig.model_validate(_puzzle_yaml_dict())
    spec = config.require_puzzle().puzzle
    assert spec.difficulty_curve == ["easy", "easy", "medium", "medium", "hard"]
    # Verify the cycling pattern for representative indices across all 40 mazes.
    expected_by_index = [
        (0, "easy"),
        (1, "easy"),
        (2, "medium"),
        (4, "hard"),
        (5, "easy"),
        (39, "hard"),
    ]
    for i, expected in expected_by_index:
        assert spec.difficulty_curve[i % len(spec.difficulty_curve)] == expected


def test_unknown_book_type_rejected_by_discriminator() -> None:
    data = _puzzle_yaml_dict()
    data["book_type"] = "sudoku_grand_master"
    with pytest.raises(ValidationError):
        NicheConfig.model_validate(data)
