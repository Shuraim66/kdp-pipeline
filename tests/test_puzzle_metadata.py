"""Tests for the book-type-aware metadata prompt + checklist (step 7)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from src.config.loader import load_niche_config
from src.config.schema import NicheConfig
from src.generators.metadata import (
    BookMetadata,
    _product_type_and_contents,
    build_checklist,
    build_metadata_prompt,
)

_NICHES_DIR = Path(__file__).resolve().parent.parent / "niches"
_SNAPSHOT_BOOK_ID = UUID("12345678-1234-5678-1234-567812345678")


def _puzzle_dict(**overrides: Any) -> dict[str, Any]:
    """A small puzzle niche config (12 mazes + 12 solutions = 24 pages)."""
    data: dict[str, Any] = {
        "slug": "maze_test_v1",
        "niche": "kids_mazes",
        "imprint": "pawpress",
        "book_type": "puzzle_maze",
        "book": {
            "trim_size": "8.5x11",
            "page_count": 24,
            "price_usd": 6.99,
            "target_audience": "Kids 6-10",
        },
        "metadata": {
            "title_seed": "Big Maze Adventure",
            "subtitle_seed": "Fun puzzles for kids",
            "author": "PawPress Studio",
            "keywords_seed": ["maze book for kids", "kids puzzle book"],
            "categories": [
                "Books > Children's > Activity > Mazes",
                "Books > Children's > Activity > General",
            ],
        },
        "cover": {
            "background_color": "#fff8e7",
            "accent_color": "#c45a3a",
            "text_color": "#2b1810",
            "hero_subject": "h",
            "hero_style": "s",
            "font_family": "Fraunces",
            "bullets": ["12 fun mazes", "Easy to medium", "Full solutions"],
        },
        "puzzle": {
            "type": "maze",
            "algorithm": "prim",
            "count": 12,
            "difficulty_curve": ["easy", "medium"],
            "grid_sizes": {"easy": [8, 8], "medium": [10, 10], "hard": [12, 12]},
            "include_solutions": True,
            "solutions_section": "end",
        },
        "front_matter": {"title_page": True, "intro_page": True, "intro_text": "Welcome!"},
    }
    data.update(overrides)
    return data


def _puzzle_config() -> NicheConfig:
    return NicheConfig.model_validate(_puzzle_dict())


def _puzzle_config_with_curve(curve: list[str]) -> NicheConfig:
    """A test puzzle config with a custom difficulty curve."""
    data = _puzzle_dict()
    data["puzzle"]["difficulty_curve"] = curve
    return NicheConfig.model_validate(data)


# --- _product_type_and_contents (regression for the easy-to-hard bug) -----


@pytest.mark.parametrize(
    ("curve", "expected_phrase"),
    [
        # Three difficulties — alphabetical sort would give "easy to medium"
        # (because sorted strings = ['easy','hard','medium'] and stride-2 picks
        # ['easy','medium']). The ordinal sort must pick the actual span.
        (["easy", "medium", "hard"], "easy to hard"),
        # The real kids_book curve — used to bake the listing copy that
        # mismatched the cover before the fix.
        (["easy", "easy", "medium", "medium", "hard"], "easy to hard"),
        # Two-difficulty spans.
        (["easy", "hard"], "easy to hard"),
        (["medium", "hard"], "medium to hard"),
        (["easy", "medium"], "easy to medium"),
        # Single-difficulty curves drop the " to " phrase entirely.
        (["easy"], "easy"),
        (["medium"], "medium"),
        (["hard"], "hard"),
    ],
)
def test_difficulty_phrase_respects_ordinal_order(curve: list[str], expected_phrase: str) -> None:
    """Curve difficulties must order by ordinal (easy < medium < hard), not alphabetically.

    Pre-fix bug: sorted(set(['easy','medium','hard'])) was alphabetical
    (['easy','hard','medium']) and the renderer's stride pulled 'easy' +
    'medium', silently advertising 'easy to medium' on books that ship
    hard mazes. Listing copy ended up lying about the difficulty range.
    """
    config = _puzzle_config_with_curve(curve)
    _product_type, contents_block = _product_type_and_contents(config)
    assert f"({expected_phrase})" in contents_block, (
        f"contents block for curve={curve} should mention '({expected_phrase})'; "
        f"got: {contents_block}"
    )


# --- build_metadata_prompt -------------------------------------------------


def test_puzzle_prompt_uses_maze_book_product_type() -> None:
    config = _puzzle_config()
    _system, user = build_metadata_prompt(config, book_id=_SNAPSHOT_BOOK_ID)
    assert "Generate KDP listing metadata for this maze book" in user
    assert 'Title must contain "maze book" and "maze book for kids"' in user
    # Coloring-only phrasing must NOT appear in a puzzle prompt.
    assert "coloring book" not in user
    assert "Subject areas" not in user
    assert "bold and easy line-art" not in user


def test_puzzle_prompt_contents_block_mentions_count_and_solutions() -> None:
    config = _puzzle_config()
    _system, user = build_metadata_prompt(config, book_id=_SNAPSHOT_BOOK_ID)
    assert "12 hand-crafted maze puzzles" in user
    assert "easy to medium" in user
    assert "full solutions section at the back" in user


def test_puzzle_prompt_without_solutions_section_omits_solutions_clause() -> None:
    config = NicheConfig.model_validate(
        _puzzle_dict(
            book={
                "trim_size": "8.5x11",
                "page_count": 24,
                "price_usd": 6.99,
                "target_audience": "Kids 6-10",
            },
            puzzle={
                "type": "maze",
                "algorithm": "prim",
                "count": 24,  # page_count == count when no solutions section
                "difficulty_curve": ["easy"],
                "grid_sizes": {"easy": [8, 8], "medium": [10, 10], "hard": [12, 12]},
                "include_solutions": False,
                "solutions_section": "end",
            },
        )
    )
    _system, user = build_metadata_prompt(config, book_id=_SNAPSHOT_BOOK_ID)
    assert "full solutions section" not in user


def test_coloring_prompt_unchanged_after_parameterisation() -> None:
    """Re-check the snapshot — duplicates the regression test in test_metadata.py
    but lives here too so a puzzle-only test run still catches drift."""
    import hashlib

    config = load_niche_config(_NICHES_DIR / "cozy_dogs_v1.yaml")
    _system, user = build_metadata_prompt(config, book_id=_SNAPSHOT_BOOK_ID)
    assert (
        hashlib.sha256(user.encode()).hexdigest()
        == "805cd480f0b3d24e3402e69f837a8e5b58f99b89ddcbe8d72368970e9f4d7445"
    )


# --- build_checklist -------------------------------------------------------


def _stub_metadata() -> BookMetadata:
    return BookMetadata(
        title="Big Maze Adventure",
        subtitle="Fun mazes",
        description="d" * 1600,
        keywords=[f"keyword {i}" for i in range(7)],
        categories=[
            "Books > Children's > Activity > Mazes",
            "Books > Children's > Activity > General",
        ],
    )


def test_puzzle_checklist_renders_puzzle_ai_disclosure(make_book) -> None:
    config = _puzzle_config()
    book = make_book(slug=config.slug, title="Big Maze Adventure")
    text = build_checklist(book, config, _stub_metadata(), interior_page_count=28)
    assert "AI disclosure**: AI used for cover hero only" in text
    assert "Interior is algorithmically generated" in text
    # The coloring disclosure must NOT appear in a puzzle checklist.
    assert "AI-assisted generation throughout" not in text


def test_coloring_checklist_renders_coloring_ai_disclosure(make_niche_config, make_book) -> None:
    config = make_niche_config()
    book = make_book(slug=config.slug)
    text = build_checklist(book, config, _stub_metadata(), interior_page_count=26)
    assert "AI disclosure**: AI-assisted generation throughout" in text
