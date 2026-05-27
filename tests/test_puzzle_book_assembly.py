"""End-to-end tests for puzzle interior PDF assembly + puzzle QA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pypdf import PdfReader
from src.config.schema import NicheConfig
from src.puzzle.book_builder import (
    assemble_puzzle_pdf,
    generate_and_render_book,
    render_all_puzzles,
)
from src.puzzle.mazes.generator import generate_book_mazes
from src.puzzle.qa import validate_puzzles


def _puzzle_dict(**overrides: Any) -> dict[str, Any]:
    """A 12-maze puzzle config (minimum page_count is 24)."""
    data: dict[str, Any] = {
        "slug": "asm_test_v1",
        "niche": "test_mazes",
        "book_type": "puzzle_maze",
        "book": {
            "trim_size": "8.5x11",
            "page_count": 24,  # 12 mazes + 12 solutions
            "price_usd": 6.99,
            "target_audience": "Kids 6-10",
        },
        "metadata": {
            "title_seed": "Test Maze Book",
            "subtitle_seed": "Sub",
            "author": "Test Studio",
            "keywords_seed": ["kw"],
            "categories": ["Books > A > B", "Books > C > D"],
        },
        "cover": {
            "background_color": "#fff8e7",
            "accent_color": "#c45a3a",
            "text_color": "#2b1810",
            "hero_subject": "h",
            "hero_style": "s",
            "font_family": "Fraunces",
            "bullets": ["one", "two", "three"],
        },
        "puzzle": {
            "type": "maze",
            "algorithm": "prim",
            "count": 12,
            "difficulty_curve": ["easy", "easy", "medium"],
            "grid_sizes": {"easy": [8, 8], "medium": [10, 10], "hard": [12, 12]},
            "include_solutions": True,
            "solutions_section": "end",
        },
        "front_matter": {
            "title_page": True,
            "intro_page": True,
            "intro_text": "Welcome! Use a pencil so you can erase.",
        },
    }
    data.update(overrides)
    return data


def _config(**overrides: Any) -> NicheConfig:
    return NicheConfig.model_validate(_puzzle_dict(**overrides))


# --- render_all_puzzles ----------------------------------------------------


def test_render_all_puzzles_writes_paired_pngs(tmp_path: Path) -> None:
    config = _config()
    puzzle_body = config.require_puzzle()
    mazes = generate_book_mazes(puzzle_body.puzzle, book_seed_prefix=1)
    maze_paths, solution_paths = render_all_puzzles(
        mazes,
        target_side_px=600,
        output_dir=tmp_path,
        slug=config.slug,
        wall_thickness_px=4,  # small target → small floor so the maze fits
        path_wall_ratios={str(k): v for k, v in puzzle_body.puzzle.render.path_wall_ratios.items()},
    )
    assert len(maze_paths) == len(solution_paths) == 12
    for path in maze_paths + solution_paths:
        assert path.is_file()
        assert path.stat().st_size > 0


# --- puzzle QA -------------------------------------------------------------


def test_validate_puzzles_passes_for_correctly_rendered_pngs(tmp_path: Path) -> None:
    config = _config()
    puzzle_body = config.require_puzzle()
    mazes = generate_book_mazes(puzzle_body.puzzle, book_seed_prefix=1)
    maze_paths, solution_paths = render_all_puzzles(
        mazes,
        target_side_px=900,
        output_dir=tmp_path,
        slug=config.slug,
        wall_thickness_px=4,
        path_wall_ratios={str(k): v for k, v in puzzle_body.puzzle.render.path_wall_ratios.items()},
    )
    issues = validate_puzzles(
        maze_paths=maze_paths,
        solution_paths=solution_paths,
        expected_dimensions=(900, 900),
    )
    assert issues == []


def test_validate_puzzles_flags_missing_file(tmp_path: Path) -> None:
    issues = validate_puzzles(
        maze_paths=[tmp_path / "nope.png"],
        solution_paths=[],
        expected_dimensions=(900, 900),
    )
    assert any("missing on disk" in i for i in issues)


def test_validate_puzzles_flags_wrong_dimensions(tmp_path: Path) -> None:
    from PIL import Image

    bad = tmp_path / "small.png"
    Image.new("RGB", (100, 100), (255, 255, 255)).save(bad)
    issues = validate_puzzles(
        maze_paths=[bad],
        solution_paths=[],
        expected_dimensions=(900, 900),
    )
    assert any("wrong dimensions" in i for i in issues)


# --- assemble_puzzle_pdf ---------------------------------------------------


def test_assemble_puzzle_pdf_page_count_with_all_front_matter_and_solutions(
    tmp_path: Path, make_book
) -> None:
    """12 mazes + title + copyright + intro + divider + 12 solutions = 27 pages."""
    config = _config()
    book = make_book(slug=config.slug, title="Test Maze Book")
    mazes, maze_paths, solution_paths = generate_and_render_book(
        book, config, output_dir=tmp_path, target_side_px=900
    )
    pdf_path = tmp_path / "interior.pdf"
    assemble_puzzle_pdf(
        book,
        config,
        mazes=mazes,
        maze_paths=maze_paths,
        solution_paths=solution_paths,
        output_path=pdf_path,
        author="Test Studio",
    )
    assert pdf_path.is_file()
    # title(1) + copyright(1) + intro(1) + 12 mazes + divider(1) + 12 solutions = 28.
    assert len(PdfReader(str(pdf_path)).pages) == 28


def test_assemble_puzzle_pdf_skips_front_matter_when_toggled_off(tmp_path: Path, make_book) -> None:
    """Title and intro pages can be toggled off; copyright is always present."""
    config = _config(front_matter={"title_page": False, "intro_page": False, "intro_text": ""})
    book = make_book(slug=config.slug, title="No Front Matter")
    mazes, maze_paths, solution_paths = generate_and_render_book(
        book, config, output_dir=tmp_path, target_side_px=900
    )
    pdf_path = tmp_path / "interior.pdf"
    assemble_puzzle_pdf(
        book,
        config,
        mazes=mazes,
        maze_paths=maze_paths,
        solution_paths=solution_paths,
        output_path=pdf_path,
        author="Test Studio",
    )
    # copyright(1) + 12 mazes + divider(1) + 12 solutions = 26.
    assert len(PdfReader(str(pdf_path)).pages) == 26


def test_assemble_puzzle_pdf_rejects_empty_maze_list(tmp_path: Path, make_book) -> None:
    config = _config()
    book = make_book(slug=config.slug, title="Empty")
    with pytest.raises(ValueError, match="no maze pages"):
        assemble_puzzle_pdf(
            book,
            config,
            mazes=[],
            maze_paths=[],
            solution_paths=[],
            output_path=tmp_path / "interior.pdf",
            author="Test Studio",
        )
