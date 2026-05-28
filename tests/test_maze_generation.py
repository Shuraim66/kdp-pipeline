"""Tests for the deterministic mazelib wrapper."""

from __future__ import annotations

from typing import Any

import pytest
from src.config.schema import NicheConfig
from src.puzzle.mazes.generator import generate_book_mazes, generate_maze


def _puzzle_dict(**overrides: Any) -> dict[str, Any]:
    """A minimal valid puzzle niche config dict for tests."""
    data: dict[str, Any] = {
        "slug": "maze_gen_test_v1",
        "niche": "test_mazes",
        "book_type": "puzzle_maze",
        "book": {
            "trim_size": "8.5x11",
            "page_count": 80,
            "price_usd": 6.99,
            "target_audience": "Kids 6-10",
        },
        "metadata": {
            "title_seed": "Test Maze Book",
            "subtitle_seed": "Sub",
            "author": "Test Studio",
            "keywords_seed": ["kw1", "kw2"],
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
            "count": 40,
            "difficulty_curve": ["easy", "easy", "medium", "medium", "hard"],
            "grid_sizes": {"easy": [10, 10], "medium": [15, 15], "hard": [20, 20]},
        },
        "front_matter": {"title_page": True, "intro_page": True},
    }
    data.update(overrides)
    return data


def _puzzle_spec(**overrides: Any):
    config = NicheConfig.model_validate(_puzzle_dict(**overrides))
    return config.require_puzzle().puzzle


# --- determinism -----------------------------------------------------------


def test_same_seed_produces_identical_maze() -> None:
    a = generate_maze(grid_size=(10, 10), algorithm="prim", seed=42, difficulty="easy", index=0)
    b = generate_maze(grid_size=(10, 10), algorithm="prim", seed=42, difficulty="easy", index=0)
    assert a.grid == b.grid
    assert a.solution == b.solution
    assert a.start == b.start and a.end == b.end


def test_different_seeds_produce_different_grids() -> None:
    a = generate_maze(grid_size=(10, 10), algorithm="prim", seed=42, difficulty="easy", index=0)
    b = generate_maze(grid_size=(10, 10), algorithm="prim", seed=43, difficulty="easy", index=0)
    assert a.grid != b.grid


def test_unknown_algorithm_raises() -> None:
    with pytest.raises(ValueError, match="unknown maze algorithm"):
        generate_maze(
            grid_size=(10, 10),
            algorithm="towers_of_hanoi",
            seed=1,
            difficulty="easy",
            index=0,
        )


# --- maze invariants -------------------------------------------------------


def test_grid_size_matches_mazelib_convention() -> None:
    maze = generate_maze(grid_size=(10, 8), algorithm="prim", seed=1, difficulty="easy", index=0)
    # Mazelib generates (2H + 1, 2W + 1); we pass (width, height) = (10, 8)
    # → (2*8+1, 2*10+1) = (17, 21).
    assert maze.height == 17
    assert maze.width == 21


def test_solution_path_is_orthogonally_connected_and_in_passages() -> None:
    maze = generate_maze(
        grid_size=(15, 15),
        algorithm="prim",
        seed=99,
        difficulty="medium",
        index=0,
    )
    assert len(maze.solution) >= 2
    grid = maze.grid
    for r, c in maze.solution:
        assert grid[r][c] == 0, f"solution cell ({r},{c}) is a wall"
    for (r1, c1), (r2, c2) in zip(maze.solution, maze.solution[1:], strict=False):
        # Orthogonally adjacent — Manhattan distance exactly 1.
        assert abs(r1 - r2) + abs(c1 - c2) == 1, (
            f"solution cells {(r1, c1)} -> {(r2, c2)} are not orthogonally adjacent"
        )


def test_start_and_end_sit_on_the_perimeter() -> None:
    maze = generate_maze(grid_size=(10, 10), algorithm="prim", seed=7, difficulty="easy", index=0)
    h, w = maze.height, maze.width
    for cell in (maze.start, maze.end):
        on_perimeter = cell[0] in (0, h - 1) or cell[1] in (0, w - 1)
        assert on_perimeter, f"{cell} not on perimeter ({h}x{w})"


# --- diagonal entrances + long-solution guarantee -------------------------


@pytest.mark.parametrize("seed", [1, 7, 42, 99, 12345])
def test_start_and_end_on_OPPOSITE_walls(seed: int) -> None:
    """Generator must pick opposing walls — never two entrances on the same edge."""
    maze = generate_maze(
        grid_size=(10, 10), algorithm="prim", seed=seed, difficulty="easy", index=0
    )
    h, w = maze.height, maze.width

    def _wall_of(cell: tuple[int, int]) -> str:
        r, c = cell
        if r == 0:
            return "top"
        if r == h - 1:
            return "bottom"
        if c == 0:
            return "left"
        if c == w - 1:
            return "right"
        return "interior"  # pragma: no cover

    opposite = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
    s_wall = _wall_of(maze.start)
    e_wall = _wall_of(maze.end)
    assert e_wall == opposite[s_wall], (
        f"start on {s_wall}, end on {e_wall} — must be opposite walls"
    )


@pytest.mark.parametrize("seed", [1, 7, 42, 99, 12345])
def test_start_and_end_diagonally_separated(seed: int) -> None:
    """On the perpendicular axis, start and end must be far apart — not co-linear.

    A short near-straight solution (like the original 026_easy bug) only
    happens when both entrances cluster near the same edge along the
    perpendicular axis. The diagonal-third placement makes that
    impossible: on a 21-wide grid, start col <= 7 implies end col >= 13.
    """
    maze = generate_maze(
        grid_size=(10, 10), algorithm="prim", seed=seed, difficulty="easy", index=0
    )
    h, w = maze.height, maze.width
    sr, sc = maze.start
    er, ec = maze.end
    if sr in (0, h - 1):
        # Top-bottom axis — check column separation
        col_gap = abs(sc - ec)
        # cols range [1, w-1) → max diff = w-2. Third of that = (w-2)/3.
        assert col_gap >= (w - 2) // 3, (
            f"start col {sc} and end col {ec} are within {col_gap} — path could skim a corridor"
        )
    else:
        row_gap = abs(sr - er)
        assert row_gap >= (h - 2) // 3, (
            f"start row {sr} and end row {er} are within {row_gap} — path could skim a corridor"
        )


@pytest.mark.parametrize(
    ("grid_size", "difficulty"),
    [
        ((10, 10), "easy"),
        ((12, 12), "medium"),
        ((15, 15), "hard"),
    ],
)
def test_solution_length_meets_minimum_threshold(grid_size: tuple[int, int], difficulty) -> None:
    """Every generated maze must have a solution path of at least 1.5x grid_max cells.

    This is the threshold the generator uses to reject and retry short mazes.
    If a maze slips through with a shorter path, the retry loop is broken.
    Tested across difficulties + multiple seeds so a one-off lucky/unlucky
    pass doesn't hide the regression.
    """
    min_cells = int(max(grid_size) * 1.5)
    for seed in (1, 7, 42, 99, 12345):
        maze = generate_maze(
            grid_size=grid_size,
            algorithm="prim",
            seed=seed,
            difficulty=difficulty,
            index=0,
        )
        assert len(maze.solution) >= min_cells, (
            f"{difficulty} {grid_size} maze (seed={seed}) has only "
            f"{len(maze.solution)} solution cells — below {min_cells} threshold"
        )


# --- book-level generation -------------------------------------------------


def test_generate_book_mazes_returns_correct_count() -> None:
    spec = _puzzle_spec()
    mazes = generate_book_mazes(spec, book_seed_prefix=12345)
    assert len(mazes) == 40


def test_generate_book_mazes_cycles_difficulty_curve() -> None:
    spec = _puzzle_spec()
    mazes = generate_book_mazes(spec, book_seed_prefix=12345)
    expected_difficulties = [spec.difficulty_curve[i % 5] for i in range(40)]
    assert [m.difficulty for m in mazes] == expected_difficulties


def test_generate_book_mazes_picks_grid_size_per_difficulty() -> None:
    spec = _puzzle_spec()
    mazes = generate_book_mazes(spec, book_seed_prefix=12345)
    # easy is (10, 10) cells → grid is 21x21; hard (20, 20) → 41x41.
    easy_maze = next(m for m in mazes if m.difficulty == "easy")
    hard_maze = next(m for m in mazes if m.difficulty == "hard")
    assert (easy_maze.height, easy_maze.width) == (21, 21)
    assert (hard_maze.height, hard_maze.width) == (41, 41)


def test_generate_book_mazes_seed_in_expected_range() -> None:
    """Each maze's recorded seed is within [base_seed, base_seed + MAX_RETRIES).

    The base seed for maze ``i`` is ``book_seed_prefix * 10000 + i``. When the
    natural seed produces a too-short solution, generate_maze bumps to
    seed+1, seed+2, ... so the recorded seed can be slightly higher than
    the base — pin the window so a runaway retry loop would be caught.
    """
    spec = _puzzle_spec()
    mazes = generate_book_mazes(spec, book_seed_prefix=12345)
    for i, m in enumerate(mazes[:3]):
        base = 12345 * 10000 + i
        assert base <= m.seed < base + 20, (
            f"maze {i}: seed {m.seed} not within [{base}, {base + 20})"
        )


def test_generate_book_mazes_is_deterministic_across_runs() -> None:
    spec = _puzzle_spec()
    run_one = generate_book_mazes(spec, book_seed_prefix=12345)
    run_two = generate_book_mazes(spec, book_seed_prefix=12345)
    assert [m.grid for m in run_one] == [m.grid for m in run_two]
    assert [m.solution for m in run_one] == [m.solution for m in run_two]
    assert [m.seed for m in run_one] == [m.seed for m in run_two]


def test_fixed_seed_overrides_per_maze_derivation() -> None:
    spec = _puzzle_spec()
    spec_fixed = spec.model_copy(
        update={"fixed_seed": 999, "count": 3, "difficulty_curve": ["easy", "easy", "easy"]}
    )
    mazes = generate_book_mazes(spec_fixed, book_seed_prefix=12345)
    # Every maze starts from the same fixed seed and therefore retries
    # identically — all three land on the same actual_seed and produce the
    # same grid. The actual_seed may differ from 999 if 999's natural maze
    # had a too-short solution and the retry loop bumped to 999+attempt.
    base_seed = 999
    assert all(base_seed <= m.seed < base_seed + 20 for m in mazes)
    assert len({m.seed for m in mazes}) == 1
    assert mazes[0].grid == mazes[1].grid == mazes[2].grid


# --- counts per difficulty bucket -----------------------------------------


def test_difficulty_curve_cycling_yields_expected_bucket_counts() -> None:
    """40 mazes, curve [easy,easy,medium,medium,hard] → 16/16/8."""
    spec = _puzzle_spec()
    mazes = generate_book_mazes(spec, book_seed_prefix=12345)
    counts: dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
    for m in mazes:
        counts[m.difficulty] += 1
    assert counts == {"easy": 16, "medium": 16, "hard": 8}
