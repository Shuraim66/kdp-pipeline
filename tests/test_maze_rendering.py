"""Tests for the PIL maze renderer (difficulty-aware path/wall ratios)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from src.puzzle.mazes.generator import generate_maze
from src.puzzle.mazes.models import Difficulty
from src.puzzle.mazes.renderer import (
    _PATH_TO_WALL_RATIO,
    _layout,
    render_maze,
    render_solution,
)


def _maze(
    seed: int = 1,
    grid_size: tuple[int, int] = (10, 10),
    difficulty: Difficulty = "easy",
):
    return generate_maze(
        grid_size=grid_size,
        algorithm="prim",
        seed=seed,
        difficulty=difficulty,
        index=0,
    )


# --- shape / colours -------------------------------------------------------


def test_render_maze_returns_square_image_at_requested_size() -> None:
    img = render_maze(_maze(), target_side_px=2250)
    assert isinstance(img, Image.Image)
    assert img.size == (2250, 2250)
    assert img.mode == "RGB"


def test_render_maze_paints_walls_black_and_passages_white() -> None:
    img = render_maze(_maze(), target_side_px=600)
    arr = np.array(img)
    pixel_set = {tuple(p) for p in arr.reshape(-1, 3)[::97]}
    assert (255, 255, 255) in pixel_set
    assert (0, 0, 0) in pixel_set


def test_render_maze_marks_start_green_and_end_red() -> None:
    maze = _maze()
    img = render_maze(maze, target_side_px=600)
    arr = np.array(img)
    # Look up exact cell rectangles via the renderer's layout helper.
    _, _, x_offsets, y_offsets, pad_x, pad_y = _layout(maze, target_side_px=600)

    def _centre_pixel(cell: tuple[int, int]) -> tuple[int, int, int]:
        r, c = cell
        x = pad_x + (x_offsets[c] + x_offsets[c + 1]) // 2
        y = pad_y + (y_offsets[r] + y_offsets[r + 1]) // 2
        return tuple(int(v) for v in arr[y, x])

    assert _centre_pixel(maze.start) == (76, 175, 80)
    assert _centre_pixel(maze.end) == (229, 57, 53)


def test_render_maze_rejects_non_positive_target_side() -> None:
    with pytest.raises(ValueError, match="target_side_px"):
        render_maze(_maze(), target_side_px=0)


# --- difficulty-aware ratios ----------------------------------------------


@pytest.mark.parametrize(
    ("difficulty", "grid_size"),
    [("easy", (10, 10)), ("medium", (15, 15)), ("hard", (20, 20))],
)
def test_path_width_matches_difficulty_ratio(
    difficulty: Difficulty, grid_size: tuple[int, int]
) -> None:
    maze = _maze(grid_size=grid_size, difficulty=difficulty)
    wall_px, path_px, *_ = _layout(maze, target_side_px=2250)
    ratio = _PATH_TO_WALL_RATIO[difficulty]
    # path_px is round(ratio * wall_px); allow a 1-pixel rounding tolerance.
    assert abs(path_px - ratio * wall_px) <= 1
    # And the path is always at least as wide as a wall.
    assert path_px >= wall_px


def test_easy_paths_are_wider_than_hard_paths_for_same_target_side() -> None:
    """A 10x10 easy maze must render with chunkier passages than a 20x20 hard maze."""
    easy = _maze(grid_size=(10, 10), difficulty="easy")
    hard = _maze(grid_size=(20, 20), difficulty="hard")
    _, easy_path, *_ = _layout(easy, target_side_px=2250)
    _, hard_path, *_ = _layout(hard, target_side_px=2250)
    # Easy paths should be at least 2x as wide as hard paths at print resolution.
    assert easy_path >= hard_path * 2


def test_maze_fits_inside_target_side() -> None:
    """For every difficulty + grid combo we ship, the maze must fit the page."""
    for difficulty, grid in [
        ("easy", (10, 10)),
        ("medium", (15, 15)),
        ("hard", (20, 20)),
    ]:
        maze = _maze(grid_size=grid, difficulty=difficulty)  # type: ignore[arg-type]
        _, _, x_offsets, y_offsets, pad_x, pad_y = _layout(maze, target_side_px=2250)
        assert x_offsets[-1] + 2 * pad_x <= 2250 + 1  # ≤ target ± 1 px rounding
        assert y_offsets[-1] + 2 * pad_y <= 2250 + 1


def test_easy_maze_path_width_in_inches_is_crayon_friendly() -> None:
    """Easy paths should be ≥ 0.4 in wide at 300 DPI (chunky crayon target)."""
    maze = _maze(grid_size=(10, 10), difficulty="easy")
    _, path_px, *_ = _layout(maze, target_side_px=2250)
    path_inches = path_px / 300.0
    assert path_inches >= 0.4, f"easy paths must be ≥ 0.4 in wide; got {path_inches:.2f} in"


# --- solution renderer -----------------------------------------------------


def test_render_solution_adds_red_pixels_not_present_in_plain_maze() -> None:
    maze = _maze(grid_size=(15, 15), difficulty="medium")
    plain = np.array(render_maze(maze, target_side_px=900))
    with_solution = np.array(render_solution(maze, target_side_px=900))
    red_plain = ((plain[:, :, 0] == 229) & (plain[:, :, 1] == 57) & (plain[:, :, 2] == 53)).sum()
    red_solution = (
        (with_solution[:, :, 0] == 229)
        & (with_solution[:, :, 1] == 57)
        & (with_solution[:, :, 2] == 53)
    ).sum()
    assert red_solution > red_plain * 5, (
        f"solution overlay should paint many red pixels (plain={red_plain},"
        f" solution={red_solution})"
    )


def test_render_solution_returns_same_size_as_plain_maze() -> None:
    maze = _maze()
    plain = render_maze(maze, target_side_px=1200)
    sol = render_solution(maze, target_side_px=1200)
    assert plain.size == sol.size


def test_render_solution_passes_through_passage_cells_only() -> None:
    """The drawn red line must touch every solution cell's centre."""
    maze = _maze(grid_size=(10, 10), seed=11, difficulty="easy")
    sol = np.array(render_solution(maze, target_side_px=900))
    _, _, x_offsets, y_offsets, pad_x, pad_y = _layout(maze, target_side_px=900)
    for r, c in maze.solution:
        x = pad_x + (x_offsets[c] + x_offsets[c + 1]) // 2
        y = pad_y + (y_offsets[r] + y_offsets[r + 1]) // 2
        assert tuple(int(v) for v in sol[y, x]) == (229, 57, 53)
