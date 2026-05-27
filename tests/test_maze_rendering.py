"""Tests for the PIL maze renderer."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from src.puzzle.mazes.generator import generate_maze
from src.puzzle.mazes.renderer import render_maze, render_solution


def _maze(seed: int = 1, grid_size: tuple[int, int] = (10, 10)):
    return generate_maze(
        grid_size=grid_size, algorithm="prim", seed=seed, difficulty="easy", index=0
    )


def test_render_maze_returns_square_image_at_requested_size() -> None:
    img = render_maze(_maze(), target_side_px=2250)
    assert isinstance(img, Image.Image)
    assert img.size == (2250, 2250)
    assert img.mode == "RGB"


def test_render_maze_paints_walls_black_and_passages_white() -> None:
    img = render_maze(_maze(), target_side_px=600)
    arr = np.array(img)
    # White and black are the dominant colours; greens/reds are tiny start/end cells.
    pixel_set = {tuple(p) for p in arr.reshape(-1, 3)[::97]}
    assert (255, 255, 255) in pixel_set
    assert (0, 0, 0) in pixel_set


def test_render_maze_marks_start_green_and_end_red() -> None:
    maze = _maze()
    img = render_maze(maze, target_side_px=600)
    arr = np.array(img)
    # Spot-check: somewhere inside the start cell must be the green colour,
    # somewhere inside the end cell must be the red colour.
    grid_side = max(maze.height, maze.width)
    cell_px = 600 // grid_side
    pad = (600 - cell_px * grid_side) // 2

    def _cell_centre_pixel(cell: tuple[int, int]) -> tuple[int, int, int]:
        r, c = cell
        y = pad + r * cell_px + cell_px // 2
        x = pad + c * cell_px + cell_px // 2
        return tuple(int(v) for v in arr[y, x])

    assert _cell_centre_pixel(maze.start) == (76, 175, 80)
    assert _cell_centre_pixel(maze.end) == (229, 57, 53)


def test_render_maze_rejects_non_positive_target_side() -> None:
    with pytest.raises(ValueError, match="target_side_px"):
        render_maze(_maze(), target_side_px=0)


# --- solution renderer -----------------------------------------------------


def test_render_solution_adds_red_pixels_not_present_in_plain_maze() -> None:
    maze = _maze(grid_size=(15, 15))
    plain = np.array(render_maze(maze, target_side_px=900))
    with_solution = np.array(render_solution(maze, target_side_px=900))
    # The solution line is the SAME red as the end cell, so pure red appears
    # in both — but the solution version has STRICTLY MORE red pixels
    # (everywhere along the path through the maze passages).
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


def test_render_solution_solution_path_passes_through_passage_cells_only() -> None:
    """The drawn red line must only touch passage cells (no painting over walls)."""
    maze = _maze(grid_size=(10, 10), seed=11)
    sol = np.array(render_solution(maze, target_side_px=900))
    grid_side = max(maze.height, maze.width)
    cell_px = 900 // grid_side
    pad = (900 - cell_px * grid_side) // 2
    # At every solution cell's centre we should find red; never at a wall centre.
    for r, c in maze.solution:
        y = pad + r * cell_px + cell_px // 2
        x = pad + c * cell_px + cell_px // 2
        assert tuple(int(v) for v in sol[y, x]) == (229, 57, 53)
