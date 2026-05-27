"""Tests for the PIL maze renderer (per-niche path/wall ratios + wall-thickness floor)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from src.config.schema import PuzzleRenderSpec
from src.puzzle.mazes.generator import generate_maze
from src.puzzle.mazes.models import Difficulty
from src.puzzle.mazes.renderer import (
    DEFAULT_PATH_WALL_RATIOS,
    DEFAULT_WALL_THICKNESS_PX,
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
    img = render_maze(_maze(), target_side_px=600, wall_thickness_px=4)
    arr = np.array(img)
    pixel_set = {tuple(p) for p in arr.reshape(-1, 3)[::97]}
    assert (255, 255, 255) in pixel_set
    assert (0, 0, 0) in pixel_set


def test_render_maze_marks_start_green_and_end_red() -> None:
    maze = _maze()
    img = render_maze(maze, target_side_px=600, wall_thickness_px=4)
    arr = np.array(img)
    _, _, x_offsets, y_offsets, pad_x, pad_y = _layout(
        maze, target_side_px=600, wall_thickness_px=4
    )

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


# --- defaults stay in sync with the schema --------------------------------


def test_renderer_defaults_match_schema_defaults() -> None:
    """If schema's PuzzleRenderSpec defaults drift, the renderer's must follow."""
    spec = PuzzleRenderSpec()
    assert spec.path_wall_ratios == DEFAULT_PATH_WALL_RATIOS
    assert spec.wall_thickness_px == DEFAULT_WALL_THICKNESS_PX


# --- ratio + grid-size combos --------------------------------------------


@pytest.mark.parametrize(
    ("difficulty", "grid_size", "ratio"),
    [
        ("easy", (10, 10), 4.5),
        ("medium", (12, 12), 3.0),
        ("hard", (15, 15), 2.2),
    ],
)
def test_custom_ratio_is_honoured(
    difficulty: Difficulty, grid_size: tuple[int, int], ratio: float
) -> None:
    """A YAML-supplied ratio drives the path:wall pixel ratio in the output."""
    maze = _maze(grid_size=grid_size, difficulty=difficulty)
    wall_px, path_px, *_ = _layout(
        maze,
        target_side_px=2250,
        wall_thickness_px=12,
        path_wall_ratios={"easy": 4.5, "medium": 3.0, "hard": 2.2},
    )
    # path_px = int(ratio * wall_px) — floor, so allow a 1-pixel slack downward.
    expected = int(ratio * wall_px)
    assert path_px == expected
    assert path_px >= wall_px  # paths never narrower than walls


def test_default_ratios_produce_chunkier_easy_than_hard() -> None:
    """With the defaults at 2250px, easy paths are wider than hard paths."""
    easy = _maze(grid_size=(10, 10), difficulty="easy")
    hard = _maze(grid_size=(20, 20), difficulty="hard")
    _, easy_path, *_ = _layout(easy, target_side_px=2250)
    _, hard_path, *_ = _layout(hard, target_side_px=2250)
    assert easy_path > hard_path


def test_kids_yaml_ratios_lift_hard_path_above_pencil_threshold() -> None:
    """Kids YAML ratios put even hard paths above the 0.3 in pencil threshold at 300 DPI."""
    maze = _maze(grid_size=(15, 15), difficulty="hard")
    _, path_px, *_ = _layout(
        maze,
        target_side_px=2250,
        wall_thickness_px=12,
        path_wall_ratios={"easy": 4.5, "medium": 3.0, "hard": 2.2},
    )
    path_inches = path_px / 300.0
    assert path_inches >= 0.30, f"hard paths must clear 0.30 in for kids; got {path_inches:.2f} in"


def test_easy_yaml_ratios_clear_crayon_threshold() -> None:
    """Kids YAML easy paths clear the 0.5 in chunky-crayon threshold at 300 DPI."""
    maze = _maze(grid_size=(10, 10), difficulty="easy")
    _, path_px, *_ = _layout(
        maze,
        target_side_px=2250,
        wall_thickness_px=12,
        path_wall_ratios={"easy": 4.5, "medium": 3.0, "hard": 2.2},
    )
    path_inches = path_px / 300.0
    assert path_inches >= 0.50, (
        f"easy paths must clear 0.50 in for crayons; got {path_inches:.2f} in"
    )


def test_maze_fits_inside_target_side() -> None:
    """For every difficulty + grid combo in the kids YAML, the maze fits the page."""
    for difficulty, grid in [
        ("easy", (10, 10)),
        ("medium", (12, 12)),
        ("hard", (15, 15)),
    ]:
        maze = _maze(grid_size=grid, difficulty=difficulty)  # type: ignore[arg-type]
        _, _, x_offsets, y_offsets, pad_x, pad_y = _layout(
            maze,
            target_side_px=2250,
            wall_thickness_px=12,
            path_wall_ratios={"easy": 4.5, "medium": 3.0, "hard": 2.2},
        )
        assert x_offsets[-1] + 2 * pad_x <= 2250 + 1
        assert y_offsets[-1] + 2 * pad_y <= 2250 + 1


def test_too_dense_maze_raises_clear_error() -> None:
    """A grid that can't fit at the wall_thickness_px floor must fail loudly."""
    # 30x30 cells at wall_floor=40 and ratio=4 needs 30*5+1 = 151 units;
    # 151 x 40 = 6040 px — way past a 2250 px canvas.
    maze = _maze(grid_size=(30, 30), difficulty="easy")
    with pytest.raises(ValueError, match=r"exceeds .* canvas"):
        _layout(
            maze,
            target_side_px=2250,
            wall_thickness_px=40,
            path_wall_ratios={"easy": 4.0, "medium": 2.5, "hard": 1.8},
        )


# --- solution renderer -----------------------------------------------------


def test_render_solution_adds_red_pixels_not_present_in_plain_maze() -> None:
    maze = _maze(grid_size=(15, 15), difficulty="medium")
    plain = np.array(render_maze(maze, target_side_px=900, wall_thickness_px=4))
    with_solution = np.array(render_solution(maze, target_side_px=900, wall_thickness_px=4))
    red_plain = ((plain[:, :, 0] == 229) & (plain[:, :, 1] == 57) & (plain[:, :, 2] == 53)).sum()
    red_solution = (
        (with_solution[:, :, 0] == 229)
        & (with_solution[:, :, 1] == 57)
        & (with_solution[:, :, 2] == 53)
    ).sum()
    assert red_solution > red_plain * 5


def test_render_solution_returns_same_size_as_plain_maze() -> None:
    maze = _maze()
    plain = render_maze(maze, target_side_px=1200)
    sol = render_solution(maze, target_side_px=1200)
    assert plain.size == sol.size


def test_render_solution_passes_through_passage_cells_only() -> None:
    """The drawn red line must touch every solution cell's centre."""
    maze = _maze(grid_size=(10, 10), seed=11, difficulty="easy")
    sol = np.array(render_solution(maze, target_side_px=900, wall_thickness_px=4))
    _, _, x_offsets, y_offsets, pad_x, pad_y = _layout(
        maze, target_side_px=900, wall_thickness_px=4
    )
    for r, c in maze.solution:
        x = pad_x + (x_offsets[c] + x_offsets[c + 1]) // 2
        y = pad_y + (y_offsets[r] + y_offsets[r + 1]) // 2
        assert tuple(int(v) for v in sol[y, x]) == (229, 57, 53)
