"""Tests for the B&W-safe PIL maze renderer.

KDP interiors print in B&W only — coloured markers turn into indistinguishable
grays. The renderer therefore uses bold black text + arrow labels for
START / FINISH and a black dashed line (NOT a coloured line) for the
solution overlay. These tests pin both: no coloured pixels anywhere, the
label band carries ink near each entrance, and the solution adds black
pixels inside passages (not on top of walls).
"""

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
    LABEL_BAND_PX,
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


def _has_pixel(arr: np.ndarray, rgb: tuple[int, int, int]) -> bool:
    """True if any pixel in `arr` exactly matches `rgb`."""
    r, g, b = rgb
    return bool(((arr[:, :, 0] == r) & (arr[:, :, 1] == g) & (arr[:, :, 2] == b)).any())


# --- shape + colour pinning ----------------------------------------------


def test_render_maze_returns_square_image_at_requested_size() -> None:
    img = render_maze(_maze(), target_side_px=2400)
    assert isinstance(img, Image.Image)
    assert img.size == (2400, 2400)
    assert img.mode == "RGB"


def test_render_maze_paints_walls_black_and_passages_white() -> None:
    img = render_maze(_maze(), target_side_px=2400)
    arr = np.array(img)
    assert _has_pixel(arr, (0, 0, 0))
    assert _has_pixel(arr, (255, 255, 255))


def test_render_maze_uses_only_pure_black_and_white_ink() -> None:
    """No green / red / coloured pixels anywhere — KDP interior is B&W only."""
    img = render_maze(_maze(grid_size=(10, 10)), target_side_px=2400)
    arr = np.array(img)
    forbidden = [
        (76, 175, 80),  # the old green START fill
        (229, 57, 53),  # the old red FINISH fill / solution line
    ]
    for rgb in forbidden:
        assert not _has_pixel(arr, rgb), f"Forbidden colour {rgb} found in B&W render"


def test_render_solution_uses_only_pure_black_and_white_ink() -> None:
    """Solution overlay must stay pure black — no red trail."""
    img = render_solution(_maze(grid_size=(10, 10)), target_side_px=2400)
    arr = np.array(img)
    for rgb in [(76, 175, 80), (229, 57, 53)]:
        assert not _has_pixel(arr, rgb), f"Forbidden colour {rgb} found in solution render"


def test_render_maze_rejects_non_positive_target_side() -> None:
    with pytest.raises(ValueError, match="target_side_px"):
        render_maze(_maze(), target_side_px=0)


def test_render_maze_rejects_target_too_small_for_label_band() -> None:
    """target_side_px must leave at least 2*LABEL_BAND_PX for the maze area."""
    with pytest.raises(ValueError, match="label band"):
        render_maze(_maze(), target_side_px=LABEL_BAND_PX)


# --- defaults stay in sync with the schema --------------------------------


def test_renderer_defaults_match_schema_defaults() -> None:
    """If schema's PuzzleRenderSpec defaults drift, the renderer's must follow."""
    spec = PuzzleRenderSpec()
    assert spec.path_wall_ratios == DEFAULT_PATH_WALL_RATIOS
    assert spec.wall_thickness_px == DEFAULT_WALL_THICKNESS_PX


# --- START / FINISH labels render in the label band ----------------------


def _label_band_has_ink(arr: np.ndarray, side: str) -> bool:
    """True if any black pixel sits in the `side` label band of the canvas."""
    if side == "top":
        band = arr[:LABEL_BAND_PX, :, :]
    elif side == "bottom":
        band = arr[-LABEL_BAND_PX:, :, :]
    elif side == "left":
        band = arr[:, :LABEL_BAND_PX, :]
    elif side == "right":
        band = arr[:, -LABEL_BAND_PX:, :]
    else:
        raise ValueError(side)
    return bool(((band[:, :, 0] == 0) & (band[:, :, 1] == 0) & (band[:, :, 2] == 0)).any())


def test_render_maze_paints_labels_in_perimeter_bands() -> None:
    """The bands on the start- and end-side edges must contain black ink (text + arrow)."""
    maze = _maze(grid_size=(10, 10))
    img = render_maze(maze, target_side_px=2400)
    arr = np.array(img)
    # Compute which sides the entrances sit on so the test isn't seed-coupled.
    sides = set()
    for cell in (maze.start, maze.end):
        r, c = cell
        if r == 0:
            sides.add("top")
        elif r == maze.height - 1:
            sides.add("bottom")
        elif c == 0:
            sides.add("left")
        elif c == maze.width - 1:
            sides.add("right")
    assert sides, "every maze has at least one perimeter entrance"
    for side in sides:
        assert _label_band_has_ink(arr, side), (
            f"label band on {side} side has no ink — START/FINISH label missing"
        )


# --- difficulty-aware ratios still honoured ------------------------------


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
    maze = _maze(grid_size=grid_size, difficulty=difficulty)
    wall_px, path_px, *_ = _layout(
        maze,
        target_side_px=2400,
        wall_thickness_px=12,
        path_wall_ratios={"easy": 4.5, "medium": 3.0, "hard": 2.2},
    )
    expected = int(ratio * wall_px)
    assert path_px == expected
    assert path_px >= wall_px


def test_default_ratios_produce_chunkier_easy_than_hard() -> None:
    easy = _maze(grid_size=(10, 10), difficulty="easy")
    hard = _maze(grid_size=(20, 20), difficulty="hard")
    _, easy_path, *_ = _layout(easy, target_side_px=2400)
    _, hard_path, *_ = _layout(hard, target_side_px=2400)
    assert easy_path > hard_path


def test_easy_yaml_ratios_clear_crayon_threshold() -> None:
    """Kids YAML easy paths must stay >= 0.50 in for crayons at 300 DPI (after label band)."""
    maze = _maze(grid_size=(10, 10), difficulty="easy")
    _, path_px, *_ = _layout(
        maze,
        target_side_px=2400,
        wall_thickness_px=12,
        path_wall_ratios={"easy": 4.5, "medium": 3.0, "hard": 2.2},
    )
    path_inches = path_px / 300.0
    assert path_inches >= 0.50, (
        f"easy paths must clear 0.50 in for crayons; got {path_inches:.2f} in"
    )


def test_maze_fits_inside_label_band_envelope() -> None:
    """Every difficulty + grid combo in the kids YAML fits inside maze_target."""
    maze_target = 2400 - 2 * LABEL_BAND_PX
    for difficulty, grid in [
        ("easy", (10, 10)),
        ("medium", (12, 12)),
        ("hard", (15, 15)),
    ]:
        maze = _maze(grid_size=grid, difficulty=difficulty)  # type: ignore[arg-type]
        _, _, x_offsets, y_offsets, *_ = _layout(
            maze,
            target_side_px=2400,
            wall_thickness_px=12,
            path_wall_ratios={"easy": 4.5, "medium": 3.0, "hard": 2.2},
        )
        assert x_offsets[-1] <= maze_target
        assert y_offsets[-1] <= maze_target


def test_too_dense_maze_raises_clear_error() -> None:
    """A grid that can't fit even with the wall floor must fail loudly."""
    maze = _maze(grid_size=(30, 30), difficulty="easy")
    with pytest.raises(ValueError, match=r"exceeds .* maze area"):
        _layout(
            maze,
            target_side_px=2400,
            wall_thickness_px=40,
            path_wall_ratios={"easy": 4.0, "medium": 2.5, "hard": 1.8},
        )


# --- solution renderer (B&W dashed) --------------------------------------


def test_render_solution_returns_same_size_as_plain_maze() -> None:
    maze = _maze()
    plain = render_maze(maze, target_side_px=2400)
    sol = render_solution(maze, target_side_px=2400)
    assert plain.size == sol.size


def test_render_solution_adds_black_pixels_inside_passages() -> None:
    """The dashed path paints black inside passages — pixels that were white in the plain render."""
    maze = _maze(grid_size=(15, 15), difficulty="medium")
    plain = np.array(render_maze(maze, target_side_px=2400))
    sol = np.array(render_solution(maze, target_side_px=2400))
    passage_mask = (plain == 255).all(axis=-1)  # plain pixels that were pure white
    sol_black_mask = (sol == 0).all(axis=-1)
    new_black_in_passages = passage_mask & sol_black_mask
    # The dashed solution must paint a substantial number of NEW black pixels
    # in passage cells — empirically thousands for a 15x15 maze at 2400 px.
    assert new_black_in_passages.sum() > 1000, (
        f"solution overlay added only {int(new_black_in_passages.sum())} new black "
        "pixels in passages — dashed path may not be rendering"
    )


def test_render_solution_includes_gaps_along_the_path() -> None:
    """The path must be DASHED, not solid — some pixels on the path are still white."""
    maze = _maze(grid_size=(15, 15), difficulty="medium")
    sol = np.array(render_solution(maze, target_side_px=2400))
    _, _, x_offsets, y_offsets, pad_x, pad_y = _layout(maze, target_side_px=2400)

    # Sample the pixels along the solution-cell centres. With a dashed line,
    # roughly half (within tolerance) of those centres should be ON a dash
    # and the other half on a gap.
    white_hits = 0
    black_hits = 0
    for r, c in maze.solution:
        cx = pad_x + (x_offsets[c] + x_offsets[c + 1]) // 2
        cy = pad_y + (y_offsets[r] + y_offsets[r + 1]) // 2
        pixel = tuple(int(v) for v in sol[cy, cx])
        if pixel == (255, 255, 255):
            white_hits += 1
        elif pixel == (0, 0, 0):
            black_hits += 1
    # Both states should appear — that's the regression check for "no longer solid".
    assert white_hits > 0, "every solution cell centre is inked — line is solid, not dashed"
    assert black_hits > 0, "no solution cell centre is inked — line is missing entirely"
