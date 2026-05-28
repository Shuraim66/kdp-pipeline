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


# --- START / FINISH openings in the outer wall ---------------------------


def _cell_interior_is_white(arr: np.ndarray, cell_box: tuple[int, int, int, int]) -> bool:
    """True if every pixel inside the cell box (excluding 1-px shared edges) is white.

    PIL's draw.rectangle paints inclusive on both ends, so adjacent walls own
    the boundary pixels of every cell box. We sample the interior to verify
    the cell itself is unpainted — that's the property "this cell is an
    opening, not a wall".
    """
    x0, y0, x1, y1 = cell_box
    if (x1 - x0) <= 2 or (y1 - y0) <= 2:
        # Cell too thin to have an interior — trust the layout.
        return True
    region = arr[y0 + 1 : y1 - 1, x0 + 1 : x1 - 1]
    return bool(((region[..., 0] == 255) & (region[..., 1] == 255) & (region[..., 2] == 255)).all())


def test_start_and_end_cells_are_openings_not_walls() -> None:
    """The renderer must skip painting the start/end perimeter cells so the
    path visibly exits the maze (otherwise the arrows point at solid wall).

    Mazelib records start/end as perimeter coordinates but does NOT carve
    the outer wall — left to itself, every perimeter slot stays opaque
    black after _draw_grid. This test pins the carving-by-skip fix.
    """
    maze = _maze(grid_size=(10, 10))
    img = render_maze(maze, target_side_px=2400)
    arr = np.array(img)
    _, _, x_offsets, y_offsets, pad_x, pad_y = _layout(maze, target_side_px=2400)

    for cell in (maze.start, maze.end):
        r, c = cell
        x0 = pad_x + x_offsets[c]
        y0 = pad_y + y_offsets[r]
        x1 = pad_x + x_offsets[c + 1]
        y1 = pad_y + y_offsets[r + 1]
        assert _cell_interior_is_white(arr, (x0, y0, x1, y1)), (
            f"perimeter cell {cell} should be an opening (white interior), "
            "but its interior contains painted pixels (likely still a wall)"
        )


def test_opening_aligns_with_first_interior_passage() -> None:
    """The cell one step inward from each opening must be a passage too.

    Mazelib always places outer entrances at a perimeter slot adjacent
    to a passage cell — verify that invariant holds so the opening
    leads into a traversable path, not a wall.
    """
    maze = _maze(grid_size=(10, 10))
    for cell in (maze.start, maze.end):
        r, c = cell
        # Step one cell INWARD from the perimeter (decrement whichever
        # axis sits on the boundary).
        if r == 0:
            ir, ic = 1, c
        elif r == maze.height - 1:
            ir, ic = maze.height - 2, c
        elif c == 0:
            ir, ic = r, 1
        elif c == maze.width - 1:
            ir, ic = r, maze.width - 2
        else:  # pragma: no cover - mazelib outer entrances are always perimeter
            pytest.fail(f"start/end cell {cell} not on perimeter")
        assert maze.grid[ir][ic] == 0, (
            f"opening at {cell} leads into {(ir, ic)} which is a wall (value 1) — "
            "the gap would not connect to any traversable path"
        )


def test_solution_dashed_path_reaches_each_opening() -> None:
    """The dashed trail must visibly approach / exit each opening.

    The line's exact pixel state at the perimeter cell centre depends on
    the cumulative dash/gap phase, so it can land in a gap and leave the
    cell centre white. What matters visually is that the dashed trail
    reaches the opening — sample a window large enough to catch at
    least one full dash+gap cycle (~160 px) on either side of the cell.
    """
    maze = _maze(grid_size=(10, 10))
    sol = np.array(render_solution(maze, target_side_px=2400))
    _, _, x_offsets, y_offsets, pad_x, pad_y = _layout(maze, target_side_px=2400)

    for cell in (maze.start, maze.end):
        r, c = cell
        cx = pad_x + (x_offsets[c] + x_offsets[c + 1]) // 2
        cy = pad_y + (y_offsets[r] + y_offsets[r + 1]) // 2
        window = sol[max(0, cy - 100) : cy + 101, max(0, cx - 100) : cx + 101]
        has_black = ((window[..., 0] == 0) & (window[..., 1] == 0) & (window[..., 2] == 0)).any()
        assert has_black, (
            f"no black pixels within 100 px of perimeter cell {cell} — "
            "dashed path may not be reaching the opening at all"
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
