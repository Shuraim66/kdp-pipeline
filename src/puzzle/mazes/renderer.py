"""PIL renderer for generated mazes — print-ready at 300 DPI.

The maze occupies a square region whose side equals the smaller printable
dimension. Walls render as solid black squares, passages as white; the
start cell is filled green and the end cell red so kids can tell which is
which without external labels. A second function renders the same maze
with the solution path overlaid as a thick red line through cell centres.

Path / wall thickness is difficulty-aware (`_PATH_TO_WALL_RATIO`): easy
mazes get chunky, crayon-friendly paths (path = 4x wall thickness);
hard mazes pack tighter (path = 1.5x wall) for adult-style density.
"""

from __future__ import annotations

from PIL import Image, ImageDraw

from src.puzzle.mazes.models import Difficulty, GeneratedMaze

# Print-ready colours. Pure RGB tuples — no transparency.
_WALL = (0, 0, 0)
_PASSAGE = (255, 255, 255)
_START = (76, 175, 80)  # friendly green
_FINISH = (229, 57, 53)  # warm red
_SOLUTION = (229, 57, 53)  # same red as finish — visually consistent

# Path width relative to wall thickness. Easy mazes need wide paths a
# crayon can fit inside; hard mazes pack the page with tighter passages.
# These ratios are visual choices, not load-bearing — change them freely.
_PATH_TO_WALL_RATIO: dict[Difficulty, float] = {
    "easy": 4.0,
    "medium": 2.5,
    "hard": 1.5,
}

# Solution-stroke width is tied to wall thickness so the line reads at a
# consistent visual weight across all difficulties.
_SOLUTION_STROKE_TO_WALL_RATIO = 1.0
_MIN_SOLUTION_STROKE = 4


def _axis_layout(cells: int, ratio: float, target_side_px: int) -> tuple[int, int, list[int]]:
    """Return (wall_px, path_px, offsets) for one axis of a `cells`-cell maze.

    A mazelib grid is ``2 * cells + 1`` units wide on each axis: ``cells + 1``
    wall slots interleaved with ``cells`` path slots. With path = ratio * wall,
    the axis pixel side is ``wall * (cells * (ratio + 1) + 1)``. We pick the
    largest integer wall_px that fits inside ``target_side_px`` and let
    path_px = round(ratio * wall_px). The returned ``offsets`` table holds
    the pixel start position of every grid line on this axis — index it by
    grid coordinate to size or place any cell rectangle.
    """
    denom = cells * (ratio + 1.0) + 1.0
    wall_px = max(1, int(target_side_px / denom))
    path_px = max(1, round(ratio * wall_px))
    offsets = [0]
    for i in range(2 * cells + 1):
        width = path_px if i % 2 == 1 else wall_px
        offsets.append(offsets[-1] + width)
    return wall_px, path_px, offsets


def _layout(
    maze: GeneratedMaze, *, target_side_px: int
) -> tuple[int, int, list[int], list[int], int, int]:
    """Compute (wall_px, path_px, x_offsets, y_offsets, pad_x, pad_y) for a maze.

    Both axes share the same wall/path sizing — derived from the longer
    grid dimension so the maze always fits. Square mazes (our config) have
    identical x and y offset tables; rectangular mazes get a centred,
    aspect-correct layout.
    """
    cells_x = (maze.width - 1) // 2
    cells_y = (maze.height - 1) // 2
    ratio = _PATH_TO_WALL_RATIO.get(maze.difficulty, _PATH_TO_WALL_RATIO["medium"])
    # Drive sizing from the larger axis so the maze always fits target_side_px.
    sizing_cells = max(cells_x, cells_y)
    wall_px, path_px, _ = _axis_layout(sizing_cells, ratio, target_side_px)
    # Rebuild per-axis offsets with the chosen wall_px / path_px.
    _, _, x_offsets = _axis_layout_with_units(cells_x, wall_px, path_px)
    _, _, y_offsets = _axis_layout_with_units(cells_y, wall_px, path_px)
    pad_x = (target_side_px - x_offsets[-1]) // 2
    pad_y = (target_side_px - y_offsets[-1]) // 2
    return wall_px, path_px, x_offsets, y_offsets, pad_x, pad_y


def _axis_layout_with_units(cells: int, wall_px: int, path_px: int) -> tuple[int, int, list[int]]:
    """Build offsets for one axis with a pre-chosen wall_px / path_px pair."""
    offsets = [0]
    for i in range(2 * cells + 1):
        width = path_px if i % 2 == 1 else wall_px
        offsets.append(offsets[-1] + width)
    return wall_px, path_px, offsets


def _cell_box(
    r: int, c: int, x_offsets: list[int], y_offsets: list[int], pad_x: int, pad_y: int
) -> tuple[int, int, int, int]:
    """Pixel bounding box of grid cell (r, c) in the rendered image."""
    return (
        pad_x + x_offsets[c],
        pad_y + y_offsets[r],
        pad_x + x_offsets[c + 1],
        pad_y + y_offsets[r + 1],
    )


def _cell_centre(
    r: int, c: int, x_offsets: list[int], y_offsets: list[int], pad_x: int, pad_y: int
) -> tuple[int, int]:
    """Pixel coordinates of the centre of grid cell (r, c)."""
    x0, y0, x1, y1 = _cell_box(r, c, x_offsets, y_offsets, pad_x, pad_y)
    return (x0 + x1) // 2, (y0 + y1) // 2


def _draw_grid(
    draw: ImageDraw.ImageDraw,
    maze: GeneratedMaze,
    *,
    x_offsets: list[int],
    y_offsets: list[int],
    pad_x: int,
    pad_y: int,
) -> None:
    """Paint walls black and the start/end cells in their accent colours."""
    for r, row in enumerate(maze.grid):
        for c, cell in enumerate(row):
            if cell != 1:
                continue
            draw.rectangle(_cell_box(r, c, x_offsets, y_offsets, pad_x, pad_y), fill=_WALL)
    for (r, c), colour in (
        (maze.start, _START),
        (maze.end, _FINISH),
    ):
        draw.rectangle(_cell_box(r, c, x_offsets, y_offsets, pad_x, pad_y), fill=colour)


def render_maze(maze: GeneratedMaze, *, target_side_px: int = 2250) -> Image.Image:
    """Render one maze as a print-ready PIL image with no solution overlay.

    Default ``target_side_px=2250`` matches the 7.5 in usable area inside an
    8.5x11 trim with a 0.5 in margin, at 300 DPI. The caller is free to pass
    a different page size — square 8.5x8.5 books would use 2400 (8 in x 300).
    """
    if target_side_px <= 0:
        raise ValueError(f"target_side_px must be positive, got {target_side_px}")
    _, _, x_offsets, y_offsets, pad_x, pad_y = _layout(maze, target_side_px=target_side_px)
    image = Image.new("RGB", (target_side_px, target_side_px), _PASSAGE)
    draw = ImageDraw.Draw(image)
    _draw_grid(
        draw,
        maze,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        pad_x=pad_x,
        pad_y=pad_y,
    )
    return image


def render_solution(maze: GeneratedMaze, *, target_side_px: int = 2250) -> Image.Image:
    """Render the same maze with the solution path overlaid in red."""
    image = render_maze(maze, target_side_px=target_side_px)
    if not maze.solution:
        return image
    wall_px, _, x_offsets, y_offsets, pad_x, pad_y = _layout(maze, target_side_px=target_side_px)
    draw = ImageDraw.Draw(image)
    stroke = max(_MIN_SOLUTION_STROKE, int(wall_px * _SOLUTION_STROKE_TO_WALL_RATIO))

    def _centre(cell: tuple[int, int]) -> tuple[int, int]:
        r, c = cell
        return _cell_centre(r, c, x_offsets, y_offsets, pad_x, pad_y)

    # Draw start → first solution cell → ... → last solution cell → end so
    # the path runs perimeter-to-perimeter and not just between the inner
    # cells (mazelib's solution excludes the perimeter cells themselves).
    points = [
        _centre(maze.start),
        *(_centre(cell) for cell in maze.solution),
        _centre(maze.end),
    ]
    draw.line(points, fill=_SOLUTION, width=stroke, joint="curve")
    return image
