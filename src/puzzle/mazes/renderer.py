"""PIL renderer for generated mazes — print-ready at 300 DPI.

The maze occupies a square region whose side equals the smaller printable
dimension. Walls render as solid black squares, passages as white; the
start cell is filled green and the end cell red so kids can tell which is
which without external labels. A second function renders the same maze
with the solution path overlaid as a thick red line through cell centres.

Path / wall thickness is **configurable per niche** via
``config.body.puzzle.render`` (see ``PuzzleRenderSpec`` in
``src/config/schema.py``). The renderer never hard-codes ratios — it
takes them as arguments so a kids' book and an adult book at the same
"easy" label can render very differently. ``wall_thickness_px`` is a
**floor**: the renderer fills the page as much as it can while keeping
walls at least that wide, so a YAML floor of 12 px is harmless on a
normal 8.5x11 page (the page-filling computation lands around 40 px)
but prevents hair-thin walls when a grid is too dense for the canvas.
"""

from __future__ import annotations

from PIL import Image, ImageDraw

from src.puzzle.mazes.models import GeneratedMaze

# Print-ready colours. Pure RGB tuples — no transparency.
_WALL = (0, 0, 0)
_PASSAGE = (255, 255, 255)
_START = (76, 175, 80)  # friendly green
_FINISH = (229, 57, 53)  # warm red
_SOLUTION = (229, 57, 53)  # same red as finish — visually consistent

# Fallback defaults — used when no render spec is passed. Kept in sync with
# the schema's PuzzleRenderSpec defaults; tests assert the two agree.
DEFAULT_PATH_WALL_RATIOS: dict[str, float] = {"easy": 4.0, "medium": 2.5, "hard": 1.8}
DEFAULT_WALL_THICKNESS_PX = 12

# Solution-stroke width is tied to wall thickness so the line reads at a
# consistent visual weight across all difficulties.
_SOLUTION_STROKE_TO_WALL_RATIO = 1.0
_MIN_SOLUTION_STROKE = 4


def _resolve_ratio(maze: GeneratedMaze, path_wall_ratios: dict[str, float] | None) -> float:
    """Pick the path:wall ratio for this maze, falling back to defaults."""
    ratios = path_wall_ratios if path_wall_ratios is not None else DEFAULT_PATH_WALL_RATIOS
    if maze.difficulty in ratios:
        return ratios[maze.difficulty]
    # Last-ditch: try the schema-default medium so we never crash with KeyError.
    return ratios.get("medium", DEFAULT_PATH_WALL_RATIOS["medium"])


def _build_offsets(cells: int, wall_px: int, path_px: int) -> list[int]:
    """Pixel start positions of every grid line on one axis (length 2*cells + 2)."""
    offsets = [0]
    for i in range(2 * cells + 1):
        width = path_px if i % 2 == 1 else wall_px
        offsets.append(offsets[-1] + width)
    return offsets


def _layout(
    maze: GeneratedMaze,
    *,
    target_side_px: int,
    wall_thickness_px: int = DEFAULT_WALL_THICKNESS_PX,
    path_wall_ratios: dict[str, float] | None = None,
) -> tuple[int, int, list[int], list[int], int, int]:
    """Compute (wall_px, path_px, x_offsets, y_offsets, pad_x, pad_y) for a maze.

    Algorithm: pick the largest wall_px <= the page-filling value such that
    the resulting maze fits inside ``target_side_px``; if that wall_px would
    drop below ``wall_thickness_px``, fall back to the floor (and raise if
    even the floor doesn't fit). path_px is ``int(ratio * wall_px)`` — floor
    rather than round so we never overshoot the canvas by accident.
    """
    cells_x = (maze.width - 1) // 2
    cells_y = (maze.height - 1) // 2
    ratio = _resolve_ratio(maze, path_wall_ratios)
    sizing_cells = max(cells_x, cells_y)

    # Start from the page-filling estimate; decrement until the integer-rounded
    # total actually fits the target (path_px = int(ratio * wall) can overshoot
    # by ~cells px due to rounding when ratio isn't an integer).
    wall_px = max(1, int(target_side_px / (sizing_cells * (ratio + 1.0) + 1.0)))
    while wall_px >= wall_thickness_px:
        path_px = max(1, int(ratio * wall_px))
        total = (sizing_cells + 1) * wall_px + sizing_cells * path_px
        if total <= target_side_px:
            break
        wall_px -= 1
    else:
        # Loop exhausted: wall_px dropped below the floor. Use the floor and
        # accept the possible overflow — surface it with a clear error.
        wall_px = wall_thickness_px
        path_px = max(1, int(ratio * wall_px))
        total = (sizing_cells + 1) * wall_px + sizing_cells * path_px
        if total > target_side_px:
            raise ValueError(
                f"Maze ({maze.difficulty}, {sizing_cells}x{sizing_cells} cells, "
                f"ratio={ratio}, wall_floor={wall_thickness_px}px) renders at "
                f"{total}px — exceeds {target_side_px}px canvas. Reduce "
                "puzzle.grid_sizes, lower puzzle.render.wall_thickness_px, or "
                "lower puzzle.render.path_wall_ratios in the niche YAML."
            )

    x_offsets = _build_offsets(cells_x, wall_px, path_px)
    y_offsets = _build_offsets(cells_y, wall_px, path_px)
    pad_x = (target_side_px - x_offsets[-1]) // 2
    pad_y = (target_side_px - y_offsets[-1]) // 2
    return wall_px, path_px, x_offsets, y_offsets, pad_x, pad_y


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


def render_maze(
    maze: GeneratedMaze,
    *,
    target_side_px: int = 2400,
    wall_thickness_px: int = DEFAULT_WALL_THICKNESS_PX,
    path_wall_ratios: dict[str, float] | None = None,
) -> Image.Image:
    """Render one maze as a print-ready PIL image with no solution overlay.

    Default ``target_side_px=2400`` matches the 8 in image-safe area inside
    an 8.5x11 trim with a 0.25 in margin, at 300 DPI — same constant the
    puzzle book builder uses to place the maze on the page, so the printed
    DPI is exactly 300 (not 281, which is what 2250 px stretched into 8 in
    would print at). ``wall_thickness_px`` and ``path_wall_ratios`` come
    from ``config.body.puzzle.render`` for a real book; defaults match
    ``PuzzleRenderSpec``'s defaults so direct callers (tests, ad-hoc
    previews) get sensible output.
    """
    if target_side_px <= 0:
        raise ValueError(f"target_side_px must be positive, got {target_side_px}")
    _, _, x_offsets, y_offsets, pad_x, pad_y = _layout(
        maze,
        target_side_px=target_side_px,
        wall_thickness_px=wall_thickness_px,
        path_wall_ratios=path_wall_ratios,
    )
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


def render_solution(
    maze: GeneratedMaze,
    *,
    target_side_px: int = 2400,
    wall_thickness_px: int = DEFAULT_WALL_THICKNESS_PX,
    path_wall_ratios: dict[str, float] | None = None,
) -> Image.Image:
    """Render the same maze with the solution path overlaid in red."""
    image = render_maze(
        maze,
        target_side_px=target_side_px,
        wall_thickness_px=wall_thickness_px,
        path_wall_ratios=path_wall_ratios,
    )
    if not maze.solution:
        return image
    wall_px, _, x_offsets, y_offsets, pad_x, pad_y = _layout(
        maze,
        target_side_px=target_side_px,
        wall_thickness_px=wall_thickness_px,
        path_wall_ratios=path_wall_ratios,
    )
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
