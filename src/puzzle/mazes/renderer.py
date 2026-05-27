"""PIL renderer for generated mazes — print-ready at 300 DPI.

The maze occupies a square region whose side equals the smaller printable
dimension. Walls render as solid black squares, passages as white; the
start cell is filled green and the end cell red so kids can tell which is
which without external labels. A second function renders the same maze
with the solution path overlaid as a thick red line through cell centres.
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

# Solution-path stroke width is proportional to the cell size; ~25 % gives a
# clearly-visible trail without burying the maze walls.
_SOLUTION_STROKE_RATIO = 0.25
_MIN_SOLUTION_STROKE = 4


def _cell_pixels(maze: GeneratedMaze, *, target_side_px: int) -> tuple[int, int, int]:
    """Compute (cell_px, maze_side_px, pad_px) for centring the maze on the page.

    Each grid cell becomes a `cell_px` square. Cell size is the largest
    integer that lets the full grid fit inside ``target_side_px``; whatever
    remains is split evenly as left/right (and top/bottom) padding so the
    rendered maze is centred.
    """
    grid_side = max(maze.height, maze.width)
    cell_px = max(1, target_side_px // grid_side)
    maze_side = cell_px * grid_side
    pad_px = (target_side_px - maze_side) // 2
    return cell_px, maze_side, pad_px


def _draw_grid(
    draw: ImageDraw.ImageDraw,
    maze: GeneratedMaze,
    *,
    cell_px: int,
    origin_x: int,
    origin_y: int,
) -> None:
    """Paint walls black and the start/end cells in their accent colours."""
    for r, row in enumerate(maze.grid):
        for c, cell in enumerate(row):
            if cell != 1:
                continue
            x0 = origin_x + c * cell_px
            y0 = origin_y + r * cell_px
            draw.rectangle((x0, y0, x0 + cell_px, y0 + cell_px), fill=_WALL)
    for (r, c), colour in (
        (maze.start, _START),
        (maze.end, _FINISH),
    ):
        x0 = origin_x + c * cell_px
        y0 = origin_y + r * cell_px
        draw.rectangle((x0, y0, x0 + cell_px, y0 + cell_px), fill=colour)


def render_maze(maze: GeneratedMaze, *, target_side_px: int = 2250) -> Image.Image:
    """Render one maze as a print-ready PIL image with no solution overlay.

    Default ``target_side_px=2250`` matches the 7.5 in usable area inside an
    8.5x11 trim with a 0.5 in margin, at 300 DPI. The caller is free to pass
    a different page size — square 8.5x8.5 books would use 2400 (8 in x 300).
    """
    if target_side_px <= 0:
        raise ValueError(f"target_side_px must be positive, got {target_side_px}")
    cell_px, _, pad_px = _cell_pixels(maze, target_side_px=target_side_px)
    image = Image.new("RGB", (target_side_px, target_side_px), _PASSAGE)
    draw = ImageDraw.Draw(image)
    _draw_grid(draw, maze, cell_px=cell_px, origin_x=pad_px, origin_y=pad_px)
    return image


def render_solution(maze: GeneratedMaze, *, target_side_px: int = 2250) -> Image.Image:
    """Render the same maze with the solution path overlaid in red."""
    image = render_maze(maze, target_side_px=target_side_px)
    if not maze.solution:
        return image
    cell_px, _, pad_px = _cell_pixels(maze, target_side_px=target_side_px)
    draw = ImageDraw.Draw(image)
    stroke = max(_MIN_SOLUTION_STROKE, int(cell_px * _SOLUTION_STROKE_RATIO))
    half = cell_px // 2

    def _cell_centre(cell: tuple[int, int]) -> tuple[int, int]:
        r, c = cell
        return pad_px + c * cell_px + half, pad_px + r * cell_px + half

    # Draw start → first solution cell → ... → last solution cell → end so
    # the path runs perimeter-to-perimeter and not just between the inner
    # cells (mazelib's solution excludes the perimeter cells themselves).
    points = [
        _cell_centre(maze.start),
        *(_cell_centre(cell) for cell in maze.solution),
        _cell_centre(maze.end),
    ]
    draw.line(points, fill=_SOLUTION, width=stroke, joint="curve")
    return image
