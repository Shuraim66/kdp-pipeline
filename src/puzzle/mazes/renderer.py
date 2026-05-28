"""PIL renderer for generated mazes — print-ready, B&W-safe at 300 DPI.

Pure black + white. KDP interiors print in B&W only, so coloured START /
FINISH cells would reproduce as indistinguishable mid-grays — a kid could
not tell the entrance from the exit, and a red solution path would blend
into the black walls. Instead:

* START / FINISH are bold black TEXT labels with arrows pointing into
  the entrance cell — placed in a label band reserved around the maze.
* The solution path is a thick BLACK DASHED line through cell centres,
  thinner than walls and with visible gaps so it never reads as a wall.

Path / wall thickness is configurable per niche via
``config.body.puzzle.render`` (see ``PuzzleRenderSpec`` in
``src/config/schema.py``). ``wall_thickness_px`` is a FLOOR — the renderer
fills the page as much as the canvas allows while keeping walls at least
that wide; on a normal 8.5x11 page the page-filling computation lands
well above the floor.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import pairwise
from pathlib import Path

import reportlab
from PIL import Image, ImageDraw, ImageFont

from src.puzzle.mazes.models import GeneratedMaze

# Print-ready colours. Pure RGB tuples — no transparency, no grays, no hues.
_WALL = (0, 0, 0)
_PASSAGE = (255, 255, 255)
_INK = (0, 0, 0)  # text, arrows, solution dashes — all the same black

# Pixel band reserved on each side of the maze for the START / FINISH
# labels. At a 2400 target placed at 8 in (300 DPI), 180 px = 0.6 in —
# enough room for a bold "START" + arrow without crowding the maze.
LABEL_BAND_PX = 180

# Fallback defaults — used when no render spec is passed. Kept in sync
# with the schema's PuzzleRenderSpec defaults; tests assert they agree.
DEFAULT_PATH_WALL_RATIOS: dict[str, float] = {"easy": 4.0, "medium": 2.5, "hard": 1.8}
DEFAULT_WALL_THICKNESS_PX = 12

# Solution-overlay dash pattern. Stroke thinner than walls so a glance
# never confuses the trail for a wall. Dash and gap sized relative to
# wall_px so the rhythm scales with the maze.
_SOLUTION_STROKE_TO_WALL_RATIO = 0.55
_SOLUTION_DASH_PER_WALL = 2.5
_SOLUTION_GAP_PER_WALL = 1.5
_MIN_SOLUTION_STROKE = 4

# Label font + arrow scaling — relative to wall_px so labels grow with the
# maze but stay narrow enough that the L/R label bands (the constraining
# axis) can carry the text. Earlier 2.4x produced "START"/"FINISH" too wide
# to fit, so L/R labels got clipped to "T ▶" / "◄ F" — exactly the bug B&W
# labels were meant to avoid.
_LABEL_FONT_PER_WALL = 1.7
_MIN_LABEL_FONT_PX = 48
_ARROW_PER_WALL = 1.2
_MIN_ARROW_PX = 24

_FONT_DIR = Path(reportlab.__file__).resolve().parent / "fonts"
_FONT_BOLD_PATH = _FONT_DIR / "VeraBd.ttf"


@lru_cache(maxsize=16)
def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Load Vera Bold at `size` px — cached because rendering 80 mazes reuses sizes."""
    return ImageFont.truetype(str(_FONT_BOLD_PATH), size)


def _resolve_ratio(maze: GeneratedMaze, path_wall_ratios: dict[str, float] | None) -> float:
    """Pick the path:wall ratio for this maze, falling back to defaults."""
    ratios = path_wall_ratios if path_wall_ratios is not None else DEFAULT_PATH_WALL_RATIOS
    if maze.difficulty in ratios:
        return ratios[maze.difficulty]
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

    The maze must fit inside ``target_side_px - 2 * LABEL_BAND_PX`` on each
    axis; the remaining target_side_px area on each side is the label band
    where START / FINISH text + arrows render. Walls stay >= wall_thickness_px
    px wide (a floor — page-filling math lands well above it on normal trims).
    """
    cells_x = (maze.width - 1) // 2
    cells_y = (maze.height - 1) // 2
    ratio = _resolve_ratio(maze, path_wall_ratios)
    sizing_cells = max(cells_x, cells_y)

    maze_target = target_side_px - 2 * LABEL_BAND_PX
    if maze_target <= 0:
        raise ValueError(
            f"target_side_px={target_side_px} too small after reserving "
            f"{LABEL_BAND_PX}px label band on each side."
        )

    wall_px = max(1, int(maze_target / (sizing_cells * (ratio + 1.0) + 1.0)))
    while wall_px >= wall_thickness_px:
        path_px = max(1, int(ratio * wall_px))
        total = (sizing_cells + 1) * wall_px + sizing_cells * path_px
        if total <= maze_target:
            break
        wall_px -= 1
    else:
        wall_px = wall_thickness_px
        path_px = max(1, int(ratio * wall_px))
        total = (sizing_cells + 1) * wall_px + sizing_cells * path_px
        if total > maze_target:
            raise ValueError(
                f"Maze ({maze.difficulty}, {sizing_cells}x{sizing_cells} cells, "
                f"ratio={ratio}, wall_floor={wall_thickness_px}px) renders at "
                f"{total}px — exceeds {maze_target}px maze area "
                f"(target {target_side_px}px - 2x{LABEL_BAND_PX}px label band). "
                "Reduce puzzle.grid_sizes, lower puzzle.render.wall_thickness_px, "
                "or lower puzzle.render.path_wall_ratios in the niche YAML."
            )

    x_offsets = _build_offsets(cells_x, wall_px, path_px)
    y_offsets = _build_offsets(cells_y, wall_px, path_px)
    # Centre the maze inside the full canvas — label bands surround it.
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


def _perimeter_side(cell: tuple[int, int], height: int, width: int) -> str | None:
    """Return 'top'/'right'/'bottom'/'left' for a perimeter cell, else None."""
    r, c = cell
    if r == 0:
        return "top"
    if r == height - 1:
        return "bottom"
    if c == 0:
        return "left"
    if c == width - 1:
        return "right"
    return None


def _draw_perimeter_label(
    draw: ImageDraw.ImageDraw,
    label: str,
    cell_box: tuple[int, int, int, int],
    side: str,
    font: ImageFont.FreeTypeFont,
    gap_px: int,
    arrow_len: int,
) -> None:
    """Draw `label` text + arrow just outside `cell_box` on `side`.

    Top / bottom sides use horizontal text — the wide label bands have
    plenty of room. Left / right sides use CHARACTER-STACKED vertical
    text (one letter per line) because the narrow side bands can't fit
    "START" or "FINISH" horizontally without clipping to the canvas edge.
    The arrow's tip always touches the maze entrance; text sits beyond
    the arrow tail in the band.
    """
    x0, y0, x1, y1 = cell_box
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    arrow_half = max(8, arrow_len // 3)

    if side in ("top", "bottom"):
        text_bbox = draw.textbbox((0, 0), label, font=font, anchor="lt")
        text_w = int(text_bbox[2] - text_bbox[0])
        text_h = int(text_bbox[3] - text_bbox[1])
        if side == "top":
            arrow_tip_y = y0 - gap_px
            arrow_base_y = arrow_tip_y - arrow_len
            arrow_polygon = [
                (cx - arrow_half, arrow_base_y),
                (cx + arrow_half, arrow_base_y),
                (cx, arrow_tip_y),
            ]
            text_pos: tuple[int, int] = (cx - text_w // 2, arrow_base_y - gap_px - text_h)
        else:
            arrow_tip_y = y1 + gap_px
            arrow_base_y = arrow_tip_y + arrow_len
            arrow_polygon = [
                (cx - arrow_half, arrow_base_y),
                (cx + arrow_half, arrow_base_y),
                (cx, arrow_tip_y),
            ]
            text_pos = (cx - text_w // 2, arrow_base_y + gap_px)
        draw.polygon(arrow_polygon, fill=_INK)
        draw.text(text_pos, label, fill=_INK, font=font, anchor="lt")
        return

    # Char-stacked vertical text for left / right entrances.
    chars = list(label)
    char_widths = [int(draw.textbbox((0, 0), ch, font=font, anchor="lt")[2]) for ch in chars]
    max_char_w = max(char_widths) if char_widths else 0
    line_h = font.size + max(2, font.size // 16)
    total_text_h = line_h * len(chars)
    text_top = cy - total_text_h // 2

    if side == "left":
        arrow_tip_x = x0 - gap_px
        arrow_base_x = arrow_tip_x - arrow_len
        arrow_polygon = [
            (arrow_base_x, cy - arrow_half),
            (arrow_base_x, cy + arrow_half),
            (arrow_tip_x, cy),
        ]
        text_block_right = arrow_base_x - gap_px
        text_block_left = text_block_right - max_char_w
    elif side == "right":
        arrow_tip_x = x1 + gap_px
        arrow_base_x = arrow_tip_x + arrow_len
        arrow_polygon = [
            (arrow_base_x, cy - arrow_half),
            (arrow_base_x, cy + arrow_half),
            (arrow_tip_x, cy),
        ]
        text_block_left = arrow_base_x + gap_px
        text_block_right = text_block_left + max_char_w
    else:
        return

    draw.polygon(arrow_polygon, fill=_INK)
    char_center_x = (text_block_left + text_block_right) // 2
    for i, ch in enumerate(chars):
        cw = char_widths[i]
        draw.text(
            (char_center_x - cw // 2, text_top + i * line_h),
            ch,
            fill=_INK,
            font=font,
            anchor="lt",
        )


def _draw_grid(
    draw: ImageDraw.ImageDraw,
    maze: GeneratedMaze,
    *,
    x_offsets: list[int],
    y_offsets: list[int],
    pad_x: int,
    pad_y: int,
    wall_px: int,
) -> None:
    """Paint walls black and draw START / FINISH text+arrow labels in the bands.

    Mazelib records the start/end coordinates on the perimeter but does NOT
    carve the outer wall there — the cell is left as a wall in `maze.grid`.
    To make the path visibly exit the maze (so the START / FINISH arrows
    point through an actual opening and not at a solid wall), we skip
    painting those two perimeter cells. Start/end are always on the outer
    boundary and always align with a perimeter passage slot, so leaving
    the cell white opens a one-cell gap that connects directly to the
    first interior path cell.
    """
    start_cell = (maze.start[0], maze.start[1])
    end_cell = (maze.end[0], maze.end[1])
    for r, row in enumerate(maze.grid):
        for c, cell in enumerate(row):
            if cell != 1:
                continue
            if (r, c) == start_cell or (r, c) == end_cell:
                continue
            draw.rectangle(_cell_box(r, c, x_offsets, y_offsets, pad_x, pad_y), fill=_WALL)

    font_size = max(_MIN_LABEL_FONT_PX, int(wall_px * _LABEL_FONT_PER_WALL))
    font = _load_font(font_size)
    gap_px = max(6, wall_px // 2)
    arrow_len = max(_MIN_ARROW_PX, int(wall_px * _ARROW_PER_WALL))
    labels: tuple[tuple[str, tuple[int, int]], ...] = (
        ("START", maze.start),
        ("FINISH", maze.end),
    )
    for label, (cell_r, cell_c) in labels:
        side = _perimeter_side((cell_r, cell_c), maze.height, maze.width)
        if side is None:
            continue
        box = _cell_box(cell_r, cell_c, x_offsets, y_offsets, pad_x, pad_y)
        _draw_perimeter_label(draw, label, box, side, font, gap_px, arrow_len)


def render_maze(
    maze: GeneratedMaze,
    *,
    target_side_px: int = 2400,
    wall_thickness_px: int = DEFAULT_WALL_THICKNESS_PX,
    path_wall_ratios: dict[str, float] | None = None,
) -> Image.Image:
    """Render one maze as a print-ready B&W PIL image with text labels (no solution).

    Default ``target_side_px=2400`` matches the 8 in image-safe area inside
    an 8.5x11 trim with a 0.25 in margin, at 300 DPI. ``LABEL_BAND_PX``
    on each side is reserved for the START / FINISH labels. The maze
    itself occupies the centre area; effective print DPI stays at 300.
    """
    if target_side_px <= 0:
        raise ValueError(f"target_side_px must be positive, got {target_side_px}")
    wall_px, _, x_offsets, y_offsets, pad_x, pad_y = _layout(
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
        wall_px=wall_px,
    )
    return image


def _draw_dashed_polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    *,
    fill: tuple[int, int, int],
    width: int,
    dash_px: int,
    gap_px: int,
) -> None:
    """Draw a dashed polyline — dashes through `points` separated by `gap_px` voids."""
    if dash_px <= 0:
        return
    state = "dash"
    remaining = float(dash_px)
    for (x1, y1), (x2, y2) in pairwise(points):
        dx = x2 - x1
        dy = y2 - y1
        seg_len = (dx * dx + dy * dy) ** 0.5
        if seg_len == 0:
            continue
        ux = dx / seg_len
        uy = dy / seg_len
        traveled = 0.0
        cur_x, cur_y = float(x1), float(y1)
        while traveled < seg_len:
            advance = min(remaining, seg_len - traveled)
            next_x = cur_x + ux * advance
            next_y = cur_y + uy * advance
            if state == "dash":
                draw.line(
                    [(round(cur_x), round(cur_y)), (round(next_x), round(next_y))],
                    fill=fill,
                    width=width,
                )
            cur_x, cur_y = next_x, next_y
            traveled += advance
            remaining -= advance
            if remaining <= 1e-9:
                state = "gap" if state == "dash" else "dash"
                remaining = float(gap_px if state == "gap" else dash_px)


def render_solution(
    maze: GeneratedMaze,
    *,
    target_side_px: int = 2400,
    wall_thickness_px: int = DEFAULT_WALL_THICKNESS_PX,
    path_wall_ratios: dict[str, float] | None = None,
) -> Image.Image:
    """Render the same maze with the solution path overlaid as a B&W dashed line."""
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
    dash_px = max(stroke * 2, int(wall_px * _SOLUTION_DASH_PER_WALL))
    gap_px = max(stroke, int(wall_px * _SOLUTION_GAP_PER_WALL))

    def _centre(cell: tuple[int, int]) -> tuple[int, int]:
        r, c = cell
        return _cell_centre(r, c, x_offsets, y_offsets, pad_x, pad_y)

    points = [
        _centre(maze.start),
        *(_centre(cell) for cell in maze.solution),
        _centre(maze.end),
    ]
    _draw_dashed_polyline(draw, points, fill=_INK, width=stroke, dash_px=dash_px, gap_px=gap_px)
    return image
