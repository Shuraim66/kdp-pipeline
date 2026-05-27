"""Puzzle interior PDF assembly.

`render_all_puzzles` renders every maze + solution PNG to disk under
``output/<slug>/puzzles/{mazes,solutions}/``. `assemble_puzzle_pdf` lays
them out into the interior PDF: title page (optional), copyright page
(always — solver tips by default), intro page (optional), maze pages
with a "Maze N — Difficulty" header, then a "Solutions" divider page and
the solution pages (when ``solutions_section == "end"``).

Reuses the shared front-matter helpers from `src/generators/front_matter.py`
— title and copyright pages render identically to coloring books, just with
solver-specific tips instead of coloring tips.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from src.config.schema import NicheConfig, PuzzleFrontMatterSpec
from src.db.models import Book
from src.generators.front_matter import (
    BODY_SIZE,
    FONT_BOLD,
    FONT_REGULAR,
    TITLE_SIZE,
    draw_copyright_page,
    draw_title_page,
    register_fonts,
)
from src.puzzle.mazes.generator import generate_book_mazes
from src.puzzle.mazes.models import GeneratedMaze
from src.puzzle.mazes.renderer import render_maze, render_solution
from src.utils.kdp_specs import InteriorLayout, interior_layout
from src.utils.logging import logger

# Solver-side default tips for the copyright page when the niche YAML supplies none.
_DEFAULT_MAZE_TIPS: tuple[str, ...] = (
    "Tips for solving:",
    "• Use a pencil so you can erase if you take a wrong turn.",
    "• Start at the green square, finish at the red square.",
    "• Each puzzle gets a little trickier — take your time.",
    "• Stuck? Solutions are at the back of the book.",
)

# Header text sizes for the maze and solution pages.
_PAGE_HEADER_SIZE = 22
_PAGE_SUBHEADER_SIZE = 13


def render_all_puzzles(
    mazes: list[GeneratedMaze],
    *,
    target_side_px: int,
    output_dir: Path,
    slug: str,
) -> tuple[list[Path], list[Path]]:
    """Render every maze + solution PNG to disk; return (maze_paths, solution_paths).

    Files land at ``output/<slug>/puzzles/mazes/{NNN_difficulty}.png`` and
    ``output/<slug>/puzzles/solutions/{NNN_difficulty}.png``. The directories
    are created if missing.
    """
    mazes_dir = output_dir / slug / "puzzles" / "mazes"
    solutions_dir = output_dir / slug / "puzzles" / "solutions"
    mazes_dir.mkdir(parents=True, exist_ok=True)
    solutions_dir.mkdir(parents=True, exist_ok=True)

    maze_paths: list[Path] = []
    solution_paths: list[Path] = []
    for maze in mazes:
        # Sequence number is 1-based in the filename so a reader scanning the
        # directory listing matches the page header "Maze 1", "Maze 2", ...
        name = f"{maze.index + 1:03d}_{maze.difficulty}.png"
        maze_path = mazes_dir / name
        render_maze(maze, target_side_px=target_side_px).save(maze_path)
        maze_paths.append(maze_path)

        solution_path = solutions_dir / name
        render_solution(maze, target_side_px=target_side_px).save(solution_path)
        solution_paths.append(solution_path)
    logger.info(
        "rendered {} maze PNG(s) + {} solution PNG(s) to {}",
        len(maze_paths),
        len(solution_paths),
        mazes_dir.parent,
    )
    return maze_paths, solution_paths


def _draw_intro_page(pdf: canvas.Canvas, layout: InteriorLayout, intro_text: str) -> None:
    """Render a short prose intro on a single front-matter page."""
    margin_x = (layout.page_width - layout.text_safe_width) / 2
    y = (layout.page_height + layout.text_safe_height) / 2 - BODY_SIZE
    pdf.setFont(FONT_REGULAR, BODY_SIZE)
    # Naive paragraph rendering: split on blank lines, word-wrap each.
    paragraphs = [block.strip() for block in intro_text.strip().split("\n") if block.strip()]
    for paragraph in paragraphs:
        for line in _wrap_simple(paragraph, FONT_REGULAR, BODY_SIZE, layout.text_safe_width):
            pdf.drawString(margin_x, y, line)
            y -= BODY_SIZE * 1.8


def _wrap_simple(text: str, font: str, size: float, max_width: float) -> list[str]:
    """Naive greedy word-wrap — shared utility duplicated to avoid circular imports."""
    from reportlab.pdfbase import pdfmetrics

    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and pdfmetrics.stringWidth(candidate, font, size) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def _draw_divider_page(pdf: canvas.Canvas, layout: InteriorLayout, title: str) -> None:
    """A full-page section header — used for the "Solutions" divider."""
    pdf.setFont(FONT_BOLD, TITLE_SIZE)
    pdf.drawCentredString(layout.page_width / 2, layout.page_height / 2, title)


def _draw_image_page(
    pdf: canvas.Canvas,
    layout: InteriorLayout,
    *,
    image_path: Path,
    header: str,
    subheader: str,
) -> None:
    """One maze/solution page — header at top, image centred below."""
    # Header text — bold, near top, inside the text-safe inset.
    text_margin_top = (layout.page_height - layout.text_safe_height) / 2
    pdf.setFont(FONT_BOLD, _PAGE_HEADER_SIZE)
    pdf.drawCentredString(
        layout.page_width / 2,
        layout.page_height - text_margin_top - _PAGE_HEADER_SIZE,
        header,
    )
    pdf.setFont(FONT_REGULAR, _PAGE_SUBHEADER_SIZE)
    pdf.drawCentredString(
        layout.page_width / 2,
        layout.page_height - text_margin_top - _PAGE_HEADER_SIZE - _PAGE_SUBHEADER_SIZE * 1.4,
        subheader,
    )
    # Image — placed below the header, fits the safe area minus the header band.
    header_band = _PAGE_HEADER_SIZE * 3.5
    available_width = layout.safe_width
    available_height = layout.safe_height - header_band
    with PILImage.open(image_path) as handle:
        reader = ImageReader(handle.copy())
    img_w, img_h = PILImage.open(image_path).size
    scale = min(available_width / img_w, available_height / img_h)
    draw_w = img_w * scale
    draw_h = img_h * scale
    x = (layout.page_width - draw_w) / 2
    y = (layout.page_height - layout.safe_height) / 2  # bottom-align inside safe area
    pdf.drawImage(reader, x, y, width=draw_w, height=draw_h)


def assemble_puzzle_pdf(
    book: Book,
    config: NicheConfig,
    *,
    mazes: list[GeneratedMaze],
    maze_paths: list[Path],
    solution_paths: list[Path],
    output_path: Path,
    author: str,
    year: int | None = None,
) -> Path:
    """Assemble the interior PDF: title, copyright, intro, mazes, solutions."""
    if not maze_paths:
        raise ValueError("no maze pages to assemble into a puzzle interior PDF")
    puzzle_body = config.require_puzzle()
    front_matter = puzzle_body.front_matter
    register_fonts()
    layout = interior_layout(config.book.trim_size)
    title = book.title or config.metadata.title_seed
    year = year if year is not None else datetime.now(UTC).year
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = canvas.Canvas(
        str(output_path),
        pagesize=(layout.page_width, layout.page_height),
        initialFontName=FONT_REGULAR,
    )
    pdf.setTitle(title)
    pdf.setAuthor(author)

    page_count = _draw_book_pages(
        pdf,
        layout,
        title=title,
        author=author,
        year=year,
        front_matter=front_matter,
        mazes=mazes,
        maze_paths=maze_paths,
        solution_paths=solution_paths,
        include_solutions_section=(
            puzzle_body.puzzle.include_solutions and puzzle_body.puzzle.solutions_section == "end"
        ),
    )
    pdf.save()
    logger.info("puzzle interior PDF written: {} ({} pages)", output_path, page_count)
    return output_path


def _draw_book_pages(
    pdf: canvas.Canvas,
    layout: InteriorLayout,
    *,
    title: str,
    author: str,
    year: int,
    front_matter: PuzzleFrontMatterSpec,
    mazes: list[GeneratedMaze],
    maze_paths: list[Path],
    solution_paths: list[Path],
    include_solutions_section: bool,
) -> int:
    """Lay out every page; return the total page count for logging."""
    pages = 0
    if front_matter.title_page:
        draw_title_page(pdf, layout, title, author)
        pdf.showPage()
        pages += 1
    tips = front_matter.copyright_tips or _DEFAULT_MAZE_TIPS
    draw_copyright_page(pdf, layout, author, year, tips)
    pdf.showPage()
    pages += 1
    if front_matter.intro_page and front_matter.intro_text:
        _draw_intro_page(pdf, layout, front_matter.intro_text)
        pdf.showPage()
        pages += 1
    elif front_matter.intro_page:
        # Toggled on but no text — render an empty page so the page-count math holds.
        pdf.showPage()
        pages += 1

    for maze, path in zip(mazes, maze_paths, strict=True):
        _draw_image_page(
            pdf,
            layout,
            image_path=path,
            header=f"Maze {maze.index + 1}",
            subheader=f"Difficulty: {maze.difficulty.capitalize()}",
        )
        pdf.showPage()
        pages += 1

    if include_solutions_section:
        _draw_divider_page(pdf, layout, "Solutions")
        pdf.showPage()
        pages += 1
        for maze, path in zip(mazes, solution_paths, strict=True):
            _draw_image_page(
                pdf,
                layout,
                image_path=path,
                header=f"Solution to Maze {maze.index + 1}",
                subheader=f"Difficulty: {maze.difficulty.capitalize()}",
            )
            pdf.showPage()
            pages += 1
    return pages


def generate_and_render_book(
    book: Book, config: NicheConfig, *, output_dir: Path, target_side_px: int
) -> tuple[list[GeneratedMaze], list[Path], list[Path]]:
    """Convenience entry point: generate every maze, render every PNG.

    Returns (mazes, maze_paths, solution_paths). The orchestrator calls this
    after entering the GENERATING status and uses the lengths to advance the
    book to GENERATION_DONE.
    """
    puzzle_body = config.require_puzzle()
    mazes = generate_book_mazes(puzzle_body.puzzle, book_seed_prefix=book.seed_prefix)
    maze_paths, solution_paths = render_all_puzzles(
        mazes,
        target_side_px=target_side_px,
        output_dir=output_dir,
        slug=book.slug,
    )
    return mazes, maze_paths, solution_paths
