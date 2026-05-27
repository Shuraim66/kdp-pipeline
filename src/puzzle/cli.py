"""Click subgroup for the puzzle-book pipeline.

Registered by `src/main.py` via `cli.add_command(puzzle_group)` so puzzle
commands live under `puzzle ...` while every existing flat coloring
command keeps its current name and signature.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from pydantic import ValidationError

from src.config.loader import compute_config_hash, load_niche_config
from src.db.models import BookNotFoundError, BookStatus
from src.db.repos.books import get_book_by_id, transition_status
from src.generators.cover import build_cover, generate_hero
from src.generators.images import resolve_book
from src.generators.metadata import MetadataValidationError, run_metadata
from src.providers.anthropic import get_anthropic_provider
from src.providers.fal import get_fal_provider
from src.puzzle.book_builder import (
    assemble_puzzle_pdf,
    generate_and_render_book,
)
from src.puzzle.mazes.generator import generate_book_mazes
from src.puzzle.mazes.renderer import render_maze, render_solution
from src.puzzle.orchestrator import (
    _RENDER_SIDE_PX,
    _assemble_puzzle_interior,
    run_puzzle_build,
)
from src.settings import get_settings
from src.utils.kdp_specs import total_interior_pages


@click.group("puzzle")
def puzzle_group() -> None:
    """Puzzle-book pipeline (mazes; future: word search, sudoku)."""


@puzzle_group.command("validate-niche")
@click.argument("yaml_path", type=click.Path(exists=True, dir_okay=False))
def puzzle_validate_niche(yaml_path: str) -> None:
    """Validate a puzzle niche YAML config; print a summary or the errors."""
    try:
        config = load_niche_config(yaml_path)
    except ValidationError as exc:
        click.echo(f"INVALID — {yaml_path}", err=True)
        click.echo(str(exc), err=True)
        raise SystemExit(1) from exc
    except (OSError, ValueError) as exc:
        click.echo(f"ERROR — {yaml_path}: {exc}", err=True)
        raise SystemExit(1) from exc

    if config.puzzle is None:
        click.echo(
            f"ERROR — {yaml_path}: this is a {config.body.kind!r} book; "
            "use `validate-niche` instead.",
            err=True,
        )
        raise SystemExit(1)

    puzzle = config.puzzle.puzzle
    click.echo(f"VALID — {yaml_path}")
    click.echo(f"  slug:        {config.slug}")
    click.echo(f"  niche:       {config.niche}")
    click.echo(f"  imprint:     {config.imprint}")
    click.echo(
        f"  book:        {config.book.trim_size}, "
        f"{config.book.page_count} content pages, ${config.book.price_usd}"
    )
    click.echo(f"  puzzles:     {puzzle.count} {puzzle.type}s, algorithm={puzzle.algorithm}")
    click.echo(f"  difficulty:  curve={puzzle.difficulty_curve} (grids={dict(puzzle.grid_sizes)})")
    click.echo(f"  solutions:   {puzzle.solutions_section if puzzle.include_solutions else 'none'}")
    click.echo(f"  cover:       {len(config.cover.bullets)} bullet(s) configured")
    click.echo(f"  config_hash: {compute_config_hash(config)}")


@puzzle_group.command("generate-puzzles")
@click.argument("target")
def puzzle_generate_puzzles(target: str) -> None:
    """Generate every maze for the book and persist its PNGs to disk."""
    try:
        book, config = resolve_book(target)
    except (FileNotFoundError, BookNotFoundError, ValidationError, ValueError) as exc:
        click.echo(f"ERROR — {exc}", err=True)
        raise SystemExit(1) from exc
    if config.puzzle is None:
        click.echo(f"ERROR — {book.slug} is not a puzzle book.", err=True)
        raise SystemExit(1)

    output_dir = get_settings().output_dir
    if book.status == BookStatus.CREATED:
        transition_status(book.id, BookStatus.GENERATING)
    mazes, _maze_paths, solution_paths = generate_and_render_book(
        book, config, output_dir=output_dir, target_side_px=_RENDER_SIDE_PX
    )
    book = get_book_by_id(book.id) or book
    if book.status == BookStatus.GENERATING:
        transition_status(book.id, BookStatus.GENERATION_DONE)
    click.echo(f"Generated {len(mazes)} maze(s) + {len(solution_paths)} solution(s)")
    click.echo(f"Mazes:     {output_dir / book.slug / 'puzzles' / 'mazes'}")
    click.echo(f"Solutions: {output_dir / book.slug / 'puzzles' / 'solutions'}")


@puzzle_group.command("render-puzzles")
@click.argument("slug")
def puzzle_render_puzzles(slug: str) -> None:
    """Re-render every maze + solution PNG from the stored config (idempotent)."""
    try:
        book, config = resolve_book(slug)
    except (BookNotFoundError, ValidationError, ValueError) as exc:
        click.echo(f"ERROR — {exc}", err=True)
        raise SystemExit(1) from exc
    if config.puzzle is None:
        click.echo(f"ERROR — {book.slug} is not a puzzle book.", err=True)
        raise SystemExit(1)

    output_dir = get_settings().output_dir
    puzzle_body = config.require_puzzle()
    mazes = generate_book_mazes(puzzle_body.puzzle, book_seed_prefix=book.seed_prefix)
    render_spec = puzzle_body.puzzle.render
    wall_px = render_spec.wall_thickness_px
    ratios: dict[str, float] = {str(k): v for k, v in render_spec.path_wall_ratios.items()}
    mazes_dir = output_dir / book.slug / "puzzles" / "mazes"
    solutions_dir = output_dir / book.slug / "puzzles" / "solutions"
    mazes_dir.mkdir(parents=True, exist_ok=True)
    solutions_dir.mkdir(parents=True, exist_ok=True)
    for maze in mazes:
        name = f"{maze.index + 1:03d}_{maze.difficulty}.png"
        render_maze(
            maze,
            target_side_px=_RENDER_SIDE_PX,
            wall_thickness_px=wall_px,
            path_wall_ratios=ratios,
        ).save(mazes_dir / name)
        render_solution(
            maze,
            target_side_px=_RENDER_SIDE_PX,
            wall_thickness_px=wall_px,
            path_wall_ratios=ratios,
        ).save(solutions_dir / name)
    click.echo(f"Re-rendered {len(mazes)} maze + solution pair(s) to {mazes_dir.parent}")


@puzzle_group.command("build-interior")
@click.argument("slug")
@click.option(
    "--author",
    default=None,
    help="Override the author name on the title page (defaults to the niche config).",
)
def puzzle_build_interior(slug: str, author: str | None) -> None:
    """Assemble the print-ready puzzle interior PDF."""
    try:
        book, config = resolve_book(slug)
    except (BookNotFoundError, ValidationError, ValueError) as exc:
        click.echo(f"ERROR — {exc}", err=True)
        raise SystemExit(1) from exc
    if config.puzzle is None:
        click.echo(f"ERROR — {book.slug} is not a puzzle book.", err=True)
        raise SystemExit(1)
    if book.status not in (BookStatus.QA_DONE, BookStatus.ASSEMBLING):
        click.echo(
            f"ERROR — book is '{book.status}'; puzzle interior assembly needs 'qa_done'.",
            err=True,
        )
        raise SystemExit(1)

    output_dir = get_settings().output_dir
    if book.status == BookStatus.QA_DONE:
        transition_status(book.id, BookStatus.ASSEMBLING)
    # _assemble_puzzle_interior reloads cached PNGs if present (idempotent).
    pdf_path = _assemble_puzzle_interior(book, config, output_dir)
    # The --author override only affects the in-PDF byline; we still call
    # the orchestrator helper above with config.metadata.author. To honour
    # --author, re-render via the public assembler when an override is given.
    if author is not None:
        from src.puzzle.mazes.generator import generate_book_mazes

        mazes = generate_book_mazes(
            config.require_puzzle().puzzle, book_seed_prefix=book.seed_prefix
        )
        mazes_dir = output_dir / book.slug / "puzzles" / "mazes"
        solutions_dir = output_dir / book.slug / "puzzles" / "solutions"
        assemble_puzzle_pdf(
            book,
            config,
            mazes=mazes,
            maze_paths=sorted(mazes_dir.glob("*.png")),
            solution_paths=sorted(solutions_dir.glob("*.png")),
            output_path=pdf_path,
            author=author,
        )
    click.echo(f"Puzzle interior PDF: {pdf_path}")


@puzzle_group.command("build-cover")
@click.argument("slug")
@click.option(
    "--hero",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Use this image as the hero illustration instead of generating one.",
)
def puzzle_build_cover(slug: str, hero: str | None) -> None:
    """Build the print-ready wrap cover for a puzzle book (one Fal.ai call)."""
    try:
        book, config = resolve_book(slug)
    except (BookNotFoundError, ValidationError, ValueError) as exc:
        click.echo(f"ERROR — {exc}", err=True)
        raise SystemExit(1) from exc
    if config.puzzle is None:
        click.echo(f"ERROR — {book.slug} is not a puzzle book.", err=True)
        raise SystemExit(1)
    if book.status not in (BookStatus.QA_DONE, BookStatus.ASSEMBLING):
        click.echo(
            f"ERROR — book is '{book.status}'; cover assembly needs 'qa_done'.",
            err=True,
        )
        raise SystemExit(1)

    output_dir = get_settings().output_dir
    hero_path = Path(hero) if hero else output_dir / book.slug / "cover" / "hero.png"
    if not hero_path.is_file():
        click.echo("Generating hero illustration via Fal.ai FLUX dev …")
        try:
            provider = get_fal_provider()
        except RuntimeError as exc:
            click.echo(f"ERROR — {exc}", err=True)
            raise SystemExit(1) from exc
        asyncio.run(generate_hero(provider, config, output_path=hero_path))

    if book.status == BookStatus.QA_DONE:
        transition_status(book.id, BookStatus.ASSEMBLING)

    pdf = build_cover(
        book,
        config,
        hero_path=hero_path,
        output_dir=output_dir,
        interior_page_count=total_interior_pages(config),
    )
    click.echo(f"Cover written: {pdf}")


@puzzle_group.command("generate-metadata")
@click.argument("slug")
def puzzle_generate_metadata(slug: str) -> None:
    """Generate the KDP listing metadata, description, and upload checklist."""
    try:
        book, config = resolve_book(slug)
    except (BookNotFoundError, ValidationError, ValueError) as exc:
        click.echo(f"ERROR — {exc}", err=True)
        raise SystemExit(1) from exc
    if config.puzzle is None:
        click.echo(f"ERROR — {book.slug} is not a puzzle book.", err=True)
        raise SystemExit(1)
    if book.status not in (BookStatus.ASSEMBLING, BookStatus.METADATA_PENDING):
        click.echo(
            f"ERROR — book is '{book.status}'; metadata needs status 'assembling'.",
            err=True,
        )
        raise SystemExit(1)

    try:
        provider = get_anthropic_provider()
    except RuntimeError as exc:
        click.echo(f"ERROR — {exc}", err=True)
        raise SystemExit(1) from exc

    try:
        metadata = run_metadata(
            book, config, provider=provider, output_dir=get_settings().output_dir
        )
    except MetadataValidationError as exc:
        click.echo(f"ERROR — {exc}", err=True)
        raise SystemExit(1) from exc

    book_dir = get_settings().output_dir / book.slug
    click.echo(f"Title:     {metadata.title}")
    click.echo(f"Subtitle:  {metadata.subtitle}")
    click.echo(f"Keywords:  {len(metadata.keywords)}")
    click.echo(f"Desc:      {len(metadata.description)} chars")
    click.echo(f"Written:   {book_dir / 'metadata.json'}")
    click.echo(f"           {book_dir / 'description.txt'}")
    click.echo(f"           {book_dir / 'kdp_checklist.md'}")


@puzzle_group.command("build")
@click.argument("yaml_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--yes", "assume_yes", is_flag=True, help="Skip every confirmation gate.")
@click.option("--resume", is_flag=True, help="Continue a book from its current status.")
def puzzle_build(yaml_path: str, assume_yes: bool, resume: bool) -> None:
    """Run the full puzzle pipeline: generate, QA, assemble, metadata."""
    run_puzzle_build(yaml_path, assume_yes=assume_yes, resume=resume)
