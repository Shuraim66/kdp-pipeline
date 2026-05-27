"""Puzzle-book build orchestrator — walks the standard status graph.

Mirrors `src/main.py:_run_build` for puzzle books: CREATED → GENERATING
(maze generation) → GENERATION_DONE → QA_RUNNING (trivial puzzle QA) →
QA_DONE → ASSEMBLING (interior + cover) → review gate → METADATA_PENDING
→ READY. The status graph is identical to coloring so audit trails in
`book_status_log` stay consistent across book types.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from pydantic import ValidationError

from src.config.schema import NicheConfig
from src.db.models import Book, BookNotFoundError, BookStatus
from src.db.repos.books import (
    fail_book,
    get_book_by_id,
    transition_status,
)
from src.generators.cover import build_cover, generate_hero
from src.generators.images import resolve_book
from src.generators.metadata import MetadataValidationError, run_metadata
from src.providers.anthropic import get_anthropic_provider
from src.providers.fal import get_fal_provider
from src.puzzle.book_builder import assemble_puzzle_pdf, generate_and_render_book
from src.puzzle.qa import validate_puzzles
from src.settings import get_settings
from src.utils.kdp_specs import total_interior_pages
from src.utils.logging import book_log_file, logger

# 7.5 in usable area (8.5 in trim - 0.5 in margin each side) at 300 DPI.
_RENDER_SIDE_PX = 2250


def _require_book(book_id: object) -> Book:
    """Re-fetch a book by id, asserting it still exists."""
    from uuid import UUID

    if not isinstance(book_id, UUID):
        raise TypeError(f"book_id must be UUID, got {type(book_id).__name__}")
    book = get_book_by_id(book_id)
    if book is None:  # pragma: no cover - books are never deleted mid-build
        raise BookNotFoundError(f"book {book_id} vanished mid-build")
    return book


def _ensure_puzzle_config(book: Book, config: NicheConfig) -> NicheConfig:
    """Raise a click error if the resolved config isn't a puzzle book."""
    if config.puzzle is None:
        raise click.ClickException(
            f"{book.slug} is a {config.body.kind!r} book; "
            "use `build` (coloring) instead of `puzzle build`."
        )
    return config


def _assemble_puzzle_interior(book: Book, config: NicheConfig, output_dir: Path) -> Path:
    """Generate + render mazes if needed, then assemble the interior PDF."""
    mazes_dir = output_dir / book.slug / "puzzles" / "mazes"
    solutions_dir = output_dir / book.slug / "puzzles" / "solutions"
    expected_count = config.require_puzzle().puzzle.count
    have_pngs = mazes_dir.is_dir() and len(list(mazes_dir.glob("*.png"))) == expected_count

    if have_pngs and solutions_dir.is_dir():
        # Re-render is unnecessary; but we still need the GeneratedMaze objects
        # for headers — regenerate the maze metadata from the same seeds.
        from src.puzzle.mazes.generator import generate_book_mazes

        mazes = generate_book_mazes(
            config.require_puzzle().puzzle, book_seed_prefix=book.seed_prefix
        )
        maze_paths = sorted(mazes_dir.glob("*.png"))
        solution_paths = sorted(solutions_dir.glob("*.png"))
    else:
        mazes, maze_paths, solution_paths = generate_and_render_book(
            book, config, output_dir=output_dir, target_side_px=_RENDER_SIDE_PX
        )
    pdf_path = output_dir / book.slug / "pdf" / "interior.pdf"
    assemble_puzzle_pdf(
        book,
        config,
        mazes=mazes,
        maze_paths=maze_paths,
        solution_paths=solution_paths,
        output_path=pdf_path,
        author=config.metadata.author,
    )
    return pdf_path


def _assemble_puzzle_cover(book: Book, config: NicheConfig, output_dir: Path) -> Path:
    """Build the wrap cover, generating the hero art via Fal.ai if absent."""
    hero_path = output_dir / book.slug / "cover" / "hero.png"
    if not hero_path.is_file():
        try:
            provider = get_fal_provider()
        except RuntimeError as exc:
            raise click.ClickException(str(exc)) from exc
        asyncio.run(generate_hero(provider, config, output_path=hero_path))
    return build_cover(
        book,
        config,
        hero_path=hero_path,
        output_dir=output_dir,
        interior_page_count=total_interior_pages(config),
    )


def _build_summary(book_id: object, config: NicheConfig, output_dir: Path) -> None:
    """Print the puzzle-build final summary — artifacts, cost, next step."""
    from src.db.repos.api_calls import cost_breakdown_for_book, total_cost_for_book

    book = _require_book(book_id)
    book_dir = output_dir / book.slug
    pdf_dir = book_dir / "pdf"
    click.echo("")
    click.echo(f"OK  Puzzle book ready: {book.slug}")
    click.echo("")
    click.echo(f"  Title: {book.title or '(untitled)'}")
    click.echo(f"  Pages: {config.book.page_count} content, {total_interior_pages(config)} total")
    click.echo(f"  Trim:  {config.book.trim_size} in")
    click.echo(f"  Price: ${config.book.price_usd}")
    click.echo("")
    click.echo("  Artifacts:")
    click.echo(f"    Interior:  {pdf_dir / 'interior.pdf'}")
    click.echo(f"    Cover:     {pdf_dir / 'cover.pdf'}")
    click.echo(f"    Metadata:  {book_dir / 'metadata.json'}")
    click.echo(f"    Checklist: {book_dir / 'kdp_checklist.md'}")
    click.echo("")
    click.echo("  Cost breakdown:")
    breakdown = cost_breakdown_for_book(book.id)
    if not breakdown:
        click.echo("    (no API calls — interior is algorithmic; no cover hero yet)")
    for provider in sorted(breakdown):
        click.echo(f"    {provider}: ${breakdown[provider]}")
    click.echo(f"    Total: ${total_cost_for_book(book.id)}")
    click.echo("")
    click.echo(f"  Next: follow {book_dir / 'kdp_checklist.md'}")


def run_puzzle_build(yaml_path: str, *, assume_yes: bool, resume: bool) -> None:
    """Drive a puzzle book through every pipeline phase, pausing at the review gate.

    Mirrors `_run_build` exactly in shape (status graph, gate semantics,
    --yes / --resume behaviour) but skips the image-generation cost gate
    (puzzle generation is local + free) and pauses only at Gate 2 (interior
    + cover review). Algorithmic puzzle QA always passes, so there is no
    Gate 1 equivalent.
    """
    try:
        book, config = resolve_book(yaml_path)
    except (FileNotFoundError, BookNotFoundError, ValidationError, ValueError) as exc:
        click.echo(f"ERROR — {exc}", err=True)
        raise SystemExit(1) from exc

    config = _ensure_puzzle_config(book, config)
    output_dir = get_settings().output_dir

    if book.status == BookStatus.FAILED:
        click.echo(
            f"ERROR — {book.slug} is failed ({book.failure_phase}: "
            f"{book.failure_reason}). Run `retry-failed` then `puzzle build --resume`.",
            err=True,
        )
        raise SystemExit(1)
    if book.status in (BookStatus.READY, BookStatus.PUBLISHED):
        click.echo(f"{book.slug} is already {book.status} — nothing to build.")
        _build_summary(book.id, config, output_dir)
        return
    if book.status != BookStatus.CREATED and not (resume or assume_yes):
        click.echo(
            f"ERROR — {book.slug} is mid-pipeline ({book.status}); pass --resume to continue it.",
            err=True,
        )
        raise SystemExit(1)

    with book_log_file(output_dir / book.slug / "run.log"):
        _run_phases(book, config, output_dir, assume_yes=assume_yes, yaml_path=yaml_path)


def _run_phases(
    book: Book,
    config: NicheConfig,
    output_dir: Path,
    *,
    assume_yes: bool,
    yaml_path: str,
) -> None:
    """Inner loop — phases A through F. Split out so the log-file context wraps everything."""
    # --- Phase A: puzzle generation (CREATED → GENERATING → GENERATION_DONE) ---
    if book.status in (BookStatus.CREATED, BookStatus.GENERATING):
        click.echo(f"Phase A — generate {config.require_puzzle().puzzle.count} maze(s)")
        if book.status == BookStatus.CREATED:
            transition_status(book.id, BookStatus.GENERATING)
            book = _require_book(book.id)
        try:
            generate_and_render_book(
                book, config, output_dir=output_dir, target_side_px=_RENDER_SIDE_PX
            )
        except Exception as exc:  # pragma: no cover - mazelib should never error
            fail_book(book.id, phase="puzzle_generation", reason=str(exc))
            click.echo(f"ERROR — puzzle generation failed: {exc}", err=True)
            raise SystemExit(1) from exc
        transition_status(book.id, BookStatus.GENERATION_DONE)
        book = _require_book(book.id)

    # --- Phase B: puzzle QA (GENERATION_DONE → QA_RUNNING → QA_DONE) ---
    if book.status in (BookStatus.GENERATION_DONE, BookStatus.QA_RUNNING):
        click.echo("Phase B — running puzzle QA …")
        if book.status == BookStatus.GENERATION_DONE:
            transition_status(book.id, BookStatus.QA_RUNNING)
            book = _require_book(book.id)
        maze_paths = sorted((output_dir / book.slug / "puzzles" / "mazes").glob("*.png"))
        solution_paths = sorted((output_dir / book.slug / "puzzles" / "solutions").glob("*.png"))
        issues = validate_puzzles(
            maze_paths=maze_paths,
            solution_paths=solution_paths,
            expected_dimensions=(_RENDER_SIDE_PX, _RENDER_SIDE_PX),
        )
        if issues:
            fail_book(book.id, phase="puzzle_qa", reason="; ".join(issues[:3]))
            joined = "\n  - ".join(issues)
            click.echo(f"ERROR — puzzle QA failed:\n  - {joined}", err=True)
            raise SystemExit(1)
        click.echo(f"  QA passed: {len(maze_paths)} mazes + {len(solution_paths)} solutions")
        transition_status(book.id, BookStatus.QA_DONE)
        book = _require_book(book.id)

    # --- Phases C & D: interior + cover assembly (QA_DONE → ASSEMBLING) ---
    assembled = False
    if book.status == BookStatus.QA_DONE:
        transition_status(book.id, BookStatus.ASSEMBLING)
        book = _require_book(book.id)
        click.echo("Phase C — assembling puzzle interior PDF …")
        _assemble_puzzle_interior(book, config, output_dir)
        click.echo("Phase D — building the cover (Fal.ai hero) …")
        _assemble_puzzle_cover(book, config, output_dir)
        assembled = True

    # --- Gate 2: interior + cover review ---
    if assembled and not assume_yes:
        pdf_dir = output_dir / book.slug / "pdf"
        click.echo("")
        click.echo(f"STOP  Interior + cover built in {pdf_dir}.")
        click.echo(
            f"   Review interior.pdf and cover.pdf, then `puzzle build --resume {yaml_path}`."
        )
        return

    # --- Phase E: listing metadata ---
    book = _require_book(book.id)
    if book.status in (BookStatus.ASSEMBLING, BookStatus.METADATA_PENDING):
        click.echo("Phase E — generating listing metadata …")
        try:
            anthropic_provider = get_anthropic_provider()
        except RuntimeError as exc:
            click.echo(f"ERROR — {exc}", err=True)
            raise SystemExit(1) from exc
        try:
            run_metadata(book, config, provider=anthropic_provider, output_dir=output_dir)
        except MetadataValidationError as exc:
            click.echo(f"ERROR — {exc}", err=True)
            raise SystemExit(1) from exc

    # --- Phase F: final summary ---
    _build_summary(book.id, config, output_dir)
    logger.info("puzzle build complete for {}", book.slug)
