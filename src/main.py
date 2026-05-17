"""kdp_pipeline CLI entrypoint.

Invoked as ``python -m src.main``. Subcommands are attached to the ``cli``
group; later phases add generation, QA, and assembly commands.
"""

from __future__ import annotations

import asyncio
import shutil
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import click
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from pydantic import ValidationError

from src.config.loader import compute_config_hash, load_niche_config
from src.config.schema import NicheConfig
from src.db.models import Book, BookNotFoundError, BookStatus, IllegalTransitionError
from src.db.pool import get_pool
from src.db.repos.api_calls import (
    cost_breakdown_for_book,
    total_cost_across_books,
    total_cost_for_book,
)
from src.db.repos.books import (
    fail_book,
    get_book_by_id,
    get_book_by_slug,
    get_status_log,
    list_books,
    reset_failed_book,
    set_asin,
    transition_status,
)
from src.db.repos.images import count_by_status, list_images
from src.generators.cover import build_all_covers, generate_hero
from src.generators.images import plan_generation, resolve_book, run_generation
from src.generators.interior import build_interior_pdf
from src.generators.metadata import MetadataValidationError, run_metadata
from src.providers.anthropic import get_anthropic_provider
from src.providers.fal import cost_for_image, get_fal_provider
from src.qa.pdf_qa import check_cover_pdf, check_interior_pdf
from src.qa.runner import apply_manual_verdict, build_review_html, run_qa
from src.settings import get_settings
from src.utils.kdp_specs import (
    POINTS_PER_INCH,
    compute_cover_dimensions,
    interior_layout,
    parse_trim_size,
)
from src.utils.logging import book_log_file, configure_logging

_ALEMBIC_INI = Path(__file__).resolve().parent.parent / "alembic.ini"


def alembic_config() -> Config:
    """Return an Alembic `Config` bound to the project's alembic.ini."""
    return Config(str(_ALEMBIC_INI))


@click.group()
@click.version_option("0.1.0", prog_name="kdp_pipeline")
def cli() -> None:
    """KDP coloring book pipeline."""
    configure_logging()


@cli.command("init-db")
def init_db() -> None:
    """Apply all pending database migrations (alembic upgrade head)."""
    command.upgrade(alembic_config(), "head")
    click.echo("Database migrated to head.")


@cli.command("db-status")
def db_status() -> None:
    """Show the current database migration revision."""
    head = ScriptDirectory.from_config(alembic_config()).get_current_head()
    with get_pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT EXISTS (SELECT FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = 'alembic_version')"
        )
        exists_row = cur.fetchone()
        table_exists = bool(exists_row[0]) if exists_row is not None else False
        if table_exists:
            cur.execute("SELECT version_num FROM alembic_version")
            version_row = cur.fetchone()
            current = version_row[0] if version_row is not None else None
        else:
            current = None

    click.echo(f"current revision: {current or '(none — run init-db)'}")
    click.echo(f"head revision:    {head or '(none)'}")
    if current is not None and current == head:
        click.echo("status: up to date")
    else:
        click.echo("status: OUT OF DATE — run `init-db`")


@cli.command("validate-niche")
@click.argument("yaml_path", type=click.Path(exists=True, dir_okay=False))
def validate_niche(yaml_path: str) -> None:
    """Validate a niche YAML config; print a summary or the errors."""
    try:
        config = load_niche_config(yaml_path)
    except ValidationError as exc:
        click.echo(f"INVALID — {yaml_path}", err=True)
        click.echo(str(exc), err=True)
        raise SystemExit(1) from exc
    except (OSError, ValueError) as exc:
        click.echo(f"ERROR — {yaml_path}: {exc}", err=True)
        raise SystemExit(1) from exc

    click.echo(f"VALID — {yaml_path}")
    click.echo(f"  slug:        {config.slug}")
    click.echo(f"  niche:       {config.niche}")
    click.echo(
        f"  book:        {config.book.trim_size}, "
        f"{config.book.page_count} pages, ${config.book.price_usd}"
    )
    click.echo(
        f"  subjects:    {len(config.subjects)} x {config.variations_per_subject} variations"
    )
    click.echo(f"  modifiers:   {len(config.composition_modifiers)}")
    click.echo(f"  keywords:    {len(config.metadata.keywords_seed)}")
    click.echo(f"  config_hash: {compute_config_hash(config)}")


@cli.command("generate-images")
@click.argument("target")
@click.option(
    "--test-images",
    type=click.IntRange(min=1),
    default=None,
    metavar="N",
    help="Render only N images then stop — a quality preview before the full run.",
)
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt.")
def generate_images(target: str, test_images: int | None, yes: bool) -> None:
    """Generate coloring-book images for a book.

    TARGET is either an existing book slug or the path to a niche YAML file
    (which creates the book if it does not exist yet).
    """
    try:
        book, config = resolve_book(target)
    except (FileNotFoundError, BookNotFoundError, ValidationError, ValueError) as exc:
        click.echo(f"ERROR — {exc}", err=True)
        raise SystemExit(1) from exc

    plan = plan_generation(config, list_images(book.id), limit=test_images)
    test_mode = test_images is not None
    per_image = cost_for_image(config.generation.model, *config.generation.image_dimensions)
    output_dir = get_settings().output_dir

    click.echo(f"Book:        {book.slug}  (status: {book.status})")
    click.echo(f"Slots:       {plan.total_slots} total")
    click.echo(f"  already generated: {plan.skipped}")
    if plan.exhausted:
        click.echo(f"  exhausted (retries used up): {len(plan.exhausted)}")
    label = " (test mode)" if test_mode else ""
    click.echo(f"  to generate: {len(plan.to_generate)}{label}")
    click.echo(f"Est. cost:   ${per_image * len(plan.to_generate)}  (${per_image}/image)")

    if plan.exhausted and not test_mode:
        seqs = ", ".join(f"{slot.sequence_num:03d}" for slot in plan.exhausted)
        fail_book(book.id, phase="generation", reason=f"slots exhausted retries: {seqs}")
        click.echo(f"Book marked FAILED — slots {seqs} used up their retries.", err=True)
        raise SystemExit(1)

    if not plan.to_generate:
        click.echo("Nothing to generate — every slot already has an image.")
        return

    if not yes and not click.confirm("Proceed?", default=True):
        click.echo("Aborted.")
        raise SystemExit(1)

    try:
        provider = get_fal_provider()
    except RuntimeError as exc:
        click.echo(f"ERROR — {exc}", err=True)
        raise SystemExit(1) from exc

    report = run_generation(
        book,
        config,
        plan,
        test_mode=test_mode,
        provider=provider,
        output_dir=output_dir,
    )

    click.echo("")
    click.echo(f"Generated:   {report.succeeded}")
    if report.failures:
        click.echo(
            f"Failed:      {len(report.failures)} — prompts dumped to "
            f"{output_dir / book.slug / 'failed_prompts.txt'}"
        )
    click.echo(f"Cost:        ${report.total_cost}")
    refreshed = get_book_by_id(book.id)
    if refreshed is not None:
        click.echo(f"Status:      {refreshed.status}")
    if test_mode:
        click.echo(
            f"Test mode — review {output_dir / book.slug / 'images' / 'raw'}, then "
            "re-run without --test-images for the full batch."
        )


@cli.command("run-qa")
@click.argument("slug")
def run_qa_command(slug: str) -> None:
    """Run image QA for a book — evaluate pages, regenerate rejects, finalise."""
    try:
        book, config = resolve_book(slug)
    except (BookNotFoundError, ValidationError, ValueError) as exc:
        click.echo(f"ERROR — {exc}", err=True)
        raise SystemExit(1) from exc
    if book.status not in (BookStatus.GENERATION_DONE, BookStatus.QA_RUNNING):
        click.echo(
            f"ERROR — book is '{book.status}'; QA needs status 'generation_done'.",
            err=True,
        )
        raise SystemExit(1)

    try:
        report = run_qa(
            book,
            config,
            provider_factory=get_fal_provider,
            output_dir=get_settings().output_dir,
        )
    except RuntimeError as exc:
        click.echo(f"ERROR — {exc}", err=True)
        raise SystemExit(1) from exc

    click.echo(f"QA rounds:   {report.rounds}")
    if report.regenerated:
        click.echo(f"Regenerated: {report.regenerated}")
    for status, count in sorted(report.counts.items()):
        click.echo(f"  {status}: {count}")
    click.echo(f"Status:      {report.final_status}")


@cli.command("review-qa")
@click.argument("slug")
def review_qa_command(slug: str) -> None:
    """Summarise QA, write qa_review.html, and take manual approve/reject."""
    try:
        book, _config = resolve_book(slug)
    except (BookNotFoundError, ValidationError, ValueError) as exc:
        click.echo(f"ERROR — {exc}", err=True)
        raise SystemExit(1) from exc

    images = list_images(book.id)
    if not images:
        click.echo("No images yet — run `generate-images` first.")
        return

    click.echo(f"QA summary for {book.slug}:")
    for status, count in sorted(count_by_status(book.id).items()):
        click.echo(f"  {status}: {count}")
    html_path = build_review_html(book, images, get_settings().output_dir)
    click.echo(f"Review grid: {html_path}")
    click.echo("Commands: approve <image_id> | reject <image_id> | quit")
    while True:
        line = click.prompt("review", default="quit", show_default=False).strip()
        if line in ("quit", "q", ""):
            break
        parts = line.split()
        if len(parts) != 2 or parts[0] not in ("approve", "reject"):
            click.echo("  usage: approve <image_id> | reject <image_id> | quit")
            continue
        action, raw_id = parts
        try:
            image_id = UUID(raw_id)
        except ValueError:
            click.echo(f"  not a valid image id: {raw_id}")
            continue
        apply_manual_verdict(image_id, approve=action == "approve")
        click.echo(f"  {action}d {image_id}")


@cli.command("build-interior")
@click.argument("slug")
@click.option(
    "--author",
    default=None,
    help="Override the author name on the title page (defaults to the niche config).",
)
def build_interior_command(slug: str, author: str | None) -> None:
    """Assemble the print-ready interior PDF from a book's filtered images."""
    try:
        book, config = resolve_book(slug)
    except (BookNotFoundError, ValidationError, ValueError) as exc:
        click.echo(f"ERROR — {exc}", err=True)
        raise SystemExit(1) from exc
    if book.status not in (BookStatus.QA_DONE, BookStatus.ASSEMBLING):
        click.echo(
            f"ERROR — book is '{book.status}'; interior assembly needs 'qa_done'.",
            err=True,
        )
        raise SystemExit(1)

    output_dir = get_settings().output_dir
    filtered = sorted((output_dir / book.slug / "images" / "filtered").glob("*.png"))
    if not filtered:
        click.echo(
            "ERROR — no filtered images; run `run-qa` first to produce them.",
            err=True,
        )
        raise SystemExit(1)

    if book.status == BookStatus.QA_DONE:
        transition_status(book.id, BookStatus.ASSEMBLING)

    pdf_path = output_dir / book.slug / "pdf" / "interior.pdf"
    build_interior_pdf(
        book,
        config,
        filtered_images=filtered,
        output_path=pdf_path,
        author=author if author is not None else config.metadata.author,
    )
    layout = interior_layout(config.book.trim_size)
    result = check_interior_pdf(pdf_path, expected_pages=len(filtered) + 2, layout=layout)

    click.echo(f"Interior PDF: {pdf_path}")
    click.echo(f"  pages:      {result.page_count} (expected {result.expected_pages})")
    click.echo(f"  dimensions: {'ok' if result.dimensions_ok else 'WRONG'}")
    click.echo(f"  min DPI:    {result.min_image_dpi:.0f}")
    click.echo(f"  file size:  {result.file_size_mb:.1f} MB")
    click.echo(f"  fonts:      {'embedded' if result.fonts_embedded else 'NOT embedded'}")
    if result.passed:
        click.echo("QA: PASSED")
    else:
        click.echo("QA: FAILED", err=True)
        for issue in result.issues:
            click.echo(f"  - {issue}", err=True)
        raise SystemExit(1)


@cli.command("build-cover")
@click.argument("slug")
@click.option(
    "--hero",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Use this image as the hero illustration instead of generating one.",
)
def build_cover_command(slug: str, hero: str | None) -> None:
    """Build three print-ready cover variants for a book."""
    try:
        book, config = resolve_book(slug)
    except (BookNotFoundError, ValidationError, ValueError) as exc:
        click.echo(f"ERROR — {exc}", err=True)
        raise SystemExit(1) from exc
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

    interior_pages = config.book.page_count + 2
    pdfs = build_all_covers(
        book,
        config,
        hero_path=hero_path,
        output_dir=output_dir,
        interior_page_count=interior_pages,
    )

    trim_w, trim_h = parse_trim_size(config.book.trim_size)
    dims = compute_cover_dimensions(interior_pages, trim_w, trim_h)
    expected_w = dims.total_width_in * POINTS_PER_INCH
    expected_h = dims.total_height_in * POINTS_PER_INCH

    all_passed = True
    for pdf in pdfs:
        result = check_cover_pdf(pdf, expected_width_pt=expected_w, expected_height_pt=expected_h)
        if result.passed:
            click.echo(f"  {pdf.name}: QA ok")
        else:
            all_passed = False
            click.echo(f"  {pdf.name}: QA FAILED — {'; '.join(result.issues)}", err=True)
    click.echo(
        f"3 cover variants in {output_dir / book.slug / 'pdf'} — "
        "review and rename the winner to cover.pdf."
    )
    if not all_passed:
        raise SystemExit(1)


@cli.command("generate-metadata")
@click.argument("slug")
def generate_metadata_command(slug: str) -> None:
    """Generate the KDP listing metadata, description, and upload checklist."""
    try:
        book, config = resolve_book(slug)
    except (BookNotFoundError, ValidationError, ValueError) as exc:
        click.echo(f"ERROR — {exc}", err=True)
        raise SystemExit(1) from exc
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
    refreshed = get_book_by_id(book.id)
    if refreshed is not None:
        click.echo(f"Status:    {refreshed.status}")


def _require_book(book_id: UUID) -> Book:
    """Re-fetch a book by id, asserting it still exists (it always should)."""
    book = get_book_by_id(book_id)
    if book is None:  # pragma: no cover - a book is never deleted mid-build
        raise BookNotFoundError(f"book {book_id} vanished mid-build")
    return book


def _assemble_interior(book: Book, config: NicheConfig, output_dir: Path) -> Path:
    """Build the interior PDF from a book's filtered images. Returns its path."""
    filtered = sorted((output_dir / book.slug / "images" / "filtered").glob("*.png"))
    if not filtered:
        raise click.ClickException("no filtered images — run `run-qa` first to produce them.")
    pdf_path = output_dir / book.slug / "pdf" / "interior.pdf"
    build_interior_pdf(
        book,
        config,
        filtered_images=filtered,
        output_path=pdf_path,
        author=config.metadata.author,
    )
    return pdf_path


def _assemble_cover(book: Book, config: NicheConfig, output_dir: Path) -> list[Path]:
    """Build the three cover variants, generating the hero art if absent."""
    hero_path = output_dir / book.slug / "cover" / "hero.png"
    if not hero_path.is_file():
        try:
            provider = get_fal_provider()
        except RuntimeError as exc:
            raise click.ClickException(str(exc)) from exc
        asyncio.run(generate_hero(provider, config, output_path=hero_path))
    return build_all_covers(
        book,
        config,
        hero_path=hero_path,
        output_dir=output_dir,
        interior_page_count=config.book.page_count + 2,
    )


def _build_summary(book_id: UUID, config: NicheConfig, output_dir: Path) -> None:
    """Print the spec's final build summary — artifacts, costs, next steps."""
    book = _require_book(book_id)
    book_dir = output_dir / book.slug
    pdf_dir = book_dir / "pdf"
    click.echo("")
    click.echo(f"✓ Book ready: {book.slug}")
    click.echo("")
    click.echo(f"  Title: {book.title or '(untitled)'}")
    click.echo(f"  Pages: {config.book.page_count}")
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
    breakdown = cost_breakdown_for_book(book_id)
    for provider in sorted(breakdown):
        click.echo(f"    {provider}: ${breakdown[provider]}")
    click.echo(f"    Total: ${total_cost_for_book(book_id)}")
    click.echo("")
    click.echo(f"  Next: follow {book_dir / 'kdp_checklist.md'}")


def find_orphan_images(raw_dir: Path, referenced: set[Path]) -> list[Path]:
    """Return PNGs in `raw_dir` that no `images` row references, sorted by name."""
    if not raw_dir.is_dir():
        return []
    return sorted(p for p in raw_dir.glob("*.png") if p.resolve() not in referenced)


def _run_build(yaml_path: str, *, assume_yes: bool, resume: bool, test_images: int | None) -> None:
    """Drive a book through every pipeline phase, pausing at the review gates."""
    try:
        book, config = resolve_book(yaml_path)
    except (FileNotFoundError, BookNotFoundError, ValidationError, ValueError) as exc:
        click.echo(f"ERROR — {exc}", err=True)
        raise SystemExit(1) from exc

    output_dir = get_settings().output_dir

    if book.status == BookStatus.FAILED:
        click.echo(
            f"ERROR — {book.slug} is failed ({book.failure_phase}: "
            f"{book.failure_reason}). Run `retry-failed` then `build --resume`.",
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
        # --- Phase A: image generation ---
        if book.status in (BookStatus.CREATED, BookStatus.GENERATING):
            plan = plan_generation(config, list_images(book.id), limit=test_images)
            test_mode = test_images is not None
            per_image = cost_for_image(config.generation.model, *config.generation.image_dimensions)
            est = per_image * len(plan.to_generate)

            if plan.exhausted and not test_mode:
                seqs = ", ".join(f"{s.sequence_num:03d}" for s in plan.exhausted)
                fail_book(book.id, phase="generation", reason=f"slots exhausted: {seqs}")
                click.echo(f"ERROR — slots {seqs} exhausted their retries.", err=True)
                raise SystemExit(1)

            if plan.to_generate:
                click.echo(f"Phase A — generate {len(plan.to_generate)} image(s), est. ${est}")
                ceiling = get_settings().max_book_cost_usd
                if ceiling is not None:
                    projected = total_cost_for_book(book.id) + est
                    if projected > Decimal(str(ceiling)):
                        click.echo(
                            f"ERROR — projected cost ${projected} exceeds "
                            f"MAX_BOOK_COST_USD ${ceiling}.",
                            err=True,
                        )
                        raise SystemExit(1)
                if not assume_yes and not click.confirm("Proceed?", default=True):
                    click.echo("Aborted.")
                    raise SystemExit(1)
                try:
                    provider = get_fal_provider()
                except RuntimeError as exc:
                    click.echo(f"ERROR — {exc}", err=True)
                    raise SystemExit(1) from exc
                report = run_generation(
                    book,
                    config,
                    plan,
                    test_mode=test_mode,
                    provider=provider,
                    output_dir=output_dir,
                )
                click.echo(f"  generated {report.succeeded}, cost ${report.total_cost}")
                if report.failures:
                    click.echo(
                        f"  {len(report.failures)} failed — see failed_prompts.txt",
                        err=True,
                    )
            else:
                click.echo("Phase A — images already complete.")

            if test_images is not None:
                click.echo(
                    "Test mode — review the images, then re-run `build` without --test-images."
                )
                return
            book = _require_book(book.id)
            if book.status == BookStatus.GENERATING:
                click.echo("Generation incomplete — fix the failures, then `build --resume`.")
                return

        # --- Phase B: image QA ---
        book = _require_book(book.id)
        ran_qa = False
        if book.status in (BookStatus.GENERATION_DONE, BookStatus.QA_RUNNING):
            click.echo("Phase B — running image QA …")
            try:
                qa_report = run_qa(
                    book,
                    config,
                    provider_factory=get_fal_provider,
                    output_dir=output_dir,
                )
            except RuntimeError as exc:
                click.echo(f"ERROR — {exc}", err=True)
                raise SystemExit(1) from exc
            click.echo(
                f"  QA rounds {qa_report.rounds}, "
                f"regenerated {qa_report.regenerated}, status {qa_report.final_status}"
            )
            ran_qa = True

        book = _require_book(book.id)
        if book.status == BookStatus.FAILED:
            click.echo(f"ERROR — QA failed the book: {book.failure_reason}", err=True)
            raise SystemExit(1)

        # --- Gate 1: QA review ---
        if ran_qa and not assume_yes:
            click.echo("")
            click.echo("⛔ QA complete. Review it with `review-qa`,")
            click.echo(f"   then continue with `build --resume {yaml_path}`.")
            return

        # --- Phases C & D: interior + cover assembly ---
        if book.status == BookStatus.QA_DONE:
            transition_status(book.id, BookStatus.ASSEMBLING)
            book = _require_book(book.id)
            click.echo("Phase C — assembling interior PDF …")
            _assemble_interior(book, config, output_dir)
            click.echo("Phase D — building 3 cover variants …")
            _assemble_cover(book, config, output_dir)

        # --- Gate 2: cover selection (cover.pdf is the marker) ---
        pdf_dir = output_dir / book.slug / "pdf"
        cover_pdf = pdf_dir / "cover.pdf"
        if not cover_pdf.is_file():
            variants = sorted(pdf_dir.glob("cover_variant_*.pdf"))
            if assume_yes and variants:
                shutil.copyfile(variants[0], cover_pdf)
                click.echo(f"  --yes: selected {variants[0].name} as cover.pdf")
            else:
                click.echo("")
                click.echo(f"⛔ Cover variants are ready in {pdf_dir}.")
                click.echo(
                    "   Rename the chosen one to cover.pdf, "
                    f"then continue with `build --resume {yaml_path}`."
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


@cli.command("check-env")
def check_env() -> None:
    """Validate configuration and connectivity to Postgres, Anthropic, and Fal."""
    try:
        settings = get_settings()
    except ValidationError as exc:
        click.echo("ERROR — configuration invalid (is .env present?):", err=True)
        click.echo(str(exc), err=True)
        raise SystemExit(1) from exc

    ok = True
    url = settings.database_url
    if ":6543/" in url:
        click.echo(
            "  DATABASE_URL: port 6543 is the transaction pooler — use 5432 (session pooler).",
            err=True,
        )
        ok = False
    if "sslmode=disable" in url:
        click.echo("  DATABASE_URL: sslmode=disable — Supabase requires SSL.", err=True)
        ok = False

    try:
        with get_pool().connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
        click.echo("  Database: connected.")
    except Exception as exc:  # connectivity probe — any failure is a reportable result
        detail = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        if "password authentication failed" in detail.lower():
            detail = "authentication failed — check the password in DATABASE_URL."
        click.echo(f"  Database: NOT reachable — {detail}", err=True)
        ok = False

    for label, value in (
        ("ANTHROPIC_API_KEY", settings.anthropic_api_key),
        ("FAL_KEY", settings.fal_key),
    ):
        if value:
            click.echo(f"  {label}: present.")
        else:
            click.echo(f"  {label}: missing.", err=True)
            ok = False

    if ok:
        click.echo("Environment OK.")
    else:
        click.echo("Environment has problems — see above.", err=True)
        raise SystemExit(1)


@cli.command("build")
@click.argument("yaml_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--yes", "assume_yes", is_flag=True, help="Skip every confirmation gate.")
@click.option("--resume", is_flag=True, help="Continue a book from its current status.")
@click.option(
    "--test-images",
    type=click.IntRange(min=1),
    default=None,
    metavar="N",
    help="Render only N preview images, then stop.",
)
def build_command(yaml_path: str, assume_yes: bool, resume: bool, test_images: int | None) -> None:
    """Run the full pipeline for a niche config: generate, QA, assemble, metadata."""
    _run_build(yaml_path, assume_yes=assume_yes, resume=resume, test_images=test_images)


@cli.command("list-books")
@click.option(
    "--status",
    type=click.Choice([s.value for s in BookStatus]),
    default=None,
    help="Only show books in this status.",
)
@click.option("--niche", default=None, help="Only show books in this niche.")
def list_books_command(status: str | None, niche: str | None) -> None:
    """List books, newest first, optionally filtered by status and/or niche."""
    books = list_books(status=BookStatus(status) if status is not None else None, niche=niche)
    if not books:
        click.echo("No books match.")
        return
    for book in books:
        title = book.title or "(untitled)"
        click.echo(f"  {book.slug:<32} {book.status:<18} {title}")
    click.echo(f"\n{len(books)} book(s).")


@cli.command("show")
@click.argument("slug")
def show_command(slug: str) -> None:
    """Show a book's full record — metadata, status history, and total cost."""
    book = get_book_by_slug(slug)
    if book is None:
        click.echo(f"ERROR — no book with slug {slug!r}.", err=True)
        raise SystemExit(1)

    click.echo(f"{book.slug}")
    click.echo(f"  id:       {book.id}")
    click.echo(f"  niche:    {book.niche}")
    click.echo(f"  status:   {book.status}")
    click.echo(f"  title:    {book.title or '—'}")
    click.echo(f"  price:    ${book.price_usd}" if book.price_usd else "  price:    —")
    click.echo(f"  pages:    {book.page_count or '—'}")
    click.echo(f"  trim:     {book.trim_size or '—'}")
    click.echo(f"  asin:     {book.asin or '—'}")
    if book.failure_reason:
        click.echo(f"  failure:  [{book.failure_phase}] {book.failure_reason}")
    click.echo(f"  created:  {book.created_at:%Y-%m-%d %H:%M}")
    click.echo("  status history:")
    for entry in get_status_log(book.id):
        frm = entry.from_status or "—"
        click.echo(f"    {entry.created_at:%Y-%m-%d %H:%M}  {frm} → {entry.to_status}")
    click.echo(f"  total cost: ${total_cost_for_book(book.id)}")


@cli.command("show-costs")
@click.argument("slug", required=False)
@click.option("--all", "all_books", is_flag=True, help="Total spend across all books.")
@click.option(
    "--since",
    default=None,
    metavar="YYYY-MM-DD",
    help="With --all, restrict the total to calls on or after this date.",
)
def show_costs_command(slug: str | None, all_books: bool, since: str | None) -> None:
    """Show per-book cost breakdown, or total spend with --all."""
    if all_books:
        date_range = None
        if since is not None:
            try:
                start = datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=UTC)
            except ValueError as exc:
                click.echo(f"ERROR — bad --since date {since!r} (use YYYY-MM-DD).", err=True)
                raise SystemExit(1) from exc
            date_range = (start, datetime.now(UTC))
        total = total_cost_across_books(date_range)
        click.echo(f"Total spend{f' since {since}' if since else ''}: ${total}")
        return

    if slug is None:
        click.echo("ERROR — pass a book slug, or --all for the grand total.", err=True)
        raise SystemExit(1)
    book = get_book_by_slug(slug)
    if book is None:
        click.echo(f"ERROR — no book with slug {slug!r}.", err=True)
        raise SystemExit(1)
    breakdown = cost_breakdown_for_book(book.id)
    if not breakdown:
        click.echo(f"{slug}: no API calls recorded yet.")
        return
    click.echo(f"{slug} costs:")
    for provider in sorted(breakdown):
        click.echo(f"  {provider}: ${breakdown[provider]}")
    click.echo(f"  total: ${total_cost_for_book(book.id)}")


@cli.command("set-asin")
@click.argument("slug")
@click.argument("asin")
def set_asin_command(slug: str, asin: str) -> None:
    """Record the Amazon ASIN for a published book."""
    book = get_book_by_slug(slug)
    if book is None:
        click.echo(f"ERROR — no book with slug {slug!r}.", err=True)
        raise SystemExit(1)
    set_asin(book.id, asin)
    click.echo(f"{slug}: ASIN set to {asin}.")


@cli.command("mark-published")
@click.argument("slug")
def mark_published_command(slug: str) -> None:
    """Mark a `ready` book as published on KDP."""
    book = get_book_by_slug(slug)
    if book is None:
        click.echo(f"ERROR — no book with slug {slug!r}.", err=True)
        raise SystemExit(1)
    try:
        transition_status(book.id, BookStatus.PUBLISHED)
    except IllegalTransitionError as exc:
        click.echo(f"ERROR — {exc}; the book must be 'ready' first.", err=True)
        raise SystemExit(1) from exc
    click.echo(f"{slug}: marked published.")


@cli.command("retry-failed")
@click.argument("slug")
def retry_failed_command(slug: str) -> None:
    """Reset a failed book to `created` so the pipeline can be re-run."""
    book = get_book_by_slug(slug)
    if book is None:
        click.echo(f"ERROR — no book with slug {slug!r}.", err=True)
        raise SystemExit(1)
    if reset_failed_book(book.id):
        click.echo(f"{slug}: reset to 'created' — re-run `build --resume`.")
    else:
        click.echo(f"{slug} is '{book.status}', not 'failed' — nothing to reset.")


@cli.command("cleanup-orphans")
@click.argument("slug")
@click.option("--yes", "assume_yes", is_flag=True, help="Delete without confirmation.")
def cleanup_orphans_command(slug: str, assume_yes: bool) -> None:
    """Delete raw image files on disk that no `images` row references."""
    book = get_book_by_slug(slug)
    if book is None:
        click.echo(f"ERROR — no book with slug {slug!r}.", err=True)
        raise SystemExit(1)

    referenced = {
        Path(img.local_path).resolve() for img in list_images(book.id) if img.local_path is not None
    }
    raw_dir = get_settings().output_dir / book.slug / "images" / "raw"
    orphans = find_orphan_images(raw_dir, referenced)
    if not orphans:
        click.echo(f"{slug}: no orphan files in {raw_dir}.")
        return

    click.echo(f"{slug}: {len(orphans)} orphan file(s) in {raw_dir}:")
    for path in orphans:
        click.echo(f"  {path.name}")
    if not assume_yes and not click.confirm("Delete them?", default=False):
        click.echo("Left untouched.")
        return
    for path in orphans:
        path.unlink()
    click.echo(f"Deleted {len(orphans)} file(s).")


if __name__ == "__main__":
    cli()
