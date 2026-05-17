"""kdp_pipeline CLI entrypoint.

Invoked as ``python -m src.main``. Subcommands are attached to the ``cli``
group; later phases add generation, QA, and assembly commands.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import UUID

import click
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from pydantic import ValidationError

from src.config.loader import compute_config_hash, load_niche_config
from src.db.models import BookNotFoundError, BookStatus
from src.db.pool import get_pool
from src.db.repos.books import fail_book, get_book_by_id, transition_status
from src.db.repos.images import count_by_status, list_images
from src.generators.cover import build_all_covers, generate_hero
from src.generators.images import plan_generation, resolve_book, run_generation
from src.generators.interior import build_interior_pdf
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
from src.utils.logging import configure_logging

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


if __name__ == "__main__":
    cli()
