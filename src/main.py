"""kdp_pipeline CLI entrypoint.

Invoked as ``python -m src.main``. Subcommands are attached to the ``cli``
group; later phases add generation, QA, and assembly commands.
"""

from __future__ import annotations

from pathlib import Path

import click
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from pydantic import ValidationError

from src.config.loader import compute_config_hash, load_niche_config
from src.db.models import BookNotFoundError
from src.db.pool import get_pool
from src.db.repos.books import fail_book, get_book_by_id
from src.db.repos.images import list_images
from src.generators.images import plan_generation, resolve_book, run_generation
from src.providers.fal import cost_for_image, get_fal_provider
from src.settings import get_settings
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


if __name__ == "__main__":
    cli()
