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
from src.db.pool import get_pool
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


if __name__ == "__main__":
    cli()
