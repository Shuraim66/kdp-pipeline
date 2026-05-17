"""Alembic migration environment for kdp_pipeline.

The database URL is read from application settings (`DATABASE_URL`) and
adapted for SQLAlchemy's psycopg3 driver (`postgresql+psycopg://`). The
application's own runtime uses psycopg directly; only Alembic needs the
SQLAlchemy engine, and SQLAlchemy defaults to psycopg2 unless told otherwise.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from src.settings import get_settings

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Raw-SQL migrations — there is no ORM metadata to autogenerate against.
target_metadata = None


def _database_url() -> str:
    """Application DB URL, adapted to SQLAlchemy's psycopg3 driver."""
    url = get_settings().database_url
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode — emit SQL without a DB connection."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode — against a live DB connection."""
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _database_url()
    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
