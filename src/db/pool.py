"""psycopg3 connection pool — a lazily opened, process-wide singleton."""

from __future__ import annotations

import atexit

from psycopg import Connection
from psycopg.rows import TupleRow
from psycopg_pool import ConnectionPool

from src.settings import get_settings

_pool: ConnectionPool[Connection[TupleRow]] | None = None


def get_pool() -> ConnectionPool[Connection[TupleRow]]:
    """Return the shared connection pool, opening it on first use.

    Sync `ConnectionPool` — the pipeline has no async DB needs. Sizing stays
    well under Supabase's free-tier connection cap; psycopg negotiates SSL
    automatically, which Supabase requires.
    """
    global _pool
    if _pool is None:
        pool: ConnectionPool[Connection[TupleRow]] = ConnectionPool(
            conninfo=get_settings().database_url,
            min_size=1,
            max_size=10,
            open=False,
            name="kdp-pipeline",
        )
        pool.open()
        _pool = pool
    return _pool


def close_pool() -> None:
    """Close the pool if open. Registered to run at interpreter shutdown."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


atexit.register(close_pool)
