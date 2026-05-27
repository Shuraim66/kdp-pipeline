"""Book repository — all SQL touching `books` and `book_status_log`."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from src.db.models import (
    Book,
    BookNotFoundError,
    BookStatus,
    IllegalTransitionError,
    StatusLogEntry,
    is_legal_transition,
)
from src.db.pool import get_pool
from src.utils.hashing import canonical_hash

# Columns that `update_metadata` is permitted to write.
_METADATA_COLUMNS: frozenset[str] = frozenset(
    {
        "title",
        "subtitle",
        "description",
        "keywords",
        "categories",
        "page_count",
        "trim_size",
        "price_usd",
    }
)


def _book_from_row(row: dict[str, Any]) -> Book:
    return Book(
        id=row["id"],
        slug=row["slug"],
        niche=row["niche"],
        status=BookStatus(row["status"]),
        config=row["config"],
        config_hash=row["config_hash"],
        title=row["title"],
        subtitle=row["subtitle"],
        description=row["description"],
        keywords=row["keywords"],
        categories=row["categories"],
        page_count=row["page_count"],
        trim_size=row["trim_size"],
        price_usd=row["price_usd"],
        asin=row["asin"],
        failure_reason=row["failure_reason"],
        failure_phase=row["failure_phase"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        generation_started_at=row["generation_started_at"],
        generation_finished_at=row["generation_finished_at"],
        published_at=row["published_at"],
        # Pre-005 rows won't have this column when running an old DB; treat
        # absent as the historical default rather than KeyError.
        book_type=row.get("book_type") or "coloring",
    )


def create_book(
    slug: str, niche: str, config: dict[str, Any], *, book_type: str = "coloring"
) -> Book:
    """Insert a new book in status `created`. Raises on a duplicate slug.

    `book_type` is denormalised from `config["body"]["kind"]`; callers that
    already know the type pass it explicitly. Default keeps the historical
    coloring behaviour for any code path that hasn't been book-type-aware yet.
    """
    config_hash = canonical_hash(config)
    with (
        get_pool().connection() as conn,
        conn.transaction(),
        conn.cursor(row_factory=dict_row) as cur,
    ):
        cur.execute(
            "INSERT INTO books (slug, niche, config, config_hash, book_type) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING *",
            (slug, niche, Jsonb(config), config_hash, book_type),
        )
        row = cur.fetchone()
    assert row is not None  # INSERT ... RETURNING always yields one row
    return _book_from_row(row)


def get_book_by_slug(slug: str) -> Book | None:
    """Return the book with this slug, or None."""
    with (
        get_pool().connection() as conn,
        conn.cursor(row_factory=dict_row) as cur,
    ):
        cur.execute("SELECT * FROM books WHERE slug = %s", (slug,))
        row = cur.fetchone()
    return _book_from_row(row) if row is not None else None


def get_book_by_id(book_id: UUID) -> Book | None:
    """Return the book with this id, or None."""
    with (
        get_pool().connection() as conn,
        conn.cursor(row_factory=dict_row) as cur,
    ):
        cur.execute("SELECT * FROM books WHERE id = %s", (book_id,))
        row = cur.fetchone()
    return _book_from_row(row) if row is not None else None


def transition_status(book_id: UUID, to_status: BookStatus, reason: str | None = None) -> None:
    """Move a book to `to_status`, enforcing the state machine.

    The `book_status_log` entry is written by a DB trigger. A `reason`, when
    given, is stored on `books.failure_reason` so the trigger records it; it
    is mainly meaningful for transitions into `failed` (see `fail_book`).
    """
    with (
        get_pool().connection() as conn,
        conn.transaction(),
        conn.cursor(row_factory=dict_row) as cur,
    ):
        cur.execute("SELECT status FROM books WHERE id = %s FOR UPDATE", (book_id,))
        row = cur.fetchone()
        if row is None:
            raise BookNotFoundError(f"no book with id {book_id}")
        current = BookStatus(row["status"])
        if not is_legal_transition(current, to_status):
            raise IllegalTransitionError(current, to_status)
        if reason is None:
            cur.execute(
                "UPDATE books SET status = %s::book_status WHERE id = %s",
                (to_status, book_id),
            )
        else:
            cur.execute(
                "UPDATE books SET status = %s::book_status, failure_reason = %s WHERE id = %s",
                (to_status, reason, book_id),
            )


def fail_book(book_id: UUID, phase: str, reason: str) -> None:
    """Transition a book to `failed`, recording the phase and reason."""
    with (
        get_pool().connection() as conn,
        conn.transaction(),
        conn.cursor(row_factory=dict_row) as cur,
    ):
        cur.execute("SELECT status FROM books WHERE id = %s FOR UPDATE", (book_id,))
        row = cur.fetchone()
        if row is None:
            raise BookNotFoundError(f"no book with id {book_id}")
        current = BookStatus(row["status"])
        if not is_legal_transition(current, BookStatus.FAILED):
            raise IllegalTransitionError(current, BookStatus.FAILED)
        cur.execute(
            "UPDATE books SET status = 'failed'::book_status, "
            "failure_reason = %s, failure_phase = %s WHERE id = %s",
            (reason, phase, book_id),
        )


def reset_failed_book(book_id: UUID) -> bool:
    """Reset a `failed` book to `created` so the pipeline can be re-run.

    Returns whether a failed book was actually reset. This deliberately
    bypasses the state machine — `failed` is otherwise a terminal state.
    """
    with (
        get_pool().connection() as conn,
        conn.transaction(),
        conn.cursor() as cur,
    ):
        cur.execute(
            "UPDATE books SET status = 'created'::book_status, "
            "failure_reason = NULL, failure_phase = NULL "
            "WHERE id = %s AND status = 'failed'::book_status",
            (book_id,),
        )
        return cur.rowcount > 0


def update_metadata(book_id: UUID, **fields: Any) -> None:
    """Update whitelisted metadata columns on a book.

    Accepts any of: title, subtitle, description, keywords, categories,
    page_count, trim_size, price_usd.
    """
    unknown = set(fields) - _METADATA_COLUMNS
    if unknown:
        raise ValueError(f"columns not updatable here: {sorted(unknown)}")
    if not fields:
        return
    assignments = sql.SQL(", ").join(
        sql.SQL("{} = %s").format(sql.Identifier(column)) for column in fields
    )
    query = sql.SQL("UPDATE books SET {} WHERE id = %s").format(assignments)
    params: list[Any] = [*fields.values(), book_id]
    with (
        get_pool().connection() as conn,
        conn.transaction(),
        conn.cursor() as cur,
    ):
        cur.execute(query, params)


def set_asin(book_id: UUID, asin: str) -> None:
    """Record the Amazon ASIN for a published book."""
    with (
        get_pool().connection() as conn,
        conn.transaction(),
        conn.cursor() as cur,
    ):
        cur.execute("UPDATE books SET asin = %s WHERE id = %s", (asin, book_id))


def list_books(
    status: BookStatus | None = None,
    niche: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Book]:
    """List books, newest first, optionally filtered by status and/or niche."""
    conditions: list[sql.Composable] = []
    params: list[Any] = []
    if status is not None:
        conditions.append(sql.SQL("status = %s::book_status"))
        params.append(status)
    if niche is not None:
        conditions.append(sql.SQL("niche = %s"))
        params.append(niche)
    where: sql.Composable = (
        sql.SQL("WHERE ") + sql.SQL(" AND ").join(conditions) if conditions else sql.SQL("")
    )
    query = sql.SQL(
        "SELECT * FROM books {where} ORDER BY created_at DESC LIMIT %s OFFSET %s"
    ).format(where=where)
    params += [limit, offset]
    with (
        get_pool().connection() as conn,
        conn.cursor(row_factory=dict_row) as cur,
    ):
        cur.execute(query, params)
        rows = cur.fetchall()
    return [_book_from_row(row) for row in rows]


def get_status_log(book_id: UUID) -> list[StatusLogEntry]:
    """Return a book's status-transition log, oldest first."""
    with (
        get_pool().connection() as conn,
        conn.cursor(row_factory=dict_row) as cur,
    ):
        cur.execute(
            "SELECT * FROM book_status_log WHERE book_id = %s ORDER BY created_at",
            (book_id,),
        )
        rows = cur.fetchall()
    return [
        StatusLogEntry(
            id=row["id"],
            book_id=row["book_id"],
            from_status=(
                BookStatus(row["from_status"]) if row["from_status"] is not None else None
            ),
            to_status=BookStatus(row["to_status"]),
            reason=row["reason"],
            created_at=row["created_at"],
        )
        for row in rows
    ]
