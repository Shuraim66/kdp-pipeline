"""API-call repository — cost and outcome tracking for every provider call."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from psycopg import sql
from psycopg.types.json import Jsonb

from src.db.pool import get_pool


def log_api_call(
    *,
    provider: str,
    endpoint: str,
    operation: str,
    duration_ms: int,
    success: bool,
    book_id: UUID | None = None,
    image_id: UUID | None = None,
    request_params: dict[str, Any] | None = None,
    response_summary: dict[str, Any] | None = None,
    cost_usd: Decimal = Decimal(0),
    error_type: str | None = None,
    error_message: str | None = None,
    retry_attempt: int = 0,
) -> None:
    """Record one provider API call — success or failure — for cost tracking."""
    request = Jsonb(request_params) if request_params is not None else None
    response = Jsonb(response_summary) if response_summary is not None else None
    with (
        get_pool().connection() as conn,
        conn.transaction(),
        conn.cursor() as cur,
    ):
        cur.execute(
            "INSERT INTO api_calls (provider, endpoint, operation, duration_ms, "
            "success, book_id, image_id, request_params, response_summary, "
            "cost_usd, error_type, error_message, retry_attempt) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                provider,
                endpoint,
                operation,
                duration_ms,
                success,
                book_id,
                image_id,
                request,
                response,
                cost_usd,
                error_type,
                error_message,
                retry_attempt,
            ),
        )


def total_cost_for_book(book_id: UUID) -> Decimal:
    """Return the summed API cost attributed to a book."""
    with (
        get_pool().connection() as conn,
        conn.cursor() as cur,
    ):
        cur.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM api_calls WHERE book_id = %s",
            (book_id,),
        )
        row = cur.fetchone()
    assert row is not None
    return Decimal(row[0])


def total_cost_across_books(
    date_range: tuple[datetime, datetime] | None = None,
) -> Decimal:
    """Return the summed API cost, optionally within a [start, end) window."""
    query: sql.SQL | sql.Composed = sql.SQL("SELECT COALESCE(SUM(cost_usd), 0) FROM api_calls")
    params: list[Any] = []
    if date_range is not None:
        query = query + sql.SQL(" WHERE created_at >= %s AND created_at < %s")
        params += list(date_range)
    with (
        get_pool().connection() as conn,
        conn.cursor() as cur,
    ):
        cur.execute(query, params)
        row = cur.fetchone()
    assert row is not None
    return Decimal(row[0])
