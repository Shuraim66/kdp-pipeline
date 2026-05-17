"""Integration tests for the database repositories.

Skipped unless DATABASE_URL_TEST points at a disposable Postgres database.
When enabled, these tests apply migrations and create/delete real rows.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL_TEST"),
    reason="DATABASE_URL_TEST not set",
)


@pytest.fixture(scope="module")
def migrated_db() -> Iterator[None]:
    """Point the app at the test database and apply migrations."""
    os.environ["DATABASE_URL"] = os.environ["DATABASE_URL_TEST"]

    from alembic import command

    from src.db import pool
    from src.main import alembic_config
    from src.settings import get_settings

    get_settings.cache_clear()
    pool.close_pool()
    command.upgrade(alembic_config(), "head")
    yield
    pool.close_pool()


def _delete_book(slug: str) -> None:
    from src.db.pool import get_pool

    with (
        get_pool().connection() as conn,
        conn.transaction(),
        conn.cursor() as cur,
    ):
        cur.execute("DELETE FROM books WHERE slug = %s", (slug,))


def test_create_transition_and_log(migrated_db: None) -> None:
    from src.db.models import BookStatus, IllegalTransitionError
    from src.db.repos import books

    slug = "test_state_machine_v1"
    _delete_book(slug)
    book = books.create_book(slug=slug, niche="test", config={"k": "v"})
    try:
        assert book.status is BookStatus.CREATED

        path = [
            BookStatus.GENERATING,
            BookStatus.GENERATION_DONE,
            BookStatus.QA_RUNNING,
            BookStatus.QA_DONE,
        ]
        for target in path:
            books.transition_status(book.id, target)

        refreshed = books.get_book_by_id(book.id)
        assert refreshed is not None
        assert refreshed.status is BookStatus.QA_DONE

        log = books.get_status_log(book.id)
        assert [entry.to_status for entry in log] == path
        assert log[0].from_status is BookStatus.CREATED

        # qa_done -> published is not a legal transition.
        with pytest.raises(IllegalTransitionError):
            books.transition_status(book.id, BookStatus.PUBLISHED)
    finally:
        _delete_book(slug)


def test_fail_book_records_reason(migrated_db: None) -> None:
    from src.db.models import BookStatus
    from src.db.repos import books

    slug = "test_fail_book_v1"
    _delete_book(slug)
    book = books.create_book(slug=slug, niche="test", config={"k": "v"})
    try:
        books.fail_book(book.id, phase="generation", reason="provider timeout")
        refreshed = books.get_book_by_id(book.id)
        assert refreshed is not None
        assert refreshed.status is BookStatus.FAILED
        assert refreshed.failure_reason == "provider timeout"
        assert refreshed.failure_phase == "generation"
    finally:
        _delete_book(slug)
