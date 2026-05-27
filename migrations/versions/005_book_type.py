"""book_type column on books

Revision ID: 005
Revises: 004
Create Date: 2026-05-28

Adds ``books.book_type`` (TEXT, NOT NULL, DEFAULT 'coloring') with a CHECK
constraint pinning the two known values: ``coloring`` and ``puzzle_maze``.
Puzzle books are introduced in this revision; existing rows default to
``coloring`` so old data keeps working. Reversible.
"""

from __future__ import annotations

from alembic import op

revision: str = "005"
down_revision: str | None = "004"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE books ADD COLUMN book_type TEXT NOT NULL DEFAULT 'coloring'"
    )
    op.execute(
        "ALTER TABLE books ADD CONSTRAINT books_book_type_check "
        "CHECK (book_type IN ('coloring', 'puzzle_maze'))"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE books DROP CONSTRAINT IF EXISTS books_book_type_check")
    op.execute("ALTER TABLE books DROP COLUMN IF EXISTS book_type")
