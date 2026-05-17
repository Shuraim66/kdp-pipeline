"""vision QA columns and statuses

Revision ID: 002
Revises: 001
Create Date: 2026-05-18

Adds the Vision QA stage's storage: five ``rejected_vision_*`` values on the
``image_qa_status`` enum, seven ``vision_qa_*`` columns on ``images``, and a
partial index over ``(book_id, vision_qa_score)``. Reversible.
"""

from __future__ import annotations

from alembic import op

revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | None = None
depends_on: str | None = None

# The semantic-failure verdicts Vision QA can return, on top of the pixel-QA
# statuses created in 001.
_VISION_STATUSES: tuple[str, ...] = (
    "rejected_vision_subject",
    "rejected_vision_composition",
    "rejected_vision_lines",
    "rejected_vision_anatomy",
    "rejected_vision_lowscore",
)

# The 001 enum members — the downgrade rebuilds the type with exactly these.
_ORIGINAL_STATUSES: tuple[str, ...] = (
    "pending",
    "passed",
    "rejected_white_pct",
    "rejected_gray_pct",
    "rejected_edges",
    "rejected_margins",
    "rejected_resolution",
    "rejected_manual",
)


def upgrade() -> None:
    # Postgres 12+ allows ALTER TYPE ... ADD VALUE inside a transaction as long
    # as the new value is not *used* in the same transaction — it is not here.
    for value in _VISION_STATUSES:
        op.execute(f"ALTER TYPE image_qa_status ADD VALUE IF NOT EXISTS '{value}'")

    op.execute(
        """
        ALTER TABLE images
            ADD COLUMN vision_qa_score        INT,
            ADD COLUMN vision_qa_subscores    JSONB,
            ADD COLUMN vision_qa_issues       TEXT[],
            ADD COLUMN vision_qa_prompt_hint  TEXT,
            ADD COLUMN vision_qa_model        TEXT,
            ADD COLUMN vision_qa_cost_usd     DECIMAL(8,5),
            ADD COLUMN vision_qa_evaluated_at TIMESTAMPTZ
        """
    )
    op.execute(
        "CREATE INDEX idx_images_vision_score ON images(book_id, vision_qa_score) "
        "WHERE vision_qa_score IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_images_vision_score")
    op.execute(
        """
        ALTER TABLE images
            DROP COLUMN IF EXISTS vision_qa_score,
            DROP COLUMN IF EXISTS vision_qa_subscores,
            DROP COLUMN IF EXISTS vision_qa_issues,
            DROP COLUMN IF EXISTS vision_qa_prompt_hint,
            DROP COLUMN IF EXISTS vision_qa_model,
            DROP COLUMN IF EXISTS vision_qa_cost_usd,
            DROP COLUMN IF EXISTS vision_qa_evaluated_at
        """
    )
    # Postgres has no ALTER TYPE ... DROP VALUE, so rebuild the enum without the
    # vision verdicts. Any image still carrying one is first reset to a pixel-QA
    # rejection so the text->enum cast below cannot fail on a stale value.
    op.execute(
        "UPDATE images SET qa_status = 'rejected_manual' "
        "WHERE qa_status::text LIKE 'rejected_vision_%'"
    )
    op.execute("ALTER TYPE image_qa_status RENAME TO image_qa_status_old")
    members = ", ".join(f"'{status}'" for status in _ORIGINAL_STATUSES)
    op.execute(f"CREATE TYPE image_qa_status AS ENUM ({members})")
    op.execute("ALTER TABLE images ALTER COLUMN qa_status DROP DEFAULT")
    op.execute(
        "ALTER TABLE images ALTER COLUMN qa_status TYPE image_qa_status "
        "USING qa_status::text::image_qa_status"
    )
    op.execute("ALTER TABLE images ALTER COLUMN qa_status SET DEFAULT 'pending'")
    op.execute("DROP TYPE image_qa_status_old")
