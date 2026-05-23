"""composition QA verdict

Revision ID: 004
Revises: 003
Create Date: 2026-05-24

Adds ``rejected_composition`` to the ``image_qa_status`` enum — the new pixel
QA gate (`src/qa/composition_qa.py`) demotes pages whose subject bounding box
covers too little of the inner canvas. Distinct from the existing
``rejected_vision_composition`` (semantic). Reversible.
"""

from __future__ import annotations

from alembic import op

revision: str = "004"
down_revision: str | None = "003"
branch_labels: str | None = None
depends_on: str | None = None

_NEW_VALUE = "rejected_composition"

# Post-003 enum members (8 from 001 + 5 vision from 002). The downgrade
# rebuilds the type with exactly these — no row may carry `_NEW_VALUE` first.
_POST_003_STATUSES: tuple[str, ...] = (
    "pending",
    "passed",
    "rejected_white_pct",
    "rejected_gray_pct",
    "rejected_edges",
    "rejected_margins",
    "rejected_resolution",
    "rejected_manual",
    "rejected_vision_subject",
    "rejected_vision_composition",
    "rejected_vision_lines",
    "rejected_vision_anatomy",
    "rejected_vision_lowscore",
)


def upgrade() -> None:
    op.execute(f"ALTER TYPE image_qa_status ADD VALUE IF NOT EXISTS '{_NEW_VALUE}'")


def downgrade() -> None:
    op.execute(
        "UPDATE images SET qa_status = 'rejected_manual' "
        f"WHERE qa_status::text = '{_NEW_VALUE}'"
    )
    op.execute("ALTER TYPE image_qa_status RENAME TO image_qa_status_old")
    members = ", ".join(f"'{s}'" for s in _POST_003_STATUSES)
    op.execute(f"CREATE TYPE image_qa_status AS ENUM ({members})")
    op.execute("ALTER TABLE images ALTER COLUMN qa_status DROP DEFAULT")
    op.execute(
        "ALTER TABLE images ALTER COLUMN qa_status TYPE image_qa_status "
        "USING qa_status::text::image_qa_status"
    )
    op.execute("ALTER TABLE images ALTER COLUMN qa_status SET DEFAULT 'pending'")
    op.execute("DROP TYPE image_qa_status_old")
