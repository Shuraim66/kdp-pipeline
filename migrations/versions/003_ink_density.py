"""ink density metric column

Revision ID: 003
Revises: 002
Create Date: 2026-05-18

Adds ``images.ink_density_pct`` — the percentage of a generated page that is
ink, measured by the post-process (`src/utils/line_art.py`). It is advisory
only: it feeds the Q6 feedback-loop tooling (`ink-density`,
`subject-performance`) and is never tied to ``qa_status``. ``REAL`` so the
driver returns a plain ``float``; nullable, so pre-Q6 rows simply carry NULL.
Reversible.
"""

from __future__ import annotations

from alembic import op

revision: str = "003"
down_revision: str | None = "002"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE images ADD COLUMN ink_density_pct REAL")


def downgrade() -> None:
    op.execute("ALTER TABLE images DROP COLUMN IF EXISTS ink_density_pct")
