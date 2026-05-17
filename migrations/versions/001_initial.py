"""initial schema

Revision ID: 001
Revises:
Create Date: 2026-05-17

Creates the full kdp_pipeline schema: the two status enums, the books /
images / api_calls / book_status_log tables, indexes, the updated_at trigger,
and the status-transition logging trigger.
"""

from __future__ import annotations

from alembic import op

revision: str = "001"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    op.execute(
        """
        CREATE TYPE book_status AS ENUM (
            'created', 'generating', 'generation_done', 'qa_running', 'qa_done',
            'assembling', 'metadata_pending', 'ready', 'published', 'failed'
        )
        """
    )
    op.execute(
        """
        CREATE TYPE image_qa_status AS ENUM (
            'pending', 'passed', 'rejected_white_pct', 'rejected_gray_pct',
            'rejected_edges', 'rejected_margins', 'rejected_resolution',
            'rejected_manual'
        )
        """
    )

    op.execute(
        """
        CREATE TABLE books (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            slug            TEXT UNIQUE NOT NULL CHECK (slug ~ '^[a-z0-9_]+$'),
            niche           TEXT NOT NULL,
            status          book_status NOT NULL DEFAULT 'created',
            config          JSONB NOT NULL,
            config_hash     TEXT NOT NULL,
            title           TEXT,
            subtitle        TEXT,
            description     TEXT,
            keywords        TEXT[] CHECK (array_length(keywords, 1) <= 7),
            categories      TEXT[] CHECK (array_length(categories, 1) <= 2),
            page_count      INT,
            trim_size       TEXT,
            price_usd       DECIMAL(5,2),
            asin            TEXT,
            failure_reason  TEXT,
            failure_phase   TEXT,
            created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
            generation_started_at   TIMESTAMPTZ,
            generation_finished_at  TIMESTAMPTZ,
            published_at            TIMESTAMPTZ
        )
        """
    )
    op.execute(
        """
        CREATE TABLE images (
            id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            book_id             UUID NOT NULL REFERENCES books(id) ON DELETE CASCADE,
            sequence_num        INT NOT NULL CHECK (sequence_num >= 0),
            prompt              TEXT NOT NULL,
            negative_prompt     TEXT,
            seed                BIGINT NOT NULL,
            model               TEXT NOT NULL,
            generation_params   JSONB NOT NULL,
            local_path          TEXT,
            fal_url             TEXT,
            file_sha256         TEXT,
            qa_status           image_qa_status NOT NULL DEFAULT 'pending',
            qa_metrics          JSONB,
            qa_checked_at       TIMESTAMPTZ,
            cost_usd            DECIMAL(8,5) NOT NULL DEFAULT 0,
            retry_of_image_id   UUID REFERENCES images(id) ON DELETE SET NULL,
            retry_attempt       INT NOT NULL DEFAULT 0,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (book_id, sequence_num, retry_attempt)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE api_calls (
            id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            book_id             UUID REFERENCES books(id) ON DELETE SET NULL,
            image_id            UUID REFERENCES images(id) ON DELETE SET NULL,
            provider            TEXT NOT NULL,
            endpoint            TEXT NOT NULL,
            operation           TEXT NOT NULL,
            request_params      JSONB,
            response_summary    JSONB,
            cost_usd            DECIMAL(8,5) NOT NULL DEFAULT 0,
            duration_ms         INT NOT NULL,
            success             BOOLEAN NOT NULL,
            error_type          TEXT,
            error_message       TEXT,
            retry_attempt       INT NOT NULL DEFAULT 0,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE book_status_log (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            book_id         UUID NOT NULL REFERENCES books(id) ON DELETE CASCADE,
            from_status     book_status,
            to_status       book_status NOT NULL,
            reason          TEXT,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )

    op.execute("CREATE INDEX idx_books_status ON books(status)")
    op.execute("CREATE INDEX idx_books_niche ON books(niche)")
    op.execute("CREATE INDEX idx_books_created_at ON books(created_at DESC)")
    op.execute("CREATE INDEX idx_images_book_id_seq ON images(book_id, sequence_num)")
    op.execute(
        "CREATE INDEX idx_images_qa_pending ON images(qa_status) "
        "WHERE qa_status = 'pending'"
    )
    op.execute("CREATE INDEX idx_api_calls_book_id ON api_calls(book_id, created_at)")
    op.execute(
        "CREATE INDEX idx_api_calls_failed ON api_calls(provider, created_at) "
        "WHERE success = false"
    )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION set_updated_at() RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = now();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        """
        CREATE TRIGGER books_updated_at
            BEFORE UPDATE ON books
            FOR EACH ROW EXECUTE FUNCTION set_updated_at()
        """
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION log_status_change() RETURNS TRIGGER AS $$
        BEGIN
            IF OLD.status IS DISTINCT FROM NEW.status THEN
                INSERT INTO book_status_log (book_id, from_status, to_status, reason)
                VALUES (NEW.id, OLD.status, NEW.status, NEW.failure_reason);
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        """
        CREATE TRIGGER books_status_log
            AFTER UPDATE OF status ON books
            FOR EACH ROW EXECUTE FUNCTION log_status_change()
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS books_status_log ON books")
    op.execute("DROP TRIGGER IF EXISTS books_updated_at ON books")
    op.execute("DROP FUNCTION IF EXISTS log_status_change()")
    op.execute("DROP FUNCTION IF EXISTS set_updated_at()")
    op.execute("DROP TABLE IF EXISTS book_status_log")
    op.execute("DROP TABLE IF EXISTS api_calls")
    op.execute("DROP TABLE IF EXISTS images")
    op.execute("DROP TABLE IF EXISTS books")
    op.execute("DROP TYPE IF EXISTS image_qa_status")
    op.execute("DROP TYPE IF EXISTS book_status")
