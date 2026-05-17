"""Image repository — all SQL touching the `images` table."""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from uuid import UUID

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from src.db.models import Image, ImageQAStatus
from src.db.pool import get_pool


def _image_from_row(row: dict[str, Any]) -> Image:
    return Image(
        id=row["id"],
        book_id=row["book_id"],
        sequence_num=row["sequence_num"],
        prompt=row["prompt"],
        negative_prompt=row["negative_prompt"],
        seed=row["seed"],
        model=row["model"],
        generation_params=row["generation_params"],
        local_path=row["local_path"],
        fal_url=row["fal_url"],
        file_sha256=row["file_sha256"],
        qa_status=ImageQAStatus(row["qa_status"]),
        qa_metrics=row["qa_metrics"],
        qa_checked_at=row["qa_checked_at"],
        cost_usd=row["cost_usd"],
        retry_of_image_id=row["retry_of_image_id"],
        retry_attempt=row["retry_attempt"],
        created_at=row["created_at"],
        vision_qa_score=row.get("vision_qa_score"),
        vision_qa_subscores=row.get("vision_qa_subscores"),
        vision_qa_issues=row.get("vision_qa_issues"),
        vision_qa_prompt_hint=row.get("vision_qa_prompt_hint"),
        vision_qa_model=row.get("vision_qa_model"),
        vision_qa_cost_usd=row.get("vision_qa_cost_usd"),
        vision_qa_evaluated_at=row.get("vision_qa_evaluated_at"),
    )


def create_image(
    book_id: UUID,
    sequence_num: int,
    prompt: str,
    seed: int,
    model: str,
    generation_params: dict[str, Any],
    *,
    negative_prompt: str | None = None,
    local_path: str | None = None,
    fal_url: str | None = None,
    file_sha256: str | None = None,
    cost_usd: Decimal = Decimal(0),
    retry_of_image_id: UUID | None = None,
    retry_attempt: int = 0,
) -> Image:
    """Insert a generated-image row and return it."""
    with (
        get_pool().connection() as conn,
        conn.transaction(),
        conn.cursor(row_factory=dict_row) as cur,
    ):
        cur.execute(
            "INSERT INTO images (book_id, sequence_num, prompt, negative_prompt, "
            "seed, model, generation_params, local_path, fal_url, file_sha256, "
            "cost_usd, retry_of_image_id, retry_attempt) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "RETURNING *",
            (
                book_id,
                sequence_num,
                prompt,
                negative_prompt,
                seed,
                model,
                Jsonb(generation_params),
                local_path,
                fal_url,
                file_sha256,
                cost_usd,
                retry_of_image_id,
                retry_attempt,
            ),
        )
        row = cur.fetchone()
    assert row is not None  # INSERT ... RETURNING always yields one row
    return _image_from_row(row)


def update_qa(
    image_id: UUID,
    qa_status: ImageQAStatus,
    qa_metrics: dict[str, Any] | None = None,
) -> None:
    """Record the QA verdict and metrics for an image."""
    metrics = Jsonb(qa_metrics) if qa_metrics is not None else None
    with (
        get_pool().connection() as conn,
        conn.transaction(),
        conn.cursor() as cur,
    ):
        cur.execute(
            "UPDATE images SET qa_status = %s::image_qa_status, "
            "qa_metrics = %s, qa_checked_at = now() WHERE id = %s",
            (qa_status, metrics, image_id),
        )


def update_qa_status(image_id: UUID, qa_status: ImageQAStatus) -> None:
    """Set just an image's QA verdict — used by the vision QA override.

    Unlike `update_qa`, this leaves `qa_metrics` untouched: the vision verdict
    and its detail live in the dedicated `vision_qa_*` columns.
    """
    with (
        get_pool().connection() as conn,
        conn.transaction(),
        conn.cursor() as cur,
    ):
        cur.execute(
            "UPDATE images SET qa_status = %s::image_qa_status, qa_checked_at = now() "
            "WHERE id = %s",
            (qa_status, image_id),
        )


def update_vision_qa(
    image_id: UUID,
    *,
    score: int,
    subscores: dict[str, Any],
    issues: list[str],
    prompt_hint: str | None,
    model: str,
    cost_usd: Decimal,
) -> None:
    """Record an image's Vision QA score and detail.

    Written for every evaluated image regardless of pass/fail; a rejection
    additionally flips `qa_status` via `update_qa_status`.
    """
    with (
        get_pool().connection() as conn,
        conn.transaction(),
        conn.cursor() as cur,
    ):
        cur.execute(
            "UPDATE images SET vision_qa_score = %s, vision_qa_subscores = %s, "
            "vision_qa_issues = %s, vision_qa_prompt_hint = %s, vision_qa_model = %s, "
            "vision_qa_cost_usd = %s, vision_qa_evaluated_at = now() WHERE id = %s",
            (score, Jsonb(subscores), issues, prompt_hint, model, cost_usd, image_id),
        )


def list_images(book_id: UUID) -> list[Image]:
    """Return every image for a book, ordered by sequence then retry attempt."""
    with (
        get_pool().connection() as conn,
        conn.cursor(row_factory=dict_row) as cur,
    ):
        cur.execute(
            "SELECT * FROM images WHERE book_id = %s ORDER BY sequence_num, retry_attempt",
            (book_id,),
        )
        rows = cur.fetchall()
    return [_image_from_row(row) for row in rows]


def get_passed_images(book_id: UUID) -> list[Image]:
    """Return a book's QA-passed images, ordered by sequence number."""
    with (
        get_pool().connection() as conn,
        conn.cursor(row_factory=dict_row) as cur,
    ):
        cur.execute(
            "SELECT * FROM images WHERE book_id = %s "
            "AND qa_status = 'passed'::image_qa_status ORDER BY sequence_num",
            (book_id,),
        )
        rows = cur.fetchall()
    return [_image_from_row(row) for row in rows]


def get_failed_images(book_id: UUID) -> list[Image]:
    """Return a book's QA-rejected images, ordered by sequence number."""
    with (
        get_pool().connection() as conn,
        conn.cursor(row_factory=dict_row) as cur,
    ):
        cur.execute(
            "SELECT * FROM images WHERE book_id = %s AND qa_status NOT IN ("
            "'pending'::image_qa_status, 'passed'::image_qa_status"
            ") ORDER BY sequence_num",
            (book_id,),
        )
        rows = cur.fetchall()
    return [_image_from_row(row) for row in rows]


def count_by_status(book_id: UUID) -> dict[str, int]:
    """Return a map of qa_status -> image count for a book."""
    with (
        get_pool().connection() as conn,
        conn.cursor() as cur,
    ):
        cur.execute(
            "SELECT qa_status, count(*) FROM images WHERE book_id = %s GROUP BY qa_status",
            (book_id,),
        )
        rows = cur.fetchall()
    return {str(status): int(count) for status, count in rows}
