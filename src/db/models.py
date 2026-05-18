"""Domain models, status enums, and the book status state machine."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID


class BookStatus(enum.StrEnum):
    """Lifecycle states for a book. Mirrors the `book_status` PG enum."""

    CREATED = "created"
    GENERATING = "generating"
    GENERATION_DONE = "generation_done"
    QA_RUNNING = "qa_running"
    QA_DONE = "qa_done"
    ASSEMBLING = "assembling"
    METADATA_PENDING = "metadata_pending"
    READY = "ready"
    PUBLISHED = "published"
    FAILED = "failed"


class ImageQAStatus(enum.StrEnum):
    """QA verdict for a generated image. Mirrors the `image_qa_status` enum."""

    PENDING = "pending"
    PASSED = "passed"
    REJECTED_WHITE_PCT = "rejected_white_pct"
    REJECTED_GRAY_PCT = "rejected_gray_pct"
    REJECTED_EDGES = "rejected_edges"
    REJECTED_MARGINS = "rejected_margins"
    REJECTED_RESOLUTION = "rejected_resolution"
    REJECTED_MANUAL = "rejected_manual"
    # Vision QA verdicts — semantic failures the pixel checks cannot see.
    REJECTED_VISION_SUBJECT = "rejected_vision_subject"
    REJECTED_VISION_COMPOSITION = "rejected_vision_composition"
    REJECTED_VISION_LINES = "rejected_vision_lines"
    REJECTED_VISION_ANATOMY = "rejected_vision_anatomy"
    REJECTED_VISION_LOWSCORE = "rejected_vision_lowscore"


# The state machine: allowed `current -> {targets}` book status transitions.
LEGAL_TRANSITIONS: dict[BookStatus, frozenset[BookStatus]] = {
    BookStatus.CREATED: frozenset({BookStatus.GENERATING, BookStatus.FAILED}),
    BookStatus.GENERATING: frozenset({BookStatus.GENERATION_DONE, BookStatus.FAILED}),
    BookStatus.GENERATION_DONE: frozenset({BookStatus.QA_RUNNING, BookStatus.FAILED}),
    BookStatus.QA_RUNNING: frozenset({BookStatus.QA_DONE, BookStatus.FAILED}),
    BookStatus.QA_DONE: frozenset(
        {BookStatus.ASSEMBLING, BookStatus.GENERATING, BookStatus.FAILED}
    ),
    BookStatus.ASSEMBLING: frozenset({BookStatus.METADATA_PENDING, BookStatus.FAILED}),
    BookStatus.METADATA_PENDING: frozenset({BookStatus.READY, BookStatus.FAILED}),
    BookStatus.READY: frozenset({BookStatus.PUBLISHED}),
    BookStatus.PUBLISHED: frozenset(),
    BookStatus.FAILED: frozenset(),
}


def is_legal_transition(current: BookStatus, target: BookStatus) -> bool:
    """Return whether moving `current -> target` is allowed."""
    return target in LEGAL_TRANSITIONS[current]


class IllegalTransitionError(Exception):
    """Raised when a status transition violates the state machine."""

    def __init__(self, current: BookStatus, target: BookStatus) -> None:
        super().__init__(f"illegal book status transition: {current} -> {target}")
        self.current = current
        self.target = target


class BookNotFoundError(Exception):
    """Raised when a book lookup by id finds no row."""


@dataclass(frozen=True, slots=True)
class Book:
    """A row of the `books` table."""

    id: UUID
    slug: str
    niche: str
    status: BookStatus
    config: dict[str, Any]
    config_hash: str
    title: str | None
    subtitle: str | None
    description: str | None
    keywords: list[str] | None
    categories: list[str] | None
    page_count: int | None
    trim_size: str | None
    price_usd: Decimal | None
    asin: str | None
    failure_reason: str | None
    failure_phase: str | None
    created_at: datetime
    updated_at: datetime
    generation_started_at: datetime | None
    generation_finished_at: datetime | None
    published_at: datetime | None


@dataclass(frozen=True, slots=True)
class Image:
    """A row of the `images` table."""

    id: UUID
    book_id: UUID
    sequence_num: int
    prompt: str
    negative_prompt: str | None
    seed: int
    model: str
    generation_params: dict[str, Any]
    local_path: str | None
    fal_url: str | None
    file_sha256: str | None
    qa_status: ImageQAStatus
    qa_metrics: dict[str, Any] | None
    qa_checked_at: datetime | None
    cost_usd: Decimal
    retry_of_image_id: UUID | None
    retry_attempt: int
    created_at: datetime
    # Vision QA fields — populated by the vision QA stage; null until then.
    vision_qa_score: int | None = None
    vision_qa_subscores: dict[str, Any] | None = None
    vision_qa_issues: list[str] | None = None
    vision_qa_prompt_hint: str | None = None
    vision_qa_model: str | None = None
    vision_qa_cost_usd: Decimal | None = None
    vision_qa_evaluated_at: datetime | None = None
    # Ink-density metric — percentage of the page that is ink, measured by the
    # post-process. Advisory: feeds the Q6 feedback tooling, never gates QA.
    ink_density_pct: float | None = None


@dataclass(frozen=True, slots=True)
class StatusLogEntry:
    """A row of the `book_status_log` table."""

    id: UUID
    book_id: UUID
    from_status: BookStatus | None
    to_status: BookStatus
    reason: str | None
    created_at: datetime
