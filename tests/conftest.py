"""Shared pytest fixtures: a real niche config plus Book/Image factories."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest
from src.config.loader import load_niche_config
from src.config.schema import NicheConfig
from src.db.models import Book, BookStatus, Image, ImageQAStatus

_NICHES_DIR = Path(__file__).resolve().parent.parent / "niches"
_EPOCH = datetime(2026, 5, 17, 12, 0, tzinfo=UTC)


@pytest.fixture
def niche_config() -> NicheConfig:
    """The committed nurses niche config — 25 subjects x 2 variations = 50."""
    return load_niche_config(_NICHES_DIR / "nurses_v1.yaml")


@pytest.fixture
def make_book() -> Callable[..., Book]:
    """Factory for a `Book` with sane defaults; override any field by kwarg."""

    def _make(**overrides: object) -> Book:
        fields: dict[str, object] = {
            "id": uuid4(),
            "slug": "nurses_bold_easy_v1",
            "niche": "nurses",
            "status": BookStatus.CREATED,
            "config": {},
            "config_hash": "0" * 64,
            "title": None,
            "subtitle": None,
            "description": None,
            "keywords": None,
            "categories": None,
            "page_count": None,
            "trim_size": None,
            "price_usd": None,
            "asin": None,
            "failure_reason": None,
            "failure_phase": None,
            "created_at": _EPOCH,
            "updated_at": _EPOCH,
            "generation_started_at": None,
            "generation_finished_at": None,
            "published_at": None,
        }
        fields.update(overrides)
        return Book(**fields)

    return _make


@pytest.fixture
def make_image() -> Callable[..., Image]:
    """Factory for an `Image` with sane defaults; override any field by kwarg."""

    def _make(**overrides: object) -> Image:
        fields: dict[str, object] = {
            "id": uuid4(),
            "book_id": uuid4(),
            "sequence_num": 0,
            "prompt": "a prompt",
            "negative_prompt": "shading, gray",
            "seed": 1,
            "model": "fal-ai/flux/schnell",
            "generation_params": {},
            "local_path": None,
            "fal_url": None,
            "file_sha256": None,
            "qa_status": ImageQAStatus.PENDING,
            "qa_metrics": None,
            "qa_checked_at": None,
            "cost_usd": Decimal("0.021"),
            "retry_of_image_id": None,
            "retry_attempt": 0,
            "created_at": _EPOCH,
        }
        fields.update(overrides)
        return Image(**fields)

    return _make
