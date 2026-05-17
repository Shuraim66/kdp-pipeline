"""Tests for KDP listing-metadata generation, validation, and the checklist."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
from src.generators.metadata import (
    BookMetadata,
    MetadataValidationError,
    build_checklist,
    build_metadata_prompt,
    generate_metadata,
    validate_metadata,
)
from src.providers.anthropic import AnthropicResult


def _valid(config: Any) -> dict[str, Any]:
    """A metadata object that clears every validation rule."""
    return {
        "title": "Calm Nurses Coloring Book",
        "subtitle": "50 Bold and Easy Designs for Relaxation",
        "description": "Relaxing line art for everyone. " * 55,  # ~1760 chars
        "keywords": [f"relaxing keyword {i}" for i in range(7)],
        "categories": list(config.metadata.categories),
    }


class _FakeAnthropic:
    """A stand-in AnthropicProvider yielding canned JSON responses."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def generate_json(
        self,
        *,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1024,
        operation: str = "generate_json",
        model: str | None = None,
        book_id: object = None,
    ) -> tuple[dict[str, Any], AnthropicResult]:
        self.calls += 1
        data = self._responses.pop(0)
        result = AnthropicResult(
            text="{}",
            input_tokens=120,
            output_tokens=300,
            cost_usd=Decimal("0.0048"),
            duration_s=0.2,
            stop_reason="end_turn",
        )
        return data, result


# --- validation --------------------------------------------------------------


def test_validate_metadata_accepts_a_valid_object(make_niche_config) -> None:
    config = make_niche_config()
    assert validate_metadata(_valid(config), config) == []


def test_validate_metadata_rejects_long_title_and_subtitle(make_niche_config) -> None:
    config = make_niche_config()
    data = _valid(config)
    data["title"] = "x" * 150
    data["subtitle"] = "y" * 100
    assert any("title + subtitle" in issue for issue in validate_metadata(data, config))


def test_validate_metadata_rejects_short_description(make_niche_config) -> None:
    config = make_niche_config()
    data = _valid(config)
    data["description"] = "far too short"
    assert any("description" in issue for issue in validate_metadata(data, config))


def test_validate_metadata_rejects_wrong_keyword_count(make_niche_config) -> None:
    config = make_niche_config()
    data = _valid(config)
    data["keywords"] = data["keywords"][:5]
    assert any("keyword" in issue for issue in validate_metadata(data, config))


def test_validate_metadata_rejects_duplicate_keywords(make_niche_config) -> None:
    config = make_niche_config()
    data = _valid(config)
    data["keywords"] = ["the same keyword"] * 7
    assert any("duplicate" in issue for issue in validate_metadata(data, config))


def test_validate_metadata_rejects_overlong_keyword(make_niche_config) -> None:
    config = make_niche_config()
    data = _valid(config)
    data["keywords"][0] = "x" * 60
    assert any("too long" in issue for issue in validate_metadata(data, config))


def test_validate_metadata_rejects_wrong_categories(make_niche_config) -> None:
    config = make_niche_config()
    data = _valid(config)
    data["categories"] = ["Books > Wrong > One", "Books > Wrong > Two"]
    assert any("categories" in issue for issue in validate_metadata(data, config))


def test_validate_metadata_rejects_a_missing_field(make_niche_config) -> None:
    config = make_niche_config()
    data = _valid(config)
    del data["description"]
    assert any("missing field" in issue for issue in validate_metadata(data, config))


# --- prompt ------------------------------------------------------------------


def test_build_metadata_prompt_includes_context_and_errors(make_niche_config) -> None:
    config = make_niche_config()
    system, user = build_metadata_prompt(config)
    assert "JSON" in system
    assert config.niche in user
    assert config.metadata.title_seed in user

    _, retry = build_metadata_prompt(config, prior_errors=["description too short"])
    assert "description too short" in retry
    assert "previous attempt" in retry.lower()


# --- generation loop ---------------------------------------------------------


async def test_generate_metadata_succeeds_first_try(make_niche_config) -> None:
    config = make_niche_config()
    provider = _FakeAnthropic([_valid(config)])
    metadata, result = await generate_metadata(provider, config)  # type: ignore[arg-type]
    assert metadata.title == "Calm Nurses Coloring Book"
    assert len(metadata.keywords) == 7
    assert result.cost_usd == Decimal("0.0048")
    assert provider.calls == 1


async def test_generate_metadata_retries_then_succeeds(make_niche_config) -> None:
    config = make_niche_config()
    bad = {**_valid(config), "description": "too short"}
    provider = _FakeAnthropic([bad, _valid(config)])
    metadata, _ = await generate_metadata(provider, config)  # type: ignore[arg-type]
    assert metadata.title
    assert provider.calls == 2


async def test_generate_metadata_fails_after_max_attempts(make_niche_config) -> None:
    config = make_niche_config()
    bad = {**_valid(config), "keywords": ["only one keyword"]}
    provider = _FakeAnthropic([bad, bad, bad])
    with pytest.raises(MetadataValidationError):
        await generate_metadata(provider, config)  # type: ignore[arg-type]
    assert provider.calls == 3


# --- checklist ---------------------------------------------------------------


def test_build_checklist_contains_listing_details(make_niche_config, make_book) -> None:
    config = make_niche_config()
    book = make_book(slug="m_v1")
    metadata = BookMetadata(
        title="A Splendid Title",
        subtitle="A Splendid Subtitle",
        description="d" * 1600,
        keywords=[f"keyword {i}" for i in range(7)],
        categories=list(config.metadata.categories),
    )

    text = build_checklist(book, config, metadata, interior_page_count=config.book.page_count + 2)

    assert "A Splendid Title" in text
    assert "keyword 0" in text
    assert config.metadata.author in text
    assert "set-asin m_v1" in text
