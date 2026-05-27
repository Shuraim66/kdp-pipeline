"""Tests for KDP listing-metadata generation, validation, and the checklist."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from src.config.description_templates import (
    load_description_templates,
    pick_template,
    skeleton_ngram_overlap,
)
from src.config.loader import load_niche_config
from src.generators.metadata import (
    BookMetadata,
    MetadataValidationError,
    build_checklist,
    build_metadata_prompt,
    generate_metadata,
    persist_metadata,
    validate_metadata,
)
from src.providers.anthropic import AnthropicResult

_NICHES_DIR = Path(__file__).resolve().parent.parent / "niches"
# A fixed UUID — deterministic template rotation, deterministic prompt output.
_SNAPSHOT_BOOK_ID = UUID("12345678-1234-5678-1234-567812345678")


def _book_metadata(config: Any) -> BookMetadata:
    """A BookMetadata stand-in that clears every validation rule."""
    return BookMetadata(
        title="A Splendid Title",
        subtitle="A Splendid Subtitle",
        description="d" * 1600,
        keywords=[f"keyword {i}" for i in range(7)],
        categories=list(config.metadata.categories),
    )


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
    bid = uuid4()
    system, user = build_metadata_prompt(config, book_id=bid)
    assert "JSON" in system
    assert config.niche in user
    assert config.metadata.title_seed in user

    _, retry = build_metadata_prompt(config, book_id=bid, prior_errors=["description too short"])
    assert "description too short" in retry
    assert "previous attempt" in retry.lower()


# --- generation loop ---------------------------------------------------------


async def test_generate_metadata_succeeds_first_try(make_niche_config) -> None:
    config = make_niche_config()
    provider = _FakeAnthropic([_valid(config)])
    metadata, result = await generate_metadata(provider, config, book_id=uuid4())  # type: ignore[arg-type]
    assert metadata.title == "Calm Nurses Coloring Book"
    assert len(metadata.keywords) == 7
    assert result.cost_usd == Decimal("0.0048")
    assert provider.calls == 1


async def test_generate_metadata_retries_then_succeeds(make_niche_config) -> None:
    config = make_niche_config()
    bad = {**_valid(config), "description": "too short"}
    provider = _FakeAnthropic([bad, _valid(config)])
    metadata, _ = await generate_metadata(provider, config, book_id=uuid4())  # type: ignore[arg-type]
    assert metadata.title
    assert provider.calls == 2


async def test_generate_metadata_fails_after_max_attempts(make_niche_config) -> None:
    config = make_niche_config()
    bad = {**_valid(config), "keywords": ["only one keyword"]}
    provider = _FakeAnthropic([bad, bad, bad])
    with pytest.raises(MetadataValidationError):
        await generate_metadata(provider, config, book_id=uuid4())  # type: ignore[arg-type]
    assert provider.calls == 3


# --- refactor regression -----------------------------------------------------


def test_cozy_dogs_prompt_unchanged_after_refactor() -> None:
    """Snapshot test: the coloring prompt for cozy_dogs must not drift.

    Guardrail #1 from the puzzle-pipeline plan: after the step-2 discriminated-
    union refactor and the step-7 metadata parameterisation, the prompt built
    for a coloring book with default args must be byte-identical to the
    pre-refactor output. The SHA256 below was captured immediately before
    step 2 was committed.
    """
    import hashlib

    config = load_niche_config(_NICHES_DIR / "cozy_dogs_v1.yaml")
    _system, user = build_metadata_prompt(config, book_id=_SNAPSHOT_BOOK_ID)
    digest = hashlib.sha256(user.encode()).hexdigest()
    assert digest == "805cd480f0b3d24e3402e69f837a8e5b58f99b89ddcbe8d72368970e9f4d7445", (
        "cozy_dogs metadata prompt drifted from the pre-refactor snapshot — "
        "either restore the wording or update the snapshot intentionally"
    )


# --- checklist ---------------------------------------------------------------


def test_build_checklist_contains_listing_details(make_niche_config, make_book) -> None:
    config = make_niche_config()
    book = make_book(slug="m_v1")
    metadata = _book_metadata(config)

    text = build_checklist(book, config, metadata, interior_page_count=config.book.page_count + 2)

    assert "A Splendid Title" in text
    assert "keyword 0" in text
    assert config.metadata.author in text
    assert "set-asin m_v1" in text


# --- low-content classification ----------------------------------------------


def test_build_checklist_marks_low_content_no_by_default(make_niche_config, make_book) -> None:
    config = make_niche_config()
    book = make_book(slug="m_v1")
    text = build_checklist(book, config, _book_metadata(config), interior_page_count=26)
    assert "Low-content book**: No" in text
    assert "Low-content book**: Yes" not in text


def test_build_checklist_marks_low_content_yes_when_niche_sets_it(
    make_niche_config, make_book
) -> None:
    base = make_niche_config()
    config = base.model_copy(
        update={"metadata": base.metadata.model_copy(update={"low_content": True})}
    )
    book = make_book(slug="m_v1")
    text = build_checklist(book, config, _book_metadata(config), interior_page_count=26)
    assert "Low-content book**: Yes" in text


def test_persist_metadata_writes_low_content_into_json(
    tmp_path, make_niche_config, make_book, monkeypatch
) -> None:
    # Stub update_metadata so the test never touches the database.
    monkeypatch.setattr("src.generators.metadata.update_metadata", lambda *a, **k: None)
    config = make_niche_config()
    book = make_book(slug="m_v1")
    persist_metadata(book, config, _book_metadata(config), output_dir=tmp_path)
    payload = json.loads((tmp_path / "m_v1" / "metadata.json").read_text(encoding="utf-8"))
    assert payload["low_content"] is False


# --- imprint flows through to checklist + metadata.json ----------------------


def test_build_checklist_surfaces_default_imprint_publisher(make_niche_config, make_book) -> None:
    config = make_niche_config()
    book = make_book(slug="m_v1")
    text = build_checklist(book, config, _book_metadata(config), interior_page_count=26)
    assert "Publisher**: Quiet Hours Press" in text
    assert "Imprint**: Quiet Hours Press" in text


def test_build_checklist_surfaces_pawpress_publisher_when_set(make_niche_config, make_book) -> None:
    base = make_niche_config()
    config = base.model_copy(update={"imprint": "pawpress"})
    book = make_book(slug="m_v1")
    text = build_checklist(book, config, _book_metadata(config), interior_page_count=26)
    assert "Publisher**: PawPress Books" in text
    assert "Imprint**: PawPress" in text


def test_persist_metadata_writes_publisher_and_imprint_into_json(
    tmp_path, make_niche_config, make_book, monkeypatch
) -> None:
    monkeypatch.setattr("src.generators.metadata.update_metadata", lambda *a, **k: None)
    base = make_niche_config()
    config = base.model_copy(update={"imprint": "pawpress"})
    book = make_book(slug="m_v1")
    persist_metadata(book, config, _book_metadata(config), output_dir=tmp_path)
    payload = json.loads((tmp_path / "m_v1" / "metadata.json").read_text(encoding="utf-8"))
    assert payload["publisher"] == "PawPress Books"
    assert payload["imprint"] == "PawPress"


# --- description template rotation -------------------------------------------


def _uuid_for_prefix(prefix: str) -> UUID:
    """A UUID whose hex[:8] is `prefix * 8` — used to drive pick_template."""
    return UUID(prefix * 8 + "-0000-0000-0000-000000000000")


def test_pick_template_is_deterministic_per_book_id() -> None:
    bid = uuid4()
    assert pick_template(bid).id == pick_template(bid).id


def test_pick_template_rotates_across_books_to_cover_all_templates() -> None:
    # 6 UUIDs whose top-8-hex bytes mod 6 hit {0,5,4,3,2,1} → all six templates.
    ids = {pick_template(_uuid_for_prefix(p)).id for p in "012345"}
    assert ids == {t.id for t in load_description_templates()}


def test_skeleton_ngram_overlap_verbatim_copy_is_one() -> None:
    text = "this is a test skeleton with enough words to form several ngrams"
    assert skeleton_ngram_overlap(text, text) == pytest.approx(1.0)


def test_skeleton_ngram_overlap_unrelated_prose_is_zero() -> None:
    skeleton = "foxes and rabbits dance gently under the silver moonlight"
    description = "Mathematics describes invariants across transformations using symbolic notation."
    assert skeleton_ngram_overlap(description, skeleton) == 0.0


def test_skeleton_ngram_overlap_partial_copy_is_intermediate() -> None:
    skeleton = "alpha beta gamma delta epsilon zeta eta theta"
    description = "alpha beta gamma delta unrelated continuation new words follow"
    overlap = skeleton_ngram_overlap(description, skeleton)
    assert 0.0 < overlap < 1.0


def test_build_metadata_prompt_includes_the_rotated_template_skeleton(
    make_niche_config,
) -> None:
    config = make_niche_config()
    bid = _uuid_for_prefix("0")
    _, user = build_metadata_prompt(config, book_id=bid)
    chosen = pick_template(bid)
    # The first non-empty line of the skeleton must appear verbatim in the prompt.
    first_line = next(line for line in chosen.skeleton.splitlines() if line.strip())
    assert first_line in user
    assert "TEMPLATE TO ADAPT" in user


async def test_generate_metadata_records_template_id_and_overlap_on_payload(
    tmp_path, make_niche_config, make_book, monkeypatch
) -> None:
    monkeypatch.setattr("src.generators.metadata.update_metadata", lambda *a, **k: None)
    config = make_niche_config()
    book = make_book(slug="m_v1")
    provider = _FakeAnthropic([_valid(config)])
    metadata, _ = await generate_metadata(provider, config, book_id=book.id)  # type: ignore[arg-type]
    persist_metadata(book, config, metadata, output_dir=tmp_path)
    payload = json.loads((tmp_path / "m_v1" / "metadata.json").read_text(encoding="utf-8"))
    assert payload["description_template_id"] == pick_template(book.id).id
    assert 0.0 <= payload["description_skeleton_overlap"] <= 1.0


async def test_generate_metadata_warns_on_high_skeleton_overlap(
    make_niche_config, make_book, monkeypatch
) -> None:
    # Force a known long-form skeleton so the test can construct a description
    # with overlap > 0.60. Distinct tokens → many unique n-grams; description ==
    # skeleton verbatim → ratio ≈ 1.0. The warning goes through loguru, which
    # bypasses pytest's caplog/capfd — install a temporary sink to capture it.
    from loguru import logger as loguru_logger
    from src.config.description_templates import DescriptionTemplate

    long_skeleton = " ".join(f"word{i}" for i in range(280))[:1900]
    fake = DescriptionTemplate(id="fake_test", skeleton=long_skeleton)
    monkeypatch.setattr("src.generators.metadata.pick_template", lambda _bid: fake)

    captured: list[str] = []
    sink_id = loguru_logger.add(lambda msg: captured.append(str(msg)), level="WARNING")
    try:
        config = make_niche_config()
        book = make_book(slug="m_v1")
        response = {**_valid(config), "description": long_skeleton}
        provider = _FakeAnthropic([response])
        metadata, _ = await generate_metadata(provider, config, book_id=book.id)  # type: ignore[arg-type]
    finally:
        loguru_logger.remove(sink_id)

    assert metadata.description_skeleton_overlap > 0.60
    assert any("review manually" in msg for msg in captured)
    assert any("fake_test" in msg for msg in captured)
