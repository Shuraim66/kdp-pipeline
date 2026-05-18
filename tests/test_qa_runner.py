"""Tests for the QA run — evaluation, regeneration, filtered output, review.

The DB is replaced by an in-memory `_Store`; `_FakeFal` returns canned PNG
bytes so regeneration writes real, re-evaluable files without a network call.
"""

from __future__ import annotations

import io
import json
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock
from uuid import UUID, uuid4

import numpy as np
from PIL import Image as PILImage
from src.db.models import Book, Image, ImageQAStatus
from src.db.models import BookStatus as BS
from src.generators.images import expand_slots
from src.providers.anthropic import AnthropicResult
from src.providers.fal import FalImageResult
from src.qa.runner import (
    apply_manual_verdict,
    build_review_html,
    copy_filtered_images,
    run_qa,
)

_SLUG = "qa_test_v1"


def _encode(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    PILImage.fromarray(array, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _clean_png(width: int, height: int) -> bytes:
    """A white page with a small centred black bar — passes QA.

    Kept well under the 10% ink ceiling so it still clears the white-pct gate
    after line-art normalisation dilates the strokes during regeneration.
    """
    array = np.full((height, width, 3), 255, dtype=np.uint8)
    cy, cx, half = height // 2, width // 2, width // 4
    array[cy - 1 : cy + 1, cx - half : cx + half] = 0
    return _encode(array)


def _blank_png(width: int, height: int) -> bytes:
    """An all-white page — fails QA as blank (no line art)."""
    return _encode(np.full((height, width, 3), 255, dtype=np.uint8))


class _FakeFal:
    """A stand-in FalProvider returning fixed PNG bytes for every render."""

    def __init__(self, image_bytes: bytes) -> None:
        self._bytes = image_bytes

    async def generate_image(
        self,
        *,
        prompt: str,
        model: str,
        width: int,
        height: int,
        num_inference_steps: int,
        seed: int | None = None,
        guidance_scale: float | None = None,
        negative_prompt: str | None = None,
        loras: object = None,
        book_id: object = None,
        image_id: object = None,
    ) -> FalImageResult:
        return FalImageResult(
            image_bytes=self._bytes,
            seed=seed if seed is not None else 99,
            width=width,
            height=height,
            duration_s=0.01,
            cost_usd=Decimal("0.003"),
        )


class _Store:
    """An in-memory stand-in for the images table."""

    def __init__(self, images: list[Image]) -> None:
        self._by_id: dict[UUID, Image] = {image.id: image for image in images}

    def list_images(self, _book_id: UUID) -> list[Image]:
        return sorted(self._by_id.values(), key=lambda i: (i.sequence_num, i.retry_attempt))

    def update_qa(self, image_id: UUID, status: ImageQAStatus, metrics: object = None) -> None:
        self._by_id[image_id] = replace(self._by_id[image_id], qa_status=status, qa_metrics=metrics)

    def update_qa_status(self, image_id: UUID, status: ImageQAStatus) -> None:
        self._by_id[image_id] = replace(self._by_id[image_id], qa_status=status)

    def update_vision_qa(
        self,
        image_id: UUID,
        *,
        score: int,
        subscores: dict[str, int],
        issues: list[str],
        prompt_hint: str | None,
        model: str,
        cost_usd: Decimal,
    ) -> None:
        self._by_id[image_id] = replace(
            self._by_id[image_id],
            vision_qa_score=score,
            vision_qa_subscores=subscores,
            vision_qa_issues=issues,
            vision_qa_prompt_hint=prompt_hint,
            vision_qa_model=model,
            vision_qa_cost_usd=cost_usd,
        )

    def create_image(
        self,
        book_id: UUID,
        sequence_num: int,
        prompt: str,
        seed: int,
        model: str,
        generation_params: dict[str, object],
        **kwargs: object,
    ) -> Image:
        image = Image(
            id=uuid4(),
            book_id=book_id,
            sequence_num=sequence_num,
            prompt=prompt,
            negative_prompt=kwargs.get("negative_prompt"),  # type: ignore[arg-type]
            seed=seed,
            model=model,
            generation_params=generation_params,
            local_path=kwargs.get("local_path"),  # type: ignore[arg-type]
            fal_url=kwargs.get("fal_url"),  # type: ignore[arg-type]
            file_sha256=kwargs.get("file_sha256"),  # type: ignore[arg-type]
            qa_status=ImageQAStatus.PENDING,
            qa_metrics=None,
            qa_checked_at=None,
            cost_usd=Decimal(0),
            retry_of_image_id=kwargs.get("retry_of_image_id"),  # type: ignore[arg-type]
            retry_attempt=int(kwargs.get("retry_attempt", 0)),  # type: ignore[arg-type]
            created_at=datetime(2026, 5, 17, tzinfo=UTC),
        )
        self._by_id[image.id] = image
        return image


def _install_store(monkeypatch, store: _Store) -> None:
    monkeypatch.setattr("src.qa.runner.list_images", store.list_images)
    monkeypatch.setattr("src.qa.runner.update_qa", store.update_qa)
    monkeypatch.setattr("src.qa.runner.update_qa_status", store.update_qa_status)
    monkeypatch.setattr("src.qa.runner.update_vision_qa", store.update_vision_qa)
    monkeypatch.setattr("src.generators.images.create_image", store.create_image)


def _vision_reply(*, passed: bool) -> str:
    """A canned Vision QA JSON reply — a clear pass or a clear sub-threshold fail."""
    if passed:
        subscores = {
            "subject_recognition": 38,
            "composition_coherence": 23,
            "line_quality": 14,
            "anatomy_accuracy": 9,
            "whitespace_balance": 9,
        }
    else:
        subscores = {
            "subject_recognition": 18,
            "composition_coherence": 15,
            "line_quality": 10,
            "anatomy_accuracy": 7,
            "whitespace_balance": 7,
        }
    return json.dumps(
        {
            **subscores,
            "total_score": sum(subscores.values()),
            "red_flags": [],
            "primary_failure": "none" if passed else "subject",
            "issues": [] if passed else ["subject not recognizable"],
            "prompt_hint": None,
            "passed": passed,
        }
    )


class _FakeAnthropic:
    """A stand-in AnthropicProvider for Vision QA — optionally fails call #1."""

    def __init__(self, *, fail_first: bool = False) -> None:
        self._fail_first = fail_first
        self.calls = 0

    async def generate_vision(
        self,
        *,
        prompt: str,
        image_bytes: bytes,
        system: str | None = None,
        max_tokens: int = 1024,
        operation: str = "generate_vision",
        model: str | None = None,
        book_id: object = None,
    ) -> AnthropicResult:
        self.calls += 1
        passed = not (self._fail_first and self.calls == 1)
        return AnthropicResult(
            text=_vision_reply(passed=passed),
            input_tokens=2000,
            output_tokens=300,
            cost_usd=Decimal("0.012"),
            duration_s=0.5,
            stop_reason="end_turn",
        )


def _seed_images(config, book: Book, output_dir: Path, failing, make_image) -> list[Image]:
    """Write each slot's initial PNG to disk and build its pending Image row."""
    width, height = config.generation.image_dimensions
    raw_dir = output_dir / book.slug / "images" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    images: list[Image] = []
    for slot in expand_slots(config):
        png = (
            _blank_png(width, height) if slot.sequence_num in failing else _clean_png(width, height)
        )
        path = raw_dir / f"{slot.sequence_num:03d}_attempt0.png"
        path.write_bytes(png)
        images.append(
            make_image(
                book_id=book.id,
                sequence_num=slot.sequence_num,
                qa_status=ImageQAStatus.PENDING,
                retry_attempt=0,
                local_path=str(path),
            )
        )
    return images


# --- run_qa ------------------------------------------------------------------


def test_run_qa_all_pass_marks_done(
    make_niche_config, make_book, make_image, tmp_path, monkeypatch
) -> None:
    config = make_niche_config()
    book = make_book(slug=_SLUG, status=BS.GENERATION_DONE)
    store = _Store(_seed_images(config, book, tmp_path, frozenset(), make_image))
    _install_store(monkeypatch, store)
    transition = Mock()
    fail = Mock()
    monkeypatch.setattr("src.qa.runner.transition_status", transition)
    monkeypatch.setattr("src.qa.runner.fail_book", fail)

    def _no_provider() -> object:
        raise AssertionError("provider must not be built when nothing is rejected")

    report = run_qa(book, config, provider_factory=_no_provider, output_dir=tmp_path)

    assert report.final_status == BS.QA_DONE
    assert report.regenerated == 0
    fail.assert_not_called()
    assert [c.args[1] for c in transition.call_args_list] == [
        BS.QA_RUNNING,
        BS.QA_DONE,
    ]
    filtered = tmp_path / _SLUG / "images" / "filtered"
    assert len(list(filtered.iterdir())) == config.book.page_count


def test_run_qa_regenerates_rejected_then_passes(
    make_niche_config, make_book, make_image, tmp_path, monkeypatch
) -> None:
    config = make_niche_config()
    width, height = config.generation.image_dimensions
    book = make_book(slug=_SLUG, status=BS.GENERATION_DONE)
    store = _Store(_seed_images(config, book, tmp_path, frozenset({0}), make_image))
    _install_store(monkeypatch, store)
    transition = Mock()
    fail = Mock()
    monkeypatch.setattr("src.qa.runner.transition_status", transition)
    monkeypatch.setattr("src.qa.runner.fail_book", fail)
    fal = _FakeFal(_clean_png(width, height))

    report = run_qa(book, config, provider_factory=lambda: fal, output_dir=tmp_path)

    assert report.final_status == BS.QA_DONE
    assert report.regenerated == 1
    fail.assert_not_called()
    filtered = tmp_path / _SLUG / "images" / "filtered"
    assert len(list(filtered.iterdir())) == config.book.page_count


def test_run_qa_fails_book_when_slot_exhausts_retries(
    make_niche_config, make_book, make_image, tmp_path, monkeypatch
) -> None:
    config = make_niche_config(max_retries=2)
    width, height = config.generation.image_dimensions
    book = make_book(slug=_SLUG, status=BS.GENERATION_DONE)
    store = _Store(_seed_images(config, book, tmp_path, frozenset({0}), make_image))
    _install_store(monkeypatch, store)
    transition = Mock()
    fail = Mock()
    monkeypatch.setattr("src.qa.runner.transition_status", transition)
    monkeypatch.setattr("src.qa.runner.fail_book", fail)
    fal = _FakeFal(_blank_png(width, height))  # every regeneration is blank

    report = run_qa(book, config, provider_factory=lambda: fal, output_dir=tmp_path)

    assert report.final_status == BS.FAILED
    fail.assert_called_once()
    assert fail.call_args.kwargs["phase"] == "qa"
    # The book never reaches qa_done.
    assert BS.QA_DONE not in [c.args[1] for c in transition.call_args_list]


# --- vision QA ---------------------------------------------------------------


def test_run_qa_vision_grades_every_passing_page(
    make_niche_config, make_book, make_image, tmp_path, monkeypatch
) -> None:
    config = make_niche_config()
    book = make_book(slug=_SLUG, status=BS.GENERATION_DONE)
    store = _Store(_seed_images(config, book, tmp_path, frozenset(), make_image))
    _install_store(monkeypatch, store)
    monkeypatch.setattr("src.qa.runner.transition_status", Mock())
    monkeypatch.setattr("src.qa.runner.fail_book", Mock())
    fake_anthropic = _FakeAnthropic()

    def _no_fal() -> object:
        raise AssertionError("Fal provider must not be built when nothing regenerates")

    report = run_qa(
        book,
        config,
        provider_factory=_no_fal,
        output_dir=tmp_path,
        vision_provider_factory=lambda: fake_anthropic,
    )

    assert report.final_status == BS.QA_DONE
    assert report.vision_rejected == 0
    assert report.regenerated == 0
    # Every pixel-passed page was vision-graded exactly once.
    images = store.list_images(book.id)
    assert all(image.vision_qa_score is not None for image in images)
    assert fake_anthropic.calls == config.book.page_count


def test_run_qa_builds_a_fresh_vision_provider_each_round(
    make_niche_config, make_book, make_image, tmp_path, monkeypatch
) -> None:
    # Regression: run_qa runs each round in its own asyncio.run(), and an
    # AnthropicProvider's semaphore binds permanently to the first event loop
    # it touches. A vision provider built once and reused across rounds
    # deadlocks the second round — so run_qa must build a fresh one per round.
    config = make_niche_config(max_retries=2)
    width, height = config.generation.image_dimensions
    book = make_book(slug=_SLUG, status=BS.GENERATION_DONE)
    # Slot 0 starts blank (fails pixel QA) so the run takes more than one round.
    store = _Store(_seed_images(config, book, tmp_path, frozenset({0}), make_image))
    _install_store(monkeypatch, store)
    monkeypatch.setattr("src.qa.runner.transition_status", Mock())
    monkeypatch.setattr("src.qa.runner.fail_book", Mock())
    fal = _FakeFal(_clean_png(width, height))

    built: list[int] = []

    def _vision_factory() -> _FakeAnthropic:
        provider = _FakeAnthropic()
        built.append(id(provider))
        return provider

    run_qa(
        book,
        config,
        provider_factory=lambda: fal,
        output_dir=tmp_path,
        vision_provider_factory=_vision_factory,
    )

    # The factory was called more than once (up-front check + at least one
    # round) and every provider it returned is a distinct instance.
    assert len(built) >= 2
    assert len(set(built)) == len(built)


def test_run_qa_vision_rejection_triggers_regeneration(
    make_niche_config, make_book, make_image, tmp_path, monkeypatch
) -> None:
    config = make_niche_config()
    width, height = config.generation.image_dimensions
    book = make_book(slug=_SLUG, status=BS.GENERATION_DONE)
    store = _Store(_seed_images(config, book, tmp_path, frozenset(), make_image))
    _install_store(monkeypatch, store)
    monkeypatch.setattr("src.qa.runner.transition_status", Mock())
    fail = Mock()
    monkeypatch.setattr("src.qa.runner.fail_book", fail)
    fal = _FakeFal(_clean_png(width, height))
    # The first vision evaluation fails on subject; the rest — and the
    # regenerated retry — pass.
    fake_anthropic = _FakeAnthropic(fail_first=True)

    report = run_qa(
        book,
        config,
        provider_factory=lambda: fal,
        output_dir=tmp_path,
        vision_provider_factory=lambda: fake_anthropic,
    )

    assert report.final_status == BS.QA_DONE
    assert report.vision_rejected == 1
    assert report.regenerated == 1
    fail.assert_not_called()
    filtered = tmp_path / _SLUG / "images" / "filtered"
    assert len(list(filtered.iterdir())) == config.book.page_count


# --- filtered output, review HTML, manual verdict ----------------------------


def test_copy_filtered_images_copies_only_passed(make_book, make_image, tmp_path) -> None:
    book = make_book(slug=_SLUG)
    raw_dir = tmp_path / _SLUG / "images" / "raw"
    raw_dir.mkdir(parents=True)
    paths = []
    for seq in range(3):
        path = raw_dir / f"{seq:03d}.png"
        path.write_bytes(f"image-{seq}".encode())
        paths.append(path)
    images = [
        make_image(sequence_num=0, qa_status=ImageQAStatus.PASSED, local_path=str(paths[0])),
        make_image(
            sequence_num=1,
            qa_status=ImageQAStatus.REJECTED_GRAY_PCT,
            local_path=str(paths[1]),
        ),
        make_image(sequence_num=2, qa_status=ImageQAStatus.PASSED, local_path=str(paths[2])),
    ]

    count = copy_filtered_images(book, images, tmp_path)

    assert count == 2
    filtered = tmp_path / _SLUG / "images" / "filtered"
    assert sorted(p.name for p in filtered.iterdir()) == ["000.png", "002.png"]


def test_build_review_html_lists_every_image(make_book, make_image, tmp_path) -> None:
    book = make_book(slug=_SLUG)
    images = [
        make_image(sequence_num=0, qa_status=ImageQAStatus.PASSED, local_path="images/raw/000.png"),
        make_image(
            sequence_num=1,
            qa_status=ImageQAStatus.REJECTED_GRAY_PCT,
            local_path="images/raw/001.png",
        ),
    ]

    path = build_review_html(book, images, tmp_path)

    assert path.is_file()
    document = path.read_text(encoding="utf-8")
    assert str(images[0].id) in document
    assert str(images[1].id) in document
    assert "passed" in document
    assert "rejected" in document


def test_apply_manual_verdict_approve(monkeypatch) -> None:
    update = Mock()
    monkeypatch.setattr("src.qa.runner.update_qa", update)
    image_id = uuid4()

    apply_manual_verdict(image_id, approve=True)

    assert update.call_args.args[0] == image_id
    assert update.call_args.args[1] == ImageQAStatus.PASSED


def test_apply_manual_verdict_reject(monkeypatch) -> None:
    update = Mock()
    monkeypatch.setattr("src.qa.runner.update_qa", update)

    apply_manual_verdict(uuid4(), approve=False)

    assert update.call_args.args[1] == ImageQAStatus.REJECTED_MANUAL
