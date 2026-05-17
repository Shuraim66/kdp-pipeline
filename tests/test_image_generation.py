"""Tests for the image generation flow — slot planning and execution.

`_FakeFal` stands in for the Fal provider so these run offline; the DB write
(`create_image`) and the status transitions are patched out.
"""

from __future__ import annotations

from unittest.mock import Mock, call
from uuid import uuid4

from src.db.models import BookStatus, ImageQAStatus
from src.generators.images import (
    execute_plan,
    expand_slots,
    plan_generation,
    run_generation,
)
from src.providers.fal import FalImageResult, cost_for_image

_PNG = b"\x89PNG\r\n\x1a\n" + b"fake-image-data"


class _FakeFal:
    """A stand-in FalProvider: returns canned PNG bytes, optionally failing."""

    def __init__(self, *, fail_prompts: frozenset[str] = frozenset()) -> None:
        self._fail = fail_prompts
        self.prompts: list[str] = []

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
        book_id: object = None,
        image_id: object = None,
    ) -> FalImageResult:
        self.prompts.append(prompt)
        if prompt in self._fail:
            raise RuntimeError("simulated fal failure")
        return FalImageResult(
            image_bytes=_PNG,
            seed=seed if seed is not None else 7,
            width=width,
            height=height,
            duration_s=0.01,
            cost_usd=cost_for_image(model, width, height),
        )


# --- slot expansion ----------------------------------------------------------


def test_expand_slots_numbers_sequentially(niche_config) -> None:
    slots = expand_slots(niche_config)
    assert len(slots) == 50
    assert [s.sequence_num for s in slots] == list(range(50))
    # Subject-major ordering: two consecutive variations of one subject.
    assert slots[0].subject == slots[1].subject
    assert (slots[0].variation_idx, slots[1].variation_idx) == (0, 1)
    assert slots[2].subject != slots[0].subject


# --- plan_generation ---------------------------------------------------------


def test_plan_fresh_book_queues_all(niche_config) -> None:
    plan = plan_generation(niche_config, [])
    assert plan.total_slots == 50
    assert len(plan.to_generate) == 50
    assert plan.skipped == 0
    assert plan.exhausted == []
    assert all(p.retry_attempt == 0 for p in plan.to_generate)
    assert all(p.retry_of_image_id is None for p in plan.to_generate)


def test_plan_skips_passed_slots(niche_config, make_image) -> None:
    existing = [
        make_image(sequence_num=0, qa_status=ImageQAStatus.PASSED),
        make_image(sequence_num=7, qa_status=ImageQAStatus.PASSED),
    ]
    plan = plan_generation(niche_config, existing)
    assert plan.skipped == 2
    assert len(plan.to_generate) == 48
    queued = {p.slot.sequence_num for p in plan.to_generate}
    assert 0 not in queued
    assert 7 not in queued


def test_plan_skips_slots_awaiting_qa(niche_config, make_image) -> None:
    plan = plan_generation(
        niche_config, [make_image(sequence_num=3, qa_status=ImageQAStatus.PENDING)]
    )
    assert plan.skipped == 1
    assert len(plan.to_generate) == 49


def test_plan_requeues_rejected_with_retries(niche_config, make_image) -> None:
    rejected = make_image(
        sequence_num=0, qa_status=ImageQAStatus.REJECTED_WHITE_PCT, retry_attempt=0
    )
    plan = plan_generation(niche_config, [rejected])
    queued = {p.slot.sequence_num: p for p in plan.to_generate}
    assert queued[0].retry_attempt == 1
    assert queued[0].retry_of_image_id == rejected.id
    assert plan.exhausted == []


def test_plan_marks_exhausted_slots(niche_config, make_image) -> None:
    max_retries = niche_config.qa.max_retries_per_slot
    rejected = make_image(
        sequence_num=0,
        qa_status=ImageQAStatus.REJECTED_GRAY_PCT,
        retry_attempt=max_retries,
    )
    plan = plan_generation(niche_config, [rejected])
    assert [s.sequence_num for s in plan.exhausted] == [0]
    assert 0 not in {p.slot.sequence_num for p in plan.to_generate}


def test_plan_uses_latest_attempt(niche_config, make_image) -> None:
    # Attempt 0 rejected, attempt 1 passed -> the slot is done.
    book_id = uuid4()
    images = [
        make_image(
            book_id=book_id,
            sequence_num=0,
            retry_attempt=0,
            qa_status=ImageQAStatus.REJECTED_EDGES,
        ),
        make_image(
            book_id=book_id,
            sequence_num=0,
            retry_attempt=1,
            qa_status=ImageQAStatus.PASSED,
        ),
    ]
    plan = plan_generation(niche_config, images)
    assert plan.skipped == 1
    assert 0 not in {p.slot.sequence_num for p in plan.to_generate}


def test_plan_limit_caps_to_generate(niche_config) -> None:
    plan = plan_generation(niche_config, [], limit=5)
    assert len(plan.to_generate) == 5
    assert plan.total_slots == 50


# --- execute_plan ------------------------------------------------------------


async def test_execute_plan_writes_files_and_rows(
    niche_config, make_book, tmp_path, monkeypatch
) -> None:
    create_image = Mock()
    monkeypatch.setattr("src.generators.images.create_image", create_image)
    book = make_book()
    plan = plan_generation(niche_config, [], limit=3)

    report = await execute_plan(
        book, niche_config, plan.to_generate, _FakeFal(), tmp_path, show_progress=False
    )

    assert report.succeeded == 3
    assert report.failures == []
    assert report.total_cost > 0
    assert create_image.call_count == 3
    raw = tmp_path / book.slug / "images" / "raw"
    assert sorted(p.name for p in raw.iterdir()) == [
        "000_attempt0.png",
        "001_attempt0.png",
        "002_attempt0.png",
    ]


async def test_execute_plan_isolates_per_image_failure(
    niche_config, make_book, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("src.generators.images.create_image", Mock())
    book = make_book()
    plan = plan_generation(niche_config, [], limit=3)
    failing_prompt = plan.to_generate[1].prompt
    fake = _FakeFal(fail_prompts=frozenset({failing_prompt}))

    report = await execute_plan(
        book, niche_config, plan.to_generate, fake, tmp_path, show_progress=False
    )

    assert report.succeeded == 2
    assert len(report.failures) == 1
    assert report.failures[0].planned.prompt == failing_prompt
    raw = tmp_path / book.slug / "images" / "raw"
    assert len(list(raw.iterdir())) == 2


# --- run_generation ----------------------------------------------------------


def test_run_generation_marks_done_on_full_success(
    niche_config, make_book, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("src.generators.images.create_image", Mock())
    transition = Mock()
    monkeypatch.setattr("src.generators.images.transition_status", transition)
    book = make_book(status=BookStatus.CREATED)
    plan = plan_generation(niche_config, [])

    report = run_generation(
        book, niche_config, plan, test_mode=False, provider=_FakeFal(), output_dir=tmp_path
    )

    assert report.succeeded == 50
    assert transition.call_args_list == [
        call(book.id, BookStatus.GENERATING),
        call(book.id, BookStatus.GENERATION_DONE),
    ]


def test_run_generation_stays_generating_on_failure(
    niche_config, make_book, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("src.generators.images.create_image", Mock())
    transition = Mock()
    monkeypatch.setattr("src.generators.images.transition_status", transition)
    book = make_book(status=BookStatus.CREATED)
    plan = plan_generation(niche_config, [])
    fake = _FakeFal(fail_prompts=frozenset({plan.to_generate[0].prompt}))

    report = run_generation(
        book, niche_config, plan, test_mode=False, provider=fake, output_dir=tmp_path
    )

    assert len(report.failures) == 1
    # Generation started but is not marked done — a re-run continues.
    assert transition.call_args_list == [call(book.id, BookStatus.GENERATING)]
    assert (tmp_path / book.slug / "failed_prompts.txt").is_file()


def test_run_generation_test_mode_skips_status(
    niche_config, make_book, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("src.generators.images.create_image", Mock())
    transition = Mock()
    monkeypatch.setattr("src.generators.images.transition_status", transition)
    book = make_book(status=BookStatus.CREATED)
    plan = plan_generation(niche_config, [], limit=5)

    report = run_generation(
        book, niche_config, plan, test_mode=True, provider=_FakeFal(), output_dir=tmp_path
    )

    assert report.succeeded == 5
    assert transition.call_count == 0
