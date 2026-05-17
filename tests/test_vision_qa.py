"""Tests for the Vision QA stage — scoring, verdict mapping, the summary.

`_FakeAnthropic` stands in for the provider so these run offline: it returns
a canned model reply, and `evaluate_image` does the real parsing on top.
"""

from __future__ import annotations

import io
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from PIL import Image as PILImage
from src.db.models import ImageQAStatus
from src.providers.anthropic import AnthropicResult
from src.qa.vision_qa import (
    EVAL_IMAGE_SIZE,
    VisionQAResult,
    _downsize_for_eval,
    _map_failure_to_status,
    build_vision_qa_summary,
    evaluate_image,
)


def _vision_json(
    *,
    subject: int = 38,
    composition: int = 23,
    lines: int = 14,
    anatomy: int = 9,
    whitespace: int = 9,
    red_flags: list[str] | None = None,
    primary_failure: str = "none",
    issues: list[str] | None = None,
    prompt_hint: str | None = None,
) -> str:
    """A well-formed model reply with the given subscores."""
    payload: dict[str, Any] = {
        "subject_recognition": subject,
        "composition_coherence": composition,
        "line_quality": lines,
        "anatomy_accuracy": anatomy,
        "whitespace_balance": whitespace,
        "total_score": subject + composition + lines + anatomy + whitespace,
        "red_flags": red_flags or [],
        "primary_failure": primary_failure,
        "issues": issues or [],
        "prompt_hint": prompt_hint,
        "passed": not (red_flags or []),
    }
    return json.dumps(payload)


class _FakeAnthropic:
    """A stand-in AnthropicProvider returning one canned reply per call."""

    def __init__(self, *replies: str) -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append({"prompt": prompt, "image_bytes": image_bytes, "model": model})
        text = self._replies[min(len(self.calls) - 1, len(self._replies) - 1)]
        return AnthropicResult(
            text=text,
            input_tokens=2000,
            output_tokens=300,
            cost_usd=Decimal("0.012"),
            duration_s=0.5,
            stop_reason="end_turn",
        )


def _write_png(path: Path, size: int = 600) -> Path:
    """Write a square white PNG with a centred black bar — a stand-in page."""
    image = PILImage.new("L", (size, size), color=255)
    for y in range(size // 3, 2 * size // 3):
        for x in range(size // 2 - 2, size // 2 + 2):
            image.putpixel((x, y), 0)
    image.save(path, format="PNG")
    return path


# --- evaluate_image ----------------------------------------------------------


async def test_evaluate_image_passes_high_score(make_niche_config, tmp_path) -> None:
    path = _write_png(tmp_path / "page.png")
    provider = _FakeAnthropic(_vision_json(subject=40, composition=25, lines=15))

    result = await evaluate_image(
        path,
        subject="a teddy bear",
        sequence_num=3,
        config=make_niche_config(),
        provider=provider,  # type: ignore[arg-type]
    )

    assert isinstance(result, VisionQAResult)
    assert result.score == 98
    assert result.passed is True
    assert result.rejection_reason is None
    assert result.cost_usd == Decimal("0.012")


async def test_evaluate_image_fails_below_threshold(make_niche_config, tmp_path) -> None:
    path = _write_png(tmp_path / "page.png")
    # 20 + 18 + 10 + 7 + 7 = 62, below the default 80 threshold.
    provider = _FakeAnthropic(
        _vision_json(
            subject=20,
            composition=18,
            lines=10,
            anatomy=7,
            whitespace=7,
            primary_failure="subject",
            issues=["subject not recognizable"],
        )
    )

    result = await evaluate_image(
        path,
        subject="a stethoscope",
        sequence_num=1,
        config=make_niche_config(),
        provider=provider,  # type: ignore[arg-type]
    )

    assert result.score == 62
    assert result.passed is False
    assert result.rejection_reason == ImageQAStatus.REJECTED_VISION_SUBJECT
    assert result.issues == ["subject not recognizable"]


async def test_evaluate_image_red_flag_auto_fails_despite_high_score(
    make_niche_config, tmp_path
) -> None:
    path = _write_png(tmp_path / "page.png")
    # Subscores sum well above threshold, but a red flag forces a fail.
    provider = _FakeAnthropic(
        _vision_json(subject=40, composition=25, red_flags=["AI-melt artifacts on the face"])
    )

    result = await evaluate_image(
        path,
        subject="a teddy bear",
        sequence_num=2,
        config=make_niche_config(),
        provider=provider,  # type: ignore[arg-type]
    )

    assert result.score == 97
    assert result.passed is False
    assert result.rejection_reason == ImageQAStatus.REJECTED_VISION_ANATOMY


async def test_evaluate_image_score_is_subscore_sum_not_model_total(
    make_niche_config, tmp_path
) -> None:
    path = _write_png(tmp_path / "page.png")
    # Hand a deliberately wrong total_score; the rubric sum must win.
    payload = json.loads(
        _vision_json(subject=30, composition=20, lines=12, anatomy=8, whitespace=8)
    )
    payload["total_score"] = 999
    provider = _FakeAnthropic(json.dumps(payload))

    result = await evaluate_image(
        path,
        subject="a heart",
        sequence_num=0,
        config=make_niche_config(),
        provider=provider,  # type: ignore[arg-type]
    )

    assert result.score == 78  # 30 + 20 + 12 + 8 + 8


async def test_evaluate_image_strips_code_fence(make_niche_config, tmp_path) -> None:
    path = _write_png(tmp_path / "page.png")
    fenced = f"```json\n{_vision_json()}\n```"
    provider = _FakeAnthropic(fenced)

    result = await evaluate_image(
        path,
        subject="a heart",
        sequence_num=0,
        config=make_niche_config(),
        provider=provider,  # type: ignore[arg-type]
    )

    assert result.passed is True
    assert result.score > 0


async def test_evaluate_image_malformed_json_fails_conservatively(
    make_niche_config, tmp_path
) -> None:
    path = _write_png(tmp_path / "page.png")
    provider = _FakeAnthropic("I'm sorry, I cannot evaluate this image.")

    result = await evaluate_image(
        path,
        subject="a heart",
        sequence_num=0,
        config=make_niche_config(),
        provider=provider,  # type: ignore[arg-type]
    )

    assert result.passed is False
    assert result.score == 0
    assert result.rejection_reason == ImageQAStatus.REJECTED_VISION_LOWSCORE
    assert any("parse_error" in issue for issue in result.issues)


async def test_evaluate_image_threshold_is_inclusive(make_niche_config, tmp_path) -> None:
    path = _write_png(tmp_path / "page.png")
    provider = _FakeAnthropic(
        _vision_json(subject=30, composition=25, lines=15, anatomy=5, whitespace=5)
    )  # exactly 80

    result = await evaluate_image(
        path,
        subject="a heart",
        sequence_num=0,
        config=make_niche_config(),
        provider=provider,  # type: ignore[arg-type]
        pass_threshold=80,
    )

    assert result.score == 80
    assert result.passed is True


# --- _downsize_for_eval ------------------------------------------------------


def test_downsize_for_eval_shrinks_to_eval_size(tmp_path) -> None:
    path = _write_png(tmp_path / "big.png", size=2550)
    png = _downsize_for_eval(path)
    with PILImage.open(io.BytesIO(png)) as image:
        assert max(image.size) == EVAL_IMAGE_SIZE
        assert image.format == "PNG"


# --- _map_failure_to_status --------------------------------------------------


def test_map_failure_to_status_uses_primary_failure() -> None:
    assert _map_failure_to_status("lines", []) == ImageQAStatus.REJECTED_VISION_LINES
    assert _map_failure_to_status("composition", []) == ImageQAStatus.REJECTED_VISION_COMPOSITION
    assert _map_failure_to_status("none", []) == ImageQAStatus.REJECTED_VISION_LOWSCORE


def test_map_failure_to_status_red_flag_anatomy_wins() -> None:
    status = _map_failure_to_status("subject", ["disembodied hand floating near the head"])
    assert status == ImageQAStatus.REJECTED_VISION_ANATOMY


def test_map_failure_to_status_distorted_face_red_flag_is_anatomy() -> None:
    status = _map_failure_to_status("none", ["distorted asymmetric eyes on the bear"])
    assert status == ImageQAStatus.REJECTED_VISION_ANATOMY


def test_map_failure_to_status_subject_red_flag() -> None:
    # A "subject fundamentally different" red flag must land as a subject fault,
    # not the composition default.
    status = _map_failure_to_status(
        "composition", ["subject is fundamentally different from what was requested"]
    )
    assert status == ImageQAStatus.REJECTED_VISION_SUBJECT


# --- build_vision_qa_summary -------------------------------------------------


def test_build_vision_qa_summary_reports_distribution(make_image) -> None:
    images = [
        make_image(sequence_num=0, qa_status=ImageQAStatus.PASSED, vision_qa_score=92),
        make_image(sequence_num=1, qa_status=ImageQAStatus.PASSED, vision_qa_score=76),
        make_image(
            sequence_num=2,
            qa_status=ImageQAStatus.REJECTED_VISION_SUBJECT,
            vision_qa_score=58,
            vision_qa_issues=["subject not recognizable"],
            vision_qa_cost_usd=Decimal("0.012"),
        ),
        make_image(sequence_num=3, qa_status=ImageQAStatus.PENDING),  # not evaluated
    ]

    summary = build_vision_qa_summary("nurses_bold_easy_v1", images, threshold=80)

    assert "Total images evaluated: 3" in summary
    assert "2/3 passed" in summary
    assert "subject not recognizable" in summary
    # The weakest passing page (seq 1, score 76) is listed for manual review.
    assert "seq 001" in summary


def test_build_vision_qa_summary_handles_no_evaluations(make_image) -> None:
    summary = build_vision_qa_summary("empty_v1", [make_image()], threshold=80)
    assert "No images have been vision-evaluated yet" in summary
