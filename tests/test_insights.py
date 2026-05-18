"""Tests for the Q6 feedback-loop reports — pure aggregation, no database."""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal

from src.db.models import Image, ImageQAStatus
from src.qa.insights import (
    build_ink_density_report,
    build_probe_report,
    build_prompt_hint_report,
    build_subject_performance_report,
    rank_hint_terms,
)
from src.qa.vision_qa import VisionQAResult

_FULL_SUBSCORES = {
    "subject_recognition": 38,
    "composition_coherence": 22,
    "line_quality": 12,
    "anatomy_accuracy": 8,
    "whitespace_balance": 4,
}


def _vqa(score: int, *, passed: bool, reason: ImageQAStatus | None = None) -> VisionQAResult:
    return VisionQAResult(
        score=score,
        subscores=_FULL_SUBSCORES,
        passed=passed,
        rejection_reason=reason,
        issues=[],
        prompt_hint=None,
        cost_usd=Decimal("0.004"),
        raw_response="{}",
    )


# --- rank_hint_terms ------------------------------------------------------


def test_rank_hint_terms_surfaces_recurring_themes() -> None:
    hints = [
        "Add a decorative border around the design.",
        "A simple border would frame this better.",
        "Consider a border to balance the empty edges.",
    ]
    ranked = dict(rank_hint_terms(hints, top_n=10))
    assert ranked["border"] == 3  # the recurring content word surfaces


def test_rank_hint_terms_drops_singletons() -> None:
    # "decorative" appears once — a recurring-theme report must not list it.
    ranked = dict(rank_hint_terms(["Add a decorative flourish here."], top_n=10))
    assert ranked == {}


def test_rank_hint_terms_filters_rubric_vocabulary() -> None:
    # "line", "subject" and "composition" recur in both hints — they would be
    # count-2 terms — but they are rubric vocabulary and must be filtered out.
    # "quality" also recurs and, being a real content word, must survive.
    hints = [
        "Improve the line quality and subject composition.",
        "The line composition needs better quality and a clearer subject.",
    ]
    ranked = dict(rank_hint_terms(hints, top_n=10))
    assert "line" not in ranked
    assert "subject" not in ranked
    assert "composition" not in ranked
    assert ranked["quality"] == 2  # a recurring non-rubric word still counts


# --- build_prompt_hint_report ---------------------------------------------


def test_prompt_hint_report_no_evaluated_images(make_image: Callable[..., Image]) -> None:
    report = build_prompt_hint_report("book_v1", [make_image()])
    assert "No images have been vision-evaluated yet." in report


def test_prompt_hint_report_no_hints(make_image: Callable[..., Image]) -> None:
    images = [make_image(vision_qa_score=88, vision_qa_prompt_hint=None)]
    report = build_prompt_hint_report("book_v1", images)
    assert "1 image(s) evaluated · 0 carry a prompt hint" in report
    assert "nothing to aggregate" in report


def test_prompt_hint_report_groups_verbatim_repeats(
    make_image: Callable[..., Image],
) -> None:
    images = [
        make_image(sequence_num=2, vision_qa_score=78, vision_qa_prompt_hint="Bolden the strokes."),
        make_image(sequence_num=8, vision_qa_score=80, vision_qa_prompt_hint="bolden the strokes"),
        make_image(sequence_num=4, vision_qa_score=90, vision_qa_prompt_hint="A unique remark."),
    ]
    report = build_prompt_hint_report("book_v1", images)
    assert "3 image(s) evaluated · 3 carry a prompt hint" in report
    assert '2x  "bolden the strokes"' in report  # case + period normalised together
    assert "pages 002,008" in report
    assert "avg 79" in report  # mean of 78 and 80
    assert "1 further hint(s) occurred once each." in report


# --- build_subject_performance_report -------------------------------------


def test_subject_performance_report_formats_and_picks_top_rejection() -> None:
    perf_rows = [
        {
            "subject": "a cute teacup character",
            "slots": 6,
            "books": 2,
            "avg_score": Decimal("74.2"),
            "pass_pct": Decimal("67"),
            "retry_pct": Decimal("50"),
            "avg_ink_pct": Decimal("11.3"),
        },
        {
            "subject": "a sleepy fox",
            "slots": 6,
            "books": 2,
            "avg_score": Decimal("82.0"),
            "pass_pct": Decimal("100"),
            "retry_pct": Decimal("17"),
            "avg_ink_pct": Decimal("6.0"),
        },
    ]
    rejection_rows = [
        {"subject": "a cute teacup character", "qa_status": "rejected_vision_lines", "n": 3},
        {"subject": "a cute teacup character", "qa_status": "rejected_vision_anatomy", "n": 1},
    ]
    report = build_subject_performance_report("cottagecore", perf_rows, rejection_rows)

    assert "2 subject(s) · 2 book(s) · 12 slot(s)" in report
    assert "vision_lines (3)" in report  # the most common rejection wins
    # worst-scoring subject is listed first.
    assert report.index("a cute teacup character") < report.index("a sleepy fox")


def test_subject_performance_report_handles_missing_score() -> None:
    perf_rows = [
        {
            "subject": "an unevaluated subject",
            "slots": 2,
            "books": 1,
            "avg_score": None,
            "pass_pct": Decimal("0"),
            "retry_pct": Decimal("0"),
            "avg_ink_pct": None,
        }
    ]
    report = build_subject_performance_report("cottagecore", perf_rows, [])
    assert "an unevaluated subject" in report
    assert "—" in report  # null score / ink render as a dash, not a crash


# --- build_ink_density_report ---------------------------------------------


def test_ink_density_report_flags_dense_and_sparse(
    make_image: Callable[..., Image],
) -> None:
    images = [
        make_image(sequence_num=0, ink_density_pct=12.5, generation_params={"subject": "busy"}),
        make_image(sequence_num=1, ink_density_pct=2.0, generation_params={"subject": "thin"}),
        make_image(sequence_num=2, ink_density_pct=5.0, generation_params={"subject": "fine"}),
    ]
    report = build_ink_density_report("book_v1", images, (3.0, 8.0))
    assert "1 dense · 1 sparse · 1 in band" in report
    assert "DENSE" in report and "SPARSE" in report
    assert "seq 000" in report  # the 12.5% page
    assert "seq 001" in report  # the 2.0% page


def test_ink_density_report_uses_latest_attempt(
    make_image: Callable[..., Image],
) -> None:
    # A slot's retry re-rolled it from over-dense into the band — the report
    # must read the latest attempt, not the original.
    images = [
        make_image(sequence_num=0, retry_attempt=0, ink_density_pct=12.0),
        make_image(sequence_num=0, retry_attempt=1, ink_density_pct=5.0),
    ]
    report = build_ink_density_report("book_v1", images, (3.0, 8.0))
    assert "1 page(s) · 0 dense · 0 sparse · 1 in band" in report


def test_ink_density_report_no_measurements(make_image: Callable[..., Image]) -> None:
    report = build_ink_density_report("book_v1", [make_image(ink_density_pct=None)], (3.0, 8.0))
    assert "No pages carry an ink-density measurement yet." in report


# --- build_probe_report ---------------------------------------------------


def test_probe_report_reports_score_spread() -> None:
    results = [_vqa(84, passed=True), _vqa(80, passed=True), _vqa(88, passed=True)]
    report = build_probe_report("page.png", "a cute mushroom", results, threshold=80)
    assert "min 80  max 88  range 8" in report
    assert "3 PASS / 0 reject" in report
    assert "straddles" not in report  # a unanimous verdict is stable


def test_probe_report_flags_threshold_straddle() -> None:
    results = [
        _vqa(84, passed=True),
        _vqa(78, passed=False, reason=ImageQAStatus.REJECTED_VISION_LINES),
    ]
    report = build_probe_report("page.png", "a cute mushroom", results, threshold=80)
    assert "1 PASS / 1 reject" in report
    assert "straddles the threshold (80)" in report
    assert "reject (lines)" in report  # rejection reason is shown per run
