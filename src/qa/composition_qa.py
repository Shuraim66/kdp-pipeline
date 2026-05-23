"""Composition QA — pixel-level subject-area gate.

Runs after pixel QA returns ``PASSED`` and before Vision QA in
``src/qa/runner.py:_qa_pending``. The rule is simple and cheap: the bounding
box of all non-white pixels (grayscale < 240) must cover at least
``qa.min_subject_area_ratio`` of the inner canvas (full image minus the
QA-configured white margin per side). Pages below the threshold are recorded
as ``REJECTED_COMPOSITION`` and the existing requeue/retry loop regenerates
them — when a niche sets ``qa.composition_retry_prompt_suffix``, that hint is
appended to the regenerated prompt.

The threshold uses a plain ``<240`` cut for v1 simplicity; an adaptive
threshold from ``src/utils/line_art.py`` is a future swap-in for niches whose
ink colour drifts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

from src.config.schema import QASpec
from src.db.models import ImageQAStatus

# Anything strictly darker than this is treated as "ink" / non-background for
# the bounding-box computation. Matches the task spec.
_NON_WHITE_THRESHOLD = 240


@dataclass(frozen=True, slots=True)
class CompositionResult:
    """The composition-QA verdict for one page."""

    passed: bool
    area_ratio: float
    status: ImageQAStatus  # PASSED or REJECTED_COMPOSITION
    reason: str | None  # None when passed; one-line failure description otherwise

    def metrics(self) -> dict[str, Any]:
        out: dict[str, Any] = {"subject_area_ratio": round(self.area_ratio, 4)}
        if self.reason is not None:
            out["composition_reason"] = self.reason
        return out


def compute_subject_area_ratio(image_path: Path, margin_px: int = 0) -> float:
    """Return the bounding-box area of non-white pixels divided by the inner
    canvas area (full image dimensions minus ``margin_px`` per side).

    Returns ``0.0`` for an all-white page (no subject detected).
    """
    with PILImage.open(image_path) as handle:
        gray = np.asarray(handle.convert("L"), dtype=np.uint8)
    ink = np.where(gray < _NON_WHITE_THRESHOLD)
    rows, cols = ink[0], ink[1]
    if rows.size == 0:
        return 0.0
    bbox_h = int(rows.max()) - int(rows.min()) + 1
    bbox_w = int(cols.max()) - int(cols.min()) + 1
    h, w = gray.shape
    inner_h = max(1, h - 2 * margin_px)
    inner_w = max(1, w - 2 * margin_px)
    return float((bbox_h * bbox_w) / (inner_h * inner_w))


def check_composition(image_path: Path, qa: QASpec) -> CompositionResult:
    """Pass when the subject's bbox covers >= `qa.min_subject_area_ratio` of
    the inner canvas; otherwise REJECTED_COMPOSITION with a reason string.
    """
    ratio = compute_subject_area_ratio(image_path, margin_px=qa.required_white_margin_px)
    threshold = qa.min_subject_area_ratio
    if ratio >= threshold:
        return CompositionResult(
            passed=True,
            area_ratio=ratio,
            status=ImageQAStatus.PASSED,
            reason=None,
        )
    reason = (
        f"subject bbox area {ratio:.1%} below minimum {threshold:.0%}"
        if ratio > 0
        else "no subject detected (page is all white)"
    )
    return CompositionResult(
        passed=False,
        area_ratio=ratio,
        status=ImageQAStatus.REJECTED_COMPOSITION,
        reason=reason,
    )
