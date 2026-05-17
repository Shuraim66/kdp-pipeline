"""Per-image QA metrics for generated coloring-book pages.

A page should be a crisp black line drawing on a clean white field: mostly
white, almost no mid-grey (shading), a clear white margin, and the exact
print resolution. `evaluate_image` measures those properties with numpy and
maps the result to an `ImageQAStatus`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.config.schema import QASpec
from src.db.models import ImageQAStatus

# A page above this much white carries no line art — it is effectively blank.
_BLANK_WHITE_PCT = 99.5


@dataclass(frozen=True, slots=True)
class ImageQAResult:
    """The QA metrics for one image and the verdict they produce."""

    width: int
    height: int
    white_pct: float
    near_black_pct: float
    gray_pct: float
    edge_margin_violation: bool
    status: ImageQAStatus
    reason: str | None

    @property
    def passed(self) -> bool:
        """Whether the image cleared QA."""
        return self.status == ImageQAStatus.PASSED

    def metrics(self) -> dict[str, Any]:
        """The JSON-safe metrics dict stored in `images.qa_metrics`."""
        return {
            "width": self.width,
            "height": self.height,
            "white_pct": round(self.white_pct, 3),
            "near_black_pct": round(self.near_black_pct, 3),
            "gray_pct": round(self.gray_pct, 3),
            "edge_margin_violation": self.edge_margin_violation,
            "reason": self.reason,
        }


def _verdict(
    qa: QASpec,
    width: int,
    height: int,
    white_pct: float,
    gray_pct: float,
    edge_margin_violation: bool,
) -> tuple[ImageQAStatus, str | None]:
    """Map measured metrics to a QA status, returning the first failure."""
    if (width, height) != qa.required_dimensions:
        want = qa.required_dimensions
        return (
            ImageQAStatus.REJECTED_RESOLUTION,
            f"dimensions {width}x{height} != required {want[0]}x{want[1]}",
        )
    if white_pct > _BLANK_WHITE_PCT:
        return (
            ImageQAStatus.REJECTED_WHITE_PCT,
            f"white {white_pct:.1f}% — page is blank, no line art",
        )
    if white_pct < qa.min_white_pct:
        return (
            ImageQAStatus.REJECTED_WHITE_PCT,
            f"white {white_pct:.1f}% < required minimum {qa.min_white_pct}%",
        )
    if gray_pct > qa.max_gray_pct:
        return (
            ImageQAStatus.REJECTED_GRAY_PCT,
            f"grey {gray_pct:.1f}% > allowed maximum {qa.max_gray_pct}% (shading)",
        )
    if edge_margin_violation:
        return (
            ImageQAStatus.REJECTED_MARGINS,
            f"non-white content within {qa.required_white_margin_px}px of an edge",
        )
    return ImageQAStatus.PASSED, None


def evaluate_image(path: Path, qa: QASpec) -> ImageQAResult:
    """Measure a page's QA metrics and decide its `ImageQAStatus`."""
    with Image.open(path) as handle:
        array = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    height = int(array.shape[0])
    width = int(array.shape[1])

    white = np.all(array > 240, axis=2)
    near_black = np.all(array < 40, axis=2)
    gray = np.any((array >= 50) & (array <= 200), axis=2)

    white_pct = float(np.mean(white)) * 100.0
    near_black_pct = float(np.mean(near_black)) * 100.0
    gray_pct = float(np.mean(gray)) * 100.0

    margin = qa.required_white_margin_px
    non_white = np.logical_not(white)
    edge_margin_violation = margin > 0 and bool(
        np.any(non_white[:margin, :])
        or np.any(non_white[-margin:, :])
        or np.any(non_white[:, :margin])
        or np.any(non_white[:, -margin:])
    )

    status, reason = _verdict(qa, width, height, white_pct, gray_pct, edge_margin_violation)
    return ImageQAResult(
        width=width,
        height=height,
        white_pct=white_pct,
        near_black_pct=near_black_pct,
        gray_pct=gray_pct,
        edge_margin_violation=edge_margin_violation,
        status=status,
        reason=reason,
    )
