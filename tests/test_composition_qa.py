"""Tests for the subject-area composition QA gate.

Pure-function tests against synthetic numpy fixtures — no DB, no API. Five
boundary cases the gate must classify correctly, plus the inner-canvas margin
interaction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from src.config.schema import QASpec
from src.db.models import ImageQAStatus
from src.qa.composition_qa import check_composition, compute_subject_area_ratio


def _draw(path: Path, size: int, *, box: tuple[int, int, int, int] | None) -> Path:
    """White canvas `size`x`size`; paint `box` (y0, y1, x0, x1) black if given."""
    arr = np.full((size, size, 3), 255, dtype=np.uint8)
    if box is not None:
        y0, y1, x0, x1 = box
        arr[y0:y1, x0:x1] = 0
    Image.fromarray(arr, mode="RGB").save(path)
    return path


def _qa(*, min_ratio: float = 0.55, margin_px: int = 0) -> QASpec:
    return QASpec.model_validate(
        {
            "min_white_pct": 0.0,
            "max_gray_pct": 100.0,
            "required_white_margin_px": margin_px,
            "required_dimensions": [400, 400],
            "max_retries_per_slot": 3,
            "min_subject_area_ratio": min_ratio,
        }
    )


def test_centred_large_subject_passes(tmp_path: Path) -> None:
    # 350x350 ink box in a 400x400 canvas → bbox area 122500/160000 = 0.7656
    path = _draw(tmp_path / "large.png", 400, box=(25, 375, 25, 375))
    assert compute_subject_area_ratio(path) == pytest.approx(0.766, abs=0.005)
    result = check_composition(path, _qa())
    assert result.passed
    assert result.status == ImageQAStatus.PASSED


def test_centred_tiny_subject_is_rejected(tmp_path: Path) -> None:
    # 100x100 ink box → bbox 10000/160000 = 0.0625, well below 0.55
    path = _draw(tmp_path / "tiny.png", 400, box=(150, 250, 150, 250))
    result = check_composition(path, _qa())
    assert not result.passed
    assert result.status == ImageQAStatus.REJECTED_COMPOSITION
    assert result.reason is not None and "below minimum" in result.reason
    assert result.metrics()["composition_reason"] == result.reason


def test_off_centre_medium_subject_passes_when_bbox_meets_threshold(
    tmp_path: Path,
) -> None:
    # 320x320 ink box at the top-left corner → bbox 102400/160000 = 0.64 ≥ 0.55
    path = _draw(tmp_path / "corner.png", 400, box=(0, 320, 0, 320))
    result = check_composition(path, _qa())
    assert result.passed, result.reason


def test_full_bleed_subject_passes(tmp_path: Path) -> None:
    path = _draw(tmp_path / "full.png", 400, box=(0, 400, 0, 400))
    result = check_composition(path, _qa())
    assert result.passed
    assert result.area_ratio == pytest.approx(1.0)


def test_all_white_page_is_rejected_as_no_subject(tmp_path: Path) -> None:
    path = _draw(tmp_path / "white.png", 400, box=None)
    result = check_composition(path, _qa())
    assert not result.passed
    assert result.area_ratio == 0.0
    assert result.reason is not None and "no subject" in result.reason


def test_margin_inset_shrinks_the_inner_canvas(tmp_path: Path) -> None:
    # Same 350x350 subject; margin_px=50 → inner 300x300 → ratio rises above 1.0.
    path = _draw(tmp_path / "m.png", 400, box=(25, 375, 25, 375))
    assert compute_subject_area_ratio(path, margin_px=0) == pytest.approx(0.766, abs=0.005)
    assert compute_subject_area_ratio(path, margin_px=50) == pytest.approx(1.36, abs=0.01)


def test_qa_spec_rejects_invalid_min_subject_area_ratio() -> None:
    with pytest.raises(ValueError, match="min_subject_area_ratio"):
        QASpec.model_validate(
            {
                "min_white_pct": 90.0,
                "max_gray_pct": 3.0,
                "required_white_margin_px": 4,
                "required_dimensions": [64, 64],
                "max_retries_per_slot": 3,
                "min_subject_area_ratio": 1.5,  # out of range
            }
        )
