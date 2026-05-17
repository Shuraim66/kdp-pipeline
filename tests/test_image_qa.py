"""Tests for the per-image QA metrics — driven by synthetic images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from src.config.schema import QASpec
from src.db.models import ImageQAStatus
from src.qa.image_qa import evaluate_image

_DIM = (64, 64)  # (width, height)


def _qa_spec(dimensions: tuple[int, int] = _DIM) -> QASpec:
    return QASpec(
        min_white_pct=90.0,
        max_gray_pct=3.0,
        required_white_margin_px=4,
        required_dimensions=dimensions,
        max_retries_per_slot=2,
    )


def _save(array: np.ndarray, path: Path) -> Path:
    Image.fromarray(array, mode="RGB").save(path)
    return path


def _white_canvas() -> np.ndarray:
    # numpy arrays are (height, width, channels).
    return np.full((_DIM[1], _DIM[0], 3), 255, dtype=np.uint8)


def test_all_white_is_rejected_as_blank(tmp_path: Path) -> None:
    path = _save(_white_canvas(), tmp_path / "white.png")
    result = evaluate_image(path, _qa_spec())
    assert not result.passed
    assert result.status == ImageQAStatus.REJECTED_WHITE_PCT


def test_all_black_is_rejected(tmp_path: Path) -> None:
    array = np.zeros((_DIM[1], _DIM[0], 3), dtype=np.uint8)
    path = _save(array, tmp_path / "black.png")
    result = evaluate_image(path, _qa_spec())
    assert result.status == ImageQAStatus.REJECTED_WHITE_PCT


def test_grey_shading_is_rejected(tmp_path: Path) -> None:
    # Mostly white, with a ~6% mid-grey patch — white stays above the floor
    # but grey exceeds the 3% ceiling.
    array = _white_canvas()
    array[10:26, 10:26] = 128
    path = _save(array, tmp_path / "grey.png")
    result = evaluate_image(path, _qa_spec())
    assert result.status == ImageQAStatus.REJECTED_GRAY_PCT


def test_clean_line_art_passes(tmp_path: Path) -> None:
    # A thin black bar in the centre — content, clear margins, no grey.
    array = _white_canvas()
    array[30:34, 12:52] = 0
    path = _save(array, tmp_path / "clean.png")
    result = evaluate_image(path, _qa_spec())
    assert result.passed
    assert result.status == ImageQAStatus.PASSED
    assert result.reason is None
    assert not result.edge_margin_violation


def test_edge_touching_content_is_rejected(tmp_path: Path) -> None:
    array = _white_canvas()
    array[28:36, 12:52] = 0  # centre content
    array[0:2, 20:40] = 0  # ink inside the 4px top margin band
    path = _save(array, tmp_path / "edge.png")
    result = evaluate_image(path, _qa_spec())
    assert result.edge_margin_violation
    assert result.status == ImageQAStatus.REJECTED_MARGINS


def test_wrong_dimensions_are_rejected(tmp_path: Path) -> None:
    array = np.full((50, 50, 3), 255, dtype=np.uint8)
    path = _save(array, tmp_path / "small.png")
    result = evaluate_image(path, _qa_spec(dimensions=(64, 64)))
    assert result.status == ImageQAStatus.REJECTED_RESOLUTION
