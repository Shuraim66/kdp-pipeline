"""Tests for line-art normalisation — the minimal, dilate, and auto modes."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image
from src.utils.line_art import normalize_line_art


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _from_array(arr: np.ndarray) -> bytes:
    return _png(Image.fromarray(arr.astype(np.uint8), mode="L"))


def _decode(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as handle:
        return np.asarray(handle.convert("L"))


# --- shared behaviour -----------------------------------------------------


def test_output_is_target_size() -> None:
    source = Image.new("L", (256, 256), color=255)
    for mode in ("minimal", "dilate", "auto"):
        result = normalize_line_art(_png(source), target_size=(800, 600), mode=mode)
        with Image.open(io.BytesIO(result.image_bytes)) as handle:
            assert handle.size == (800, 600)


def test_rgb_input_is_accepted() -> None:
    source = Image.new("RGB", (128, 128), color=(255, 255, 255))
    result = normalize_line_art(_png(source), target_size=(256, 256), mode="minimal")
    with Image.open(io.BytesIO(result.image_bytes)) as handle:
        assert handle.size == (256, 256)


def test_blank_white_stays_white() -> None:
    source = Image.new("L", (128, 128), color=255)
    for mode in ("minimal", "dilate", "auto"):
        result = normalize_line_art(_png(source), target_size=(256, 256), mode=mode)
        assert bool((_decode(result.image_bytes) == 255).all())


def test_unknown_mode_is_rejected() -> None:
    source = Image.new("L", (64, 64), color=255)
    with pytest.raises(ValueError, match="unknown post-process mode"):
        normalize_line_art(_png(source), target_size=(64, 64), mode="bogus")


# --- minimal mode ---------------------------------------------------------


def test_minimal_keeps_grey_antialiasing() -> None:
    # minimal must NOT binarise — a mid-grey stroke survives as grey, so the
    # model's own smooth antialiased line art is preserved.
    arr = np.full((128, 128), 255, dtype=np.uint8)
    arr[:, 60:68] = 130  # a mid-grey band, well below the white-clip threshold
    result = normalize_line_art(_from_array(arr), target_size=(128, 128), mode="minimal")
    values = set(np.unique(_decode(result.image_bytes)).tolist())
    assert values - {0, 255}  # grey survived — not flattened to pure black/white


def test_minimal_whitens_near_white_background() -> None:
    # Faint off-white background haze is clipped to pure white.
    arr = np.full((128, 128), 251, dtype=np.uint8)  # near-white, not pure
    arr[:, 60:68] = 0
    result = normalize_line_art(_from_array(arr), target_size=(128, 128), mode="minimal")
    corner = _decode(result.image_bytes)[:20, :20]
    assert bool((corner == 255).all())  # background is now pure white


def test_minimal_does_not_thicken() -> None:
    # minimal applies no dilation — a thin line stays far thinner than dilate's.
    arr = np.full((200, 200), 255, dtype=np.uint8)
    arr[:, 100] = 0
    minimal = normalize_line_art(_from_array(arr), target_size=(200, 200), mode="minimal")
    dilated = normalize_line_art(_from_array(arr), target_size=(200, 200), mode="dilate")
    assert int((_decode(minimal.image_bytes) < 128).sum()) < int(
        (_decode(dilated.image_bytes) == 0).sum()
    )


def test_minimal_reports_mode_and_no_window() -> None:
    source = Image.new("L", (128, 128), color=255)
    result = normalize_line_art(_png(source), target_size=(128, 128), mode="minimal")
    assert result.mode == "minimal"
    assert result.dilate_window is None


# --- dilate / auto modes (binarising) -------------------------------------


def test_dilate_output_has_no_grey() -> None:
    # A uniformly grey source binarises to pure black/white — shading killed.
    source = Image.new("L", (256, 256), color=128)
    result = normalize_line_art(_png(source), target_size=(400, 400), mode="dilate")
    assert set(np.unique(_decode(result.image_bytes)).tolist()).issubset({0, 255})


def test_dilate_thickens_a_thin_line() -> None:
    # A 1-px line; dilation must widen it well beyond 1 px.
    arr = np.full((200, 200), 255, dtype=np.uint8)
    arr[:, 100] = 0
    result = normalize_line_art(_from_array(arr), target_size=(200, 200), mode="dilate")
    # The original line was 200 px of ink; dilation must more than triple it.
    assert int((_decode(result.image_bytes) == 0).sum()) > 200 * 3


def test_pale_lines_are_recovered() -> None:
    # A pale-grey line (~240): the adaptive threshold (background - 7) catches it.
    arr = np.full((300, 300), 255, dtype=np.uint8)
    arr[:, 148:152] = 240
    result = normalize_line_art(_from_array(arr), target_size=(300, 300), mode="auto")
    assert result.threshold is not None and result.threshold > 240
    assert int((_decode(result.image_bytes) == 0).sum()) > 0


def test_threshold_is_floored_on_a_dark_page() -> None:
    # A dark page (corners ~120) must not collapse the threshold; the floor
    # holds it at 180 so pixel QA can still reject the degenerate page.
    source = Image.new("L", (256, 256), color=120)
    result = normalize_line_art(_png(source), target_size=(256, 256), mode="auto")
    assert result.threshold == 180


def test_auto_sparse_page_gets_full_dilation() -> None:
    # A nearly-blank page (one thin line) is sparse — full MinFilter(5) window.
    arr = np.full((200, 200), 255, dtype=np.uint8)
    arr[:, 100] = 0
    result = normalize_line_art(_from_array(arr), target_size=(200, 200), mode="auto")
    assert result.dilate_window == 5


def test_auto_moderately_dense_page_gets_light_dilation() -> None:
    # ~8% ink lands in the middle band — a gentle MinFilter(3).
    arr = np.full((200, 200), 255, dtype=np.uint8)
    arr[:16, :] = 0  # 16 rows of 200 = 8% of the page is ink
    result = normalize_line_art(_from_array(arr), target_size=(200, 200), mode="auto")
    assert result.dilate_window == 3


def test_auto_dense_detailed_page_skips_dilation() -> None:
    # Closely-spaced fine strokes — the pattern dilation destroys by merging
    # into solid black. A dense page must skip dilation (window 1).
    arr = np.full((200, 200), 255, dtype=np.uint8)
    arr[:, ::4] = 0  # a thin vertical line every 4 px — 25% ink
    result = normalize_line_art(_from_array(arr), target_size=(200, 200), mode="auto")
    assert result.dilate_window == 1
    # the stripes stay separate — not merged into a solid black field.
    assert not bool((_decode(result.image_bytes) == 0).all())


def test_dilate_mode_always_uses_full_window() -> None:
    # `dilate` is unconditional — even a dense page gets the full window.
    arr = np.full((200, 200), 255, dtype=np.uint8)
    arr[:, ::4] = 0
    result = normalize_line_art(_from_array(arr), target_size=(200, 200), mode="dilate")
    assert result.dilate_window == 5
