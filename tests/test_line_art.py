"""Tests for line-art normalisation — binarise, thicken, upscale."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
from src.utils.line_art import normalize_line_art


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _decode(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as handle:
        return np.asarray(handle.convert("L"))


def test_output_is_target_size() -> None:
    source = Image.new("L", (256, 256), color=255)
    out = normalize_line_art(_png(source), target_size=(800, 600))
    with Image.open(io.BytesIO(out)) as handle:
        assert handle.size == (800, 600)


def test_output_has_no_grey() -> None:
    # A uniformly grey source must come out pure black/white — shading killed.
    source = Image.new("L", (256, 256), color=128)
    arr = _decode(normalize_line_art(_png(source), target_size=(400, 400)))
    assert set(np.unique(arr).tolist()).issubset({0, 255})


def test_blank_white_stays_white() -> None:
    source = Image.new("L", (128, 128), color=255)
    arr = _decode(normalize_line_art(_png(source), target_size=(256, 256)))
    assert bool((arr == 255).all())


def test_thin_line_is_thickened() -> None:
    # A 1-px black line; normalising at 1:1 scale must widen it well beyond 1 px.
    source = Image.new("L", (200, 200), color=255)
    for y in range(200):
        source.putpixel((100, y), 0)
    arr = _decode(normalize_line_art(_png(source), target_size=(200, 200)))
    black_pixels = int((arr == 0).sum())
    # The original line was 200 px of ink; dilation must more than triple it.
    assert black_pixels > 200 * 3


def test_rgb_input_is_accepted() -> None:
    source = Image.new("RGB", (128, 128), color=(255, 255, 255))
    out = normalize_line_art(_png(source), target_size=(256, 256))
    with Image.open(io.BytesIO(out)) as handle:
        assert handle.size == (256, 256)
