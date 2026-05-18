"""Line-art normalisation — prepare generated pages for print.

Every generated page must reach the print resolution QA expects (300+ DPI).
Beyond the upscale, how much to "clean" a page depends on the niche's LoRA:
some produce pale, thin strokes that need bolding; others produce
publication-ready line art that any binarise/dilate step only degrades.
`normalize_line_art` therefore has three modes, selected per niche by
`post_process.mode`:

  * ``minimal`` — upscale and whiten the background only. Keeps the model's own
    smooth, antialiased line art. The right choice when the LoRA already draws
    clean pages (e.g. the cottagecore niche); binarising would only jag the
    edges and dilation would over-ink the detail.
  * ``dilate``  — binarise and bold every stroke with a fixed `MinFilter`. For a
    LoRA whose output is uniformly pale and thin.
  * ``auto``    — binarise and bold, but choose the dilation window per image
    from the page's ink density, so detail-rich pages are not over-inked.

``minimal`` is the default; the binarising modes exist for LoRAs that need them.
"""

from __future__ import annotations

import io
from typing import NamedTuple

import numpy as np
from PIL import Image, ImageFilter

# Stroke-dilation windows (the `dilate` / `auto` modes). A `MinFilter` widens
# the model's thin strokes into bold line art — but on a detail-rich page
# (foliage, thatch, dense floral clusters) the same filter merges closely-spaced
# strokes into solid black masses. `auto` therefore picks the window per image
# from ink density (`_dilation_window`): full on a sparse page, gentle on a
# moderately dense one, none once the page is dense enough that widening would
# destroy the white space a colourist needs.
_DILATE_WINDOW_FULL = 5
_DILATE_WINDOW_LIGHT = 3
# Ink-fraction cutoffs for the windows above, measured on the binarised page
# before dilation. Calibrated on 36 nurses + cottagecore test generations (see
# CALIBRATION_NOTES.md): nurses' sparse subjects sit below 7% ink and take the
# full window safely; cottagecore's foliage/thatch pages sit at 9-15% and
# overshoot the QA white-page floor if dilated.
_DENSITY_FULL_DILATE_MAX = 0.07
_DENSITY_LIGHT_DILATE_MAX = 0.09
# Re-binarise cutoff after the upscale interpolation. Its input is an already
# binarised page blurred by LANCZOS, so a fixed cutoff is correct here.
_UPSCALE_CUTOFF = 200
# Ink is anything this many levels darker than the page background.
_BACKGROUND_MARGIN = 7
# The adaptive threshold never drops below this — a page whose corners are
# this dark is a degenerate (near-black) generation, and flooring the
# threshold keeps binarisation bounded so pixel QA can still reject it.
_THRESHOLD_FLOOR = 180
# Edge of each square corner patch sampled to estimate the background.
_CORNER_PATCH = 100


class LineArtResult(NamedTuple):
    """The output of `normalize_line_art`, with an audit trail of how it was made.

    `image_bytes` is the processed page — PNG, ``L`` mode, sized at the target.
    `threshold` is the adaptive ink/white cutoff; `dilate_window` is the
    `MinFilter` window used (1 = none). Both are ``None`` when the mode does
    not use them. All are recorded in generation_params for auditing.
    """

    image_bytes: bytes
    mode: str
    threshold: int | None
    dilate_window: int | None


def _adaptive_threshold(gray: np.ndarray) -> int:
    """An ink threshold relative to the page's own background brightness.

    The background is the mean of four corner patches — robust even when the
    page has a pale interior fill that would skew a whole-image statistic. Ink
    is anything `_BACKGROUND_MARGIN` darker, floored at `_THRESHOLD_FLOOR`.
    """
    c = _CORNER_PATCH
    corners = (gray[:c, :c], gray[:c, -c:], gray[-c:, :c], gray[-c:, -c:])
    background = float(np.mean([float(patch.mean()) for patch in corners]))
    return max(_THRESHOLD_FLOOR, round(background) - _BACKGROUND_MARGIN)


def _binarise(image: Image.Image, threshold: int) -> Image.Image:
    """Flatten to pure black-on-white, dropping antialiasing and any shading."""
    array = np.asarray(image.convert("L"))
    binary = np.where(array < threshold, 0, 255).astype(np.uint8)
    return Image.fromarray(binary, mode="L")


def _ink_fraction(binary: Image.Image) -> float:
    """Fraction of a binarised page that is ink (black)."""
    return float((np.asarray(binary) == 0).mean())


def _dilation_window(ink_fraction: float) -> int:
    """Pick a `MinFilter` dilation window from the page's ink density.

    Returns 5 (full bolding) for a sparse page, 3 (gentle) for a moderately
    dense one, and 1 — no dilation — once the page is dense enough that
    widening its strokes would merge them into uncolourable black masses.
    """
    if ink_fraction < _DENSITY_FULL_DILATE_MAX:
        return _DILATE_WINDOW_FULL
    if ink_fraction < _DENSITY_LIGHT_DILATE_MAX:
        return _DILATE_WINDOW_LIGHT
    return 1


def _to_png(image: Image.Image) -> bytes:
    """Encode an image to PNG bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _normalize_minimal(source: Image.Image, target_size: tuple[int, int]) -> LineArtResult:
    """Upscale and whiten the background — nothing else.

    Keeps the model's own antialiased strokes intact: binarising jags the edges
    and dilation over-inks detail. Only the background (everything at or above
    the adaptive threshold) is forced to pure white, so the page is clean
    without the line art itself being touched.
    """
    threshold = _adaptive_threshold(np.asarray(source))
    upscaled = source.resize(target_size, Image.Resampling.LANCZOS)
    array = np.asarray(upscaled)
    whitened = np.where(array >= threshold, 255, array).astype(np.uint8)
    return LineArtResult(_to_png(Image.fromarray(whitened, mode="L")), "minimal", threshold, None)


def _normalize_binarising(
    source: Image.Image, target_size: tuple[int, int], *, mode: str
) -> LineArtResult:
    """Binarise, bold the strokes, and upscale — the `dilate` / `auto` path.

    `dilate` bolds with a fixed full window; `auto` picks the window per image
    from ink density so detail-rich pages are not merged into solid masses.
    After the upscale the page is re-binarised so the interpolation leaves no
    grey halo.
    """
    threshold = _adaptive_threshold(np.asarray(source))
    cleaned = _binarise(source, threshold)
    window = _DILATE_WINDOW_FULL if mode == "dilate" else _dilation_window(_ink_fraction(cleaned))
    bold = cleaned if window <= 1 else cleaned.filter(ImageFilter.MinFilter(window))
    upscaled = bold.resize(target_size, Image.Resampling.LANCZOS)
    final = _binarise(upscaled, _UPSCALE_CUTOFF)
    return LineArtResult(_to_png(final), mode, threshold, window)


def normalize_line_art(
    image_bytes: bytes, *, target_size: tuple[int, int], mode: str = "minimal"
) -> LineArtResult:
    """Normalise one generated coloring-book page to print resolution.

    `mode` selects the post-process (see the module docstring): ``minimal``
    upscales and whitens only; ``dilate`` and ``auto`` binarise and bold the
    strokes. Returns a `LineArtResult` — PNG bytes of an ``L``-mode image sized
    exactly `target_size`, plus the mode and the threshold / dilation window
    used. Raises `ValueError` on an unknown mode.
    """
    with Image.open(io.BytesIO(image_bytes)) as handle:
        source = handle.convert("L")
    if mode == "minimal":
        return _normalize_minimal(source, target_size)
    if mode in ("dilate", "auto"):
        return _normalize_binarising(source, target_size, mode=mode)
    raise ValueError(f"unknown post-process mode: {mode!r}")
