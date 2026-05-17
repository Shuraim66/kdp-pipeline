"""KDP print-spec geometry — page sizes, bleed, and safe margins.

All measurements are returned in PDF points (72 per inch). Phase 7 uses the
interior layout; Phase 8 extends this module with cover geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

POINTS_PER_INCH = 72.0

# KDP paperback interior: 0.125" bleed, 0.25" safe margin, 300 DPI minimum.
BLEED_IN = 0.125
SAFE_MARGIN_IN = 0.25
INTERIOR_MIN_DPI = 300


def parse_trim_size(trim_size: str) -> tuple[float, float]:
    """Parse a trim string like ``'8.5x8.5'`` into ``(width_in, height_in)``."""
    width_str, height_str = trim_size.lower().split("x")
    return float(width_str), float(height_str)


@dataclass(frozen=True, slots=True)
class InteriorLayout:
    """Interior-page geometry, in PDF points, for one trim size.

    `page_*` is the full bleed page; `trim_*` is the cut size; `safe_*` is the
    live area an image or text must stay within (trim minus the safe margin).
    """

    page_width: float
    page_height: float
    trim_width: float
    trim_height: float
    safe_width: float
    safe_height: float


def interior_layout(trim_size: str) -> InteriorLayout:
    """Compute interior-page geometry — v1 applies bleed on all four sides."""
    trim_w_in, trim_h_in = parse_trim_size(trim_size)
    bleed = BLEED_IN * POINTS_PER_INCH
    safe = SAFE_MARGIN_IN * POINTS_PER_INCH
    trim_w = trim_w_in * POINTS_PER_INCH
    trim_h = trim_h_in * POINTS_PER_INCH
    return InteriorLayout(
        page_width=trim_w + 2 * bleed,
        page_height=trim_h + 2 * bleed,
        trim_width=trim_w,
        trim_height=trim_h,
        safe_width=trim_w - 2 * safe,
        safe_height=trim_h - 2 * safe,
    )
