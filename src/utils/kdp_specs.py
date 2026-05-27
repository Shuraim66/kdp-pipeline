"""KDP print-spec geometry — page sizes, bleed, and safe margins.

All measurements are returned in PDF points (72 per inch). Phase 7 uses the
interior layout; Phase 8 extends this module with cover geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.schema import NicheConfig

POINTS_PER_INCH = 72.0

# KDP paperback interior: 0.125" bleed, 0.25" safe margin, 300 DPI minimum.
BLEED_IN = 0.125
SAFE_MARGIN_IN = 0.25
# Text needs more clearance than images: KDP's Print Previewer flags text
# closer than 0.375" to the trim edge on a bleed PDF. 0.5" is the safer target
# applied to front-matter text (title / copyright pages).
TEXT_SAFE_MARGIN_IN = 0.5
INTERIOR_MIN_DPI = 300


def parse_trim_size(trim_size: str) -> tuple[float, float]:
    """Parse a trim string like ``'8.5x8.5'`` into ``(width_in, height_in)``."""
    width_str, height_str = trim_size.lower().split("x")
    return float(width_str), float(height_str)


@dataclass(frozen=True, slots=True)
class InteriorLayout:
    """Interior-page geometry, in PDF points, for one trim size.

    `page_*` is the full bleed page; `trim_*` is the cut size; `safe_*` is the
    live area an image must stay within (trim minus the 0.25" safe margin);
    `text_safe_*` is the tighter area front-matter text must stay within (trim
    minus the 0.5" text margin — KDP flags text nearer the trim edge).
    """

    page_width: float
    page_height: float
    trim_width: float
    trim_height: float
    safe_width: float
    safe_height: float
    text_safe_width: float
    text_safe_height: float


def interior_layout(trim_size: str) -> InteriorLayout:
    """Compute interior-page geometry — v1 applies bleed on all four sides."""
    trim_w_in, trim_h_in = parse_trim_size(trim_size)
    bleed = BLEED_IN * POINTS_PER_INCH
    safe = SAFE_MARGIN_IN * POINTS_PER_INCH
    text_safe = TEXT_SAFE_MARGIN_IN * POINTS_PER_INCH
    trim_w = trim_w_in * POINTS_PER_INCH
    trim_h = trim_h_in * POINTS_PER_INCH
    return InteriorLayout(
        page_width=trim_w + 2 * bleed,
        page_height=trim_h + 2 * bleed,
        trim_width=trim_w,
        trim_height=trim_h,
        safe_width=trim_w - 2 * safe,
        safe_height=trim_h - 2 * safe,
        text_safe_width=trim_w - 2 * text_safe,
        text_safe_height=trim_h - 2 * text_safe,
    )


# KDP wrap-cover spine factor — inches of spine per interior page, by stock.
_SPINE_FACTOR_PER_PAGE = {"white": 0.002252, "cream": 0.0025, "color": 0.002347}
COVER_DPI = 300


@dataclass(frozen=True, slots=True)
class CoverDimensions:
    """Full wrap-cover geometry: back + spine + front, with bleed."""

    total_width_in: float
    total_height_in: float
    spine_width_in: float
    bleed_in: float
    total_width_px: int
    total_height_px: int


def compute_cover_dimensions(
    page_count: int,
    trim_w_in: float,
    trim_h_in: float,
    *,
    paper: str = "white",
    dpi: int = COVER_DPI,
) -> CoverDimensions:
    """Compute KDP wrap-cover dimensions for a paperback.

    `page_count` is the *interior* page count — the spine widens with it, by a
    per-page factor that depends on the paper stock. Bleed (0.125") is added on
    all four sides. Verify against KDP's cover calculator before a print run.
    """
    if paper not in _SPINE_FACTOR_PER_PAGE:
        raise ValueError(f"unknown paper stock: {paper!r}")
    spine = page_count * _SPINE_FACTOR_PER_PAGE[paper]
    total_w = trim_w_in * 2 + spine + BLEED_IN * 2
    total_h = trim_h_in + BLEED_IN * 2
    return CoverDimensions(
        total_width_in=total_w,
        total_height_in=total_h,
        spine_width_in=spine,
        bleed_in=BLEED_IN,
        total_width_px=round(total_w * dpi),
        total_height_px=round(total_h * dpi),
    )


def total_interior_pages(config: NicheConfig) -> int:
    """Actual rendered interior page count — feeds the cover spine formula.

    ``book.page_count`` counts only the content pages (designs for coloring;
    mazes + solutions for puzzle). Front-matter pages — title, copyright,
    intro, and the solutions divider — are added here. The spine factor in
    `compute_cover_dimensions` multiplies the total to size the cover; any
    drift between ``book.page_count`` and the assembled PDF would print a
    misaligned spine.

    Coloring: ``book.page_count + 2`` (title + copyright). Identical to the
    long-standing hard-coded formula at the prior +2 call sites — coloring
    books produce the same number they did before.

    Puzzle: ``book.page_count + (1 if title_page) + 1 (copyright)
    + (1 if intro_page) + (1 if include_solutions and solutions_section=='end')``.
    """
    coloring = config.coloring
    if coloring is not None:
        return config.book.page_count + 2  # title + copyright
    puzzle_body = config.require_puzzle()
    extras = 1  # copyright always
    if puzzle_body.front_matter.title_page:
        extras += 1
    if puzzle_body.front_matter.intro_page:
        extras += 1
    if (
        puzzle_body.puzzle.include_solutions
        and puzzle_body.puzzle.solutions_section == "end"
    ):
        extras += 1  # "Solutions" divider page before the solution pages
    return config.book.page_count + extras
