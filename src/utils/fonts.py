"""Font loading for cover composition — imprint-aware, with fallbacks.

Each imprint declares a `cover_font_primary` (title face — Black / Bold /
Regular weights) and a `cover_font_secondary` (body face — Regular / Bold).
This loader maps the family names to TrueType files in ``assets/fonts/``.
Unknown families fall back to the project defaults (Fraunces + Open Sans);
missing files fall back to reportlab's bundled Bitstream Vera so the build
never blocks — both fallbacks emit a logger warning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import reportlab

from src.utils.logging import logger

_ASSETS_FONTS = Path(__file__).resolve().parents[2] / "assets" / "fonts"
_REPORTLAB_FONTS = Path(reportlab.__file__).resolve().parent / "fonts"

# Family name → variant filename in ``assets/fonts/``. Add an entry here when
# shipping a new font (and drop the TTF files alongside). Primary families
# need Black / Bold / Regular weights; secondary families need Regular / Bold.
# Fraunces files are static instances extracted from the variable font (opsz=72,
# SOFT=0, WONK=0, wght 900 / 700 / 400). Open Sans ships as static TrueType.
_PRIMARY_FAMILY_FILES: dict[str, dict[str, str]] = {
    "Fraunces": {
        "title": "Fraunces-Bold.ttf",
        "title_black": "Fraunces-Black.ttf",
        "title_regular": "Fraunces-Regular.ttf",
    },
}

_SECONDARY_FAMILY_FILES: dict[str, dict[str, str]] = {
    "Open Sans": {
        "body": "OpenSans-Regular.ttf",
        "body_bold": "OpenSans-Bold.ttf",
    },
}

_DEFAULT_PRIMARY = "Fraunces"
_DEFAULT_SECONDARY = "Open Sans"

_FALLBACK_REGULAR = _REPORTLAB_FONTS / "Vera.ttf"
_FALLBACK_BOLD = _REPORTLAB_FONTS / "VeraBd.ttf"


@dataclass(frozen=True, slots=True)
class CoverFonts:
    """Resolved font file paths for cover composition."""

    title: Path  # primary Bold — back-panel heading, spine, front-cover badge
    title_black: Path  # primary Black — front-cover headline
    title_regular: Path  # primary Regular — front-cover subtitle
    body: Path  # secondary Regular — back-panel body text
    body_bold: Path  # secondary Bold
    using_fallback: bool  # True when any face fell back to Bitstream Vera


def _resolve_family(
    family: str,
    table: dict[str, dict[str, str]],
    default: str,
    *,
    kind: str,
) -> dict[str, str]:
    """Return the variant→filename map for `family`, falling back when unknown."""
    if family in table:
        return table[family]
    logger.warning(
        "cover {} font {!r} not installed (no entry in assets/fonts/); falling back to {!r}",
        kind,
        family,
        default,
    )
    return table[default]


def load_cover_fonts(
    *,
    primary_family: str | None = None,
    secondary_family: str | None = None,
) -> CoverFonts:
    """Resolve cover fonts for an imprint, falling back to Bitstream Vera if any
    TrueType file is missing on disk.
    """
    primary = primary_family or _DEFAULT_PRIMARY
    secondary = secondary_family or _DEFAULT_SECONDARY
    p_files = _resolve_family(primary, _PRIMARY_FAMILY_FILES, _DEFAULT_PRIMARY, kind="primary")
    s_files = _resolve_family(
        secondary, _SECONDARY_FAMILY_FILES, _DEFAULT_SECONDARY, kind="secondary"
    )

    title = _ASSETS_FONTS / p_files["title"]
    title_black = _ASSETS_FONTS / p_files["title_black"]
    title_regular = _ASSETS_FONTS / p_files["title_regular"]
    body = _ASSETS_FONTS / s_files["body"]
    body_bold = _ASSETS_FONTS / s_files["body_bold"]

    wanted = (title, title_black, title_regular, body, body_bold)
    if all(path.is_file() for path in wanted):
        return CoverFonts(
            title=title,
            title_black=title_black,
            title_regular=title_regular,
            body=body,
            body_bold=body_bold,
            using_fallback=False,
        )
    logger.warning(
        "cover fonts not found in {} — using the Bitstream Vera fallback. "
        "Add {} for the intended cover design.",
        _ASSETS_FONTS,
        ", ".join(path.name for path in wanted),
    )
    return CoverFonts(
        title=title if title.is_file() else _FALLBACK_BOLD,
        title_black=title_black if title_black.is_file() else _FALLBACK_BOLD,
        title_regular=title_regular if title_regular.is_file() else _FALLBACK_REGULAR,
        body=body if body.is_file() else _FALLBACK_REGULAR,
        body_bold=body_bold if body_bold.is_file() else _FALLBACK_BOLD,
        using_fallback=True,
    )
