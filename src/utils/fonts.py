"""Font loading for cover composition — Fraunces + Open Sans, with fallback.

The intended cover fonts (SIL Open Font License, redistributable) live in
``assets/fonts/``. Until they are added, the loader falls back to reportlab's
bundled Bitstream Vera so the build never blocks — emitting a warning, since
the fallback is not the intended look.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import reportlab

from src.utils.logging import logger

_ASSETS_FONTS = Path(__file__).resolve().parents[2] / "assets" / "fonts"
_REPORTLAB_FONTS = Path(reportlab.__file__).resolve().parent / "fonts"

# Expected filenames in assets/fonts/. Open Sans ships as static TrueType from
# Google Fonts; Fraunces ships only as a variable font, so the Fraunces files
# are static instances extracted from it (opsz=72, SOFT=0, WONK=0, wght 900 /
# 700 / 400). The faces are resolved by these filenames — the niche
# `cover.font_family` field is descriptive only.
_TITLE_FILE = "Fraunces-Bold.ttf"
_TITLE_BLACK_FILE = "Fraunces-Black.ttf"
_TITLE_REGULAR_FILE = "Fraunces-Regular.ttf"
_BODY_FILE = "OpenSans-Regular.ttf"
_BODY_BOLD_FILE = "OpenSans-Bold.ttf"

_FALLBACK_REGULAR = _REPORTLAB_FONTS / "Vera.ttf"
_FALLBACK_BOLD = _REPORTLAB_FONTS / "VeraBd.ttf"


@dataclass(frozen=True, slots=True)
class CoverFonts:
    """Resolved font file paths for cover composition."""

    title: Path  # Fraunces Bold — back-panel heading, spine, front-cover badge
    title_black: Path  # Fraunces Black — front-cover headline
    title_regular: Path  # Fraunces Regular — front-cover subtitle
    body: Path  # Open Sans — back-panel body text
    body_bold: Path
    using_fallback: bool


def load_cover_fonts() -> CoverFonts:
    """Resolve cover fonts, falling back to Bitstream Vera if any are missing."""
    title = _ASSETS_FONTS / _TITLE_FILE
    title_black = _ASSETS_FONTS / _TITLE_BLACK_FILE
    title_regular = _ASSETS_FONTS / _TITLE_REGULAR_FILE
    body = _ASSETS_FONTS / _BODY_FILE
    body_bold = _ASSETS_FONTS / _BODY_BOLD_FILE
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
