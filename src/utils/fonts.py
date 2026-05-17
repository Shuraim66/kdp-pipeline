"""Font loading for cover composition — Bebas Neue + Open Sans, with fallback.

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

# Expected filenames in assets/fonts/ — Google Fonts static TrueType files.
_TITLE_FILE = "BebasNeue-Regular.ttf"
_BODY_FILE = "OpenSans-Regular.ttf"
_BODY_BOLD_FILE = "OpenSans-Bold.ttf"

_FALLBACK_REGULAR = _REPORTLAB_FONTS / "Vera.ttf"
_FALLBACK_BOLD = _REPORTLAB_FONTS / "VeraBd.ttf"


@dataclass(frozen=True, slots=True)
class CoverFonts:
    """Resolved font file paths for cover composition."""

    title: Path
    body: Path
    body_bold: Path
    using_fallback: bool


def load_cover_fonts() -> CoverFonts:
    """Resolve cover fonts, falling back to Bitstream Vera if any are missing."""
    title = _ASSETS_FONTS / _TITLE_FILE
    body = _ASSETS_FONTS / _BODY_FILE
    body_bold = _ASSETS_FONTS / _BODY_BOLD_FILE
    if title.is_file() and body.is_file() and body_bold.is_file():
        return CoverFonts(title=title, body=body, body_bold=body_bold, using_fallback=False)
    logger.warning(
        "cover fonts not found in {} — using the Bitstream Vera fallback. "
        "Add {}, {} and {} for the intended cover design.",
        _ASSETS_FONTS,
        _TITLE_FILE,
        _BODY_FILE,
        _BODY_BOLD_FILE,
    )
    return CoverFonts(
        title=title if title.is_file() else _FALLBACK_REGULAR,
        body=body if body.is_file() else _FALLBACK_REGULAR,
        body_bold=body_bold if body_bold.is_file() else _FALLBACK_BOLD,
        using_fallback=True,
    )
