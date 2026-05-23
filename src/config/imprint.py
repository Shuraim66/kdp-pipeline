"""Imprint configuration — publisher identity + cover visual overlay.

An imprint groups books under one publisher label and one visual style
(palette + font families). Each imprint is a strict-validated YAML in
``imprints/<name>.yaml``; niches reference an imprint by name via
``NicheConfig.imprint``. Cover and metadata generation resolve the imprint
at use time.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml

from src.config.schema import _Strict

_IMPRINTS_DIR = Path(__file__).resolve().parents[2] / "imprints"


class VisualStyleSpec(_Strict):
    """The ``visual_style:`` block of an imprint."""

    cover_font_primary: str
    cover_font_secondary: str
    cover_palette_name: str
    back_cover_template: str


class ImprintConfig(_Strict):
    """A publishing imprint — publisher line + visual identity."""

    name: str
    publisher_field: str
    author_default: str
    visual_style: VisualStyleSpec


class ImprintNotFoundError(LookupError):
    """No ``imprints/<name>.yaml`` file on disk."""


@cache
def load_imprint(name: str) -> ImprintConfig:
    """Parse and strictly validate ``imprints/<name>.yaml`` into an ImprintConfig.

    Raises ``ImprintNotFoundError`` when the file is missing, ``ValueError`` on
    malformed YAML or a non-mapping top-level node, and
    ``pydantic.ValidationError`` on any schema violation.
    """
    path = _IMPRINTS_DIR / f"{name}.yaml"
    if not path.is_file():
        raise ImprintNotFoundError(f"no imprint config at {path}")
    text = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"malformed imprint YAML at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"imprint config must be a YAML mapping at top level: {path}")
    return ImprintConfig.model_validate(data)
