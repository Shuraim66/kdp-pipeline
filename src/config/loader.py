"""Niche config loading: parse YAML, validate strictly, hash."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.config.schema import NicheConfig
from src.utils.hashing import canonical_hash


def load_niche_config(path: str | Path) -> NicheConfig:
    """Parse and strictly validate a niche YAML file into a `NicheConfig`.

    Raises `OSError` if the path cannot be read, `ValueError` on malformed
    YAML or a non-mapping document, and `pydantic.ValidationError` on any
    schema violation.
    """
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"malformed YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("niche config must be a YAML mapping at the top level")
    return NicheConfig.model_validate(data)


def config_to_dict(config: NicheConfig) -> dict[str, Any]:
    """The config as a JSON-safe dict — the form persisted to `books.config`."""
    return config.model_dump(mode="json")


def compute_config_hash(config: NicheConfig) -> str:
    """Deterministic SHA256 of a niche config's canonical JSON form."""
    return canonical_hash(config_to_dict(config))
