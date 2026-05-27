"""Back-compat tests: legacy stored configs + on-disk YAMLs must keep loading.

After the discriminated-union refactor (step 2), `NicheConfig.model_validate`
must accept both:
  (a) legacy flat dicts (no `body:` key; coloring keys at top level) — as
      stored in existing `books.config` JSONB rows;
  (b) the new wrapped form (`body: {kind: ..., ...}`) — produced by
      `config.model_dump()` after the refactor and stored for new books.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from src.config.loader import config_to_dict, load_niche_config
from src.config.schema import ColoringBody, NicheConfig

_NICHES_DIR = Path(__file__).resolve().parent.parent / "niches"

_ALL_COLORING_NICHES = (
    "cottagecore_mushrooms_v1.yaml",
    "cozy_dogs_v1.yaml",
    "nurses_v1.yaml",
    "test_bernese_v1.yaml",
)


@pytest.mark.parametrize("yaml_name", _ALL_COLORING_NICHES)
def test_existing_yaml_validates_unchanged(yaml_name: str) -> None:
    """Every coloring YAML on disk must still load — these are the books that ship."""
    config = load_niche_config(_NICHES_DIR / yaml_name)
    assert isinstance(config.body, ColoringBody)


@pytest.mark.parametrize("yaml_name", _ALL_COLORING_NICHES)
def test_wrapped_dump_round_trips(yaml_name: str) -> None:
    """A config dumped to JSON and re-validated must produce an equivalent NicheConfig."""
    original = load_niche_config(_NICHES_DIR / yaml_name)
    payload = config_to_dict(original)
    # Wrapped form: `body` is the discriminated key, no top-level coloring keys.
    assert "body" in payload
    for k in ("style", "subjects", "qa", "generation"):
        assert k not in payload
    rehydrated = NicheConfig.model_validate(payload)
    assert rehydrated.slug == original.slug
    assert rehydrated.book == original.book
    assert rehydrated.body == original.body


def _legacy_flat_dict_from(yaml_name: str) -> dict[str, Any]:
    """Build the LEGACY flat shape (pre-refactor) from a current YAML on disk."""
    config = load_niche_config(_NICHES_DIR / yaml_name)
    wrapped = config_to_dict(config)
    flat = {k: v for k, v in wrapped.items() if k != "body"}
    coloring_body = wrapped["body"]
    # Strip the discriminator key — the legacy DB rows have no `kind` either.
    flat.update({k: v for k, v in coloring_body.items() if k != "kind"})
    return flat


@pytest.mark.parametrize("yaml_name", _ALL_COLORING_NICHES)
def test_legacy_flat_dict_still_validates(yaml_name: str) -> None:
    """Books rows persisted before this refactor must continue to validate."""
    flat = _legacy_flat_dict_from(yaml_name)
    assert "body" not in flat
    assert "style" in flat
    assert "subjects" in flat
    config = NicheConfig.model_validate(flat)
    assert isinstance(config.body, ColoringBody)


def test_legacy_dict_and_wrapped_dict_yield_equivalent_configs() -> None:
    """Two paths into NicheConfig (flat shim vs wrapped direct) produce equal configs."""
    yaml_name = "cozy_dogs_v1.yaml"
    original = load_niche_config(_NICHES_DIR / yaml_name)
    wrapped = config_to_dict(original)
    flat = _legacy_flat_dict_from(yaml_name)
    via_flat = NicheConfig.model_validate(flat)
    via_wrapped = NicheConfig.model_validate(wrapped)
    assert via_flat == via_wrapped == original
