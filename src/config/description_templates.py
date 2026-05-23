"""Description-template rotation + n-gram overlap check.

Anti-duplication safeguard for a growing catalogue: every book is assigned a
deterministic structural template (rotated by ``book.id`` hash) which Claude
adapts in its own voice. After generation, we measure the n-gram overlap
between the produced description and the skeleton; high overlap suggests
Claude over-copied the template (e.g. left placeholders verbatim) and is
flagged for manual review.
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path
from uuid import UUID

import yaml

from src.config.schema import Slug, _Strict

_TEMPLATES_PATH = (
    Path(__file__).resolve().parents[2] / "niches" / "_shared" / "description_templates.yaml"
)


class DescriptionTemplate(_Strict):
    """One structural skeleton for a KDP description."""

    id: Slug
    skeleton: str


@cache
def load_description_templates() -> list[DescriptionTemplate]:
    """Parse the shared template pool. Cached; raises on malformed input."""
    text = _TEMPLATES_PATH.read_text(encoding="utf-8")
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"malformed description-templates YAML at {_TEMPLATES_PATH}: {exc}"
        ) from exc
    if not isinstance(raw, dict) or "templates" not in raw:
        raise ValueError(
            f"{_TEMPLATES_PATH}: expected a YAML mapping with a top-level `templates:` key"
        )
    templates = [DescriptionTemplate.model_validate(entry) for entry in raw["templates"]]
    if len(templates) < 2:
        raise ValueError(f"{_TEMPLATES_PATH}: need at least 2 templates, got {len(templates)}")
    ids = [t.id for t in templates]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{_TEMPLATES_PATH}: duplicate template ids: {ids}")
    return templates


def pick_template(book_id: UUID) -> DescriptionTemplate:
    """Deterministically pick one template for a book — same id ⇒ same template."""
    templates = load_description_templates()
    idx = int(book_id.hex[:8], 16) % len(templates)
    return templates[idx]


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    tokens = [token.lower() for token in re.findall(r"\b\w+\b", text)]
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def skeleton_ngram_overlap(generated_description: str, skeleton: str, n: int = 4) -> float:
    """Fraction of n-grams in `generated_description` that also appear in `skeleton`.

    Returns 0.0 when the description has too few words to form any n-gram —
    that's a degenerate case worth flagging upstream by other means (length
    check), not via this function.
    """
    desc_ngrams = _ngrams(generated_description, n)
    if not desc_ngrams:
        return 0.0
    skel_ngrams = _ngrams(skeleton, n)
    return len(desc_ngrams & skel_ngrams) / len(desc_ngrams)
