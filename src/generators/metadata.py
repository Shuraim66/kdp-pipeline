"""KDP listing-metadata generation — one Claude call, strictly validated.

`generate_metadata` asks Claude for a JSON listing (title, subtitle,
description, keywords, categories), validates it against KDP's hard limits,
and retries — feeding the validation errors back into the prompt — up to a
few times. `run_metadata` persists the result to the database and writes
``metadata.json``, ``description.txt`` and ``kdp_checklist.md``.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

from src.config.description_templates import (
    DescriptionTemplate,
    pick_template,
    skeleton_ngram_overlap,
)
from src.config.imprint import load_imprint
from src.config.schema import NicheConfig
from src.db.models import Book, BookStatus
from src.db.repos.books import transition_status, update_metadata
from src.providers.anthropic import (
    AnthropicProvider,
    AnthropicProviderError,
    AnthropicResult,
)
from src.utils.kdp_specs import compute_cover_dimensions, parse_trim_size
from src.utils.logging import logger

_MAX_ATTEMPTS = 3
_TITLE_SUBTITLE_MAX = 200
_DESCRIPTION_MIN = 1500
_DESCRIPTION_MAX = 2000
_KEYWORD_COUNT = 7
_KEYWORD_MAX_CHARS = 50
_REQUIRED_FIELDS = ("title", "subtitle", "description", "keywords", "categories")

# Overlap threshold above which Claude is suspected of over-copying the
# structural skeleton (e.g. leaving {placeholders} verbatim or echoing whole
# sentences). v1: warn for manual review; no auto-retry.
_SKELETON_OVERLAP_WARN = 0.60

_SYSTEM_PROMPT = (
    "You generate Amazon KDP book listing metadata. You respond with valid "
    "JSON only, no surrounding text, no markdown fences."
)

_USER_TEMPLATE = """Generate KDP listing metadata for this coloring book.

CONTEXT:
- Niche: {niche}
- Audience: {audience}
- Contents: {design_count} bold and easy line-art designs themed around {niche}
- Subject areas: {subjects}

CONSTRAINTS:
- title + subtitle combined: max 200 chars
- Title must contain "coloring book" and "{primary_keyword}"
- Description: 1500-2000 chars, 4 paragraphs (hook, what's inside, who it's \
for, call to action). Plain text, line breaks OK, no emoji, no markdown.
- Keywords: exactly 7, each <= 50 chars. Mix broad and long-tail. No brand \
names, no trademarks.
- Categories: use these exact paths: {categories}

SEEDS (improve them, don't copy verbatim):
- Title: {title_seed}
- Subtitle: {subtitle_seed}
- Keywords: {keywords_seed}

{template_block}
OUTPUT FORMAT (JSON only):
{{
  "title": "...",
  "subtitle": "...",
  "description": "...",
  "keywords": ["", "", "", "", "", "", ""],
  "categories": ["", ""]
}}"""

_CHECKLIST_TEMPLATE = """# KDP Upload Checklist — {title}

## Files
- Interior PDF: `output/{slug}/pdf/interior.pdf`
- Cover PDF: `output/{slug}/pdf/cover.pdf`

## Listing
- **Title**: {title}
- **Subtitle**: {subtitle}
- **Series**: (leave blank for v1)
- **Author**: {author}
- **Publisher**: {publisher}
- **Imprint**: {imprint_name}
- **Description**: (see description.txt)
- **Keywords** (7 total):
{keywords_block}
- **Categories** (2 total):
{categories_block}
- **Age range**: 16+
- **Language**: English
- **Low-content book**: {low_content_label}
- **Publishing rights**: I own the copyright

## Print settings
- Trim: {trim} in
- Paper: White
- Ink: Black & white
- Bleed: Yes
- Cover finish: Matte (or Glossy)
- Page count: {page_count}

## Pricing
- Amazon.com: ${price}
- Other marketplaces: auto-convert

## Pre-upload
- [ ] Interior PDF opens cleanly
- [ ] Cover matches calculated dims: {cover_w}x{cover_h} in
- [ ] All 7 keywords entered
- [ ] Categories selected from KDP browse paths
- [ ] Tax interview (W-8BEN) completed
- [ ] Payoneer USD bank linked

## Post-upload
- [ ] Record ASIN: ____________
- [ ] Run `uv run python -m src.main set-asin {slug} <ASIN>`
- [ ] Wait 72 hours for live status
- [ ] Order author proof copy for QC if first in series
"""


class MetadataValidationError(RuntimeError):
    """Claude's metadata failed validation on every attempt."""


@dataclass(frozen=True, slots=True)
class BookMetadata:
    """Validated KDP listing metadata for a book."""

    title: str
    subtitle: str
    description: str
    keywords: list[str]
    categories: list[str]
    # Set by `generate_metadata` from the rotated description template; empty
    # when the metadata was constructed directly (e.g. in tests).
    description_template_id: str = ""
    description_skeleton_overlap: float = 0.0


def _render_template_block(template: DescriptionTemplate) -> str:
    return (
        "TEMPLATE TO ADAPT (use as a structural skeleton, not verbatim — "
        "rewrite each section in your own voice, do NOT leave any "
        "{placeholders} literal in the output):\n\n"
        f"{template.skeleton}"
    )


def build_metadata_prompt(
    config: NicheConfig,
    *,
    book_id: UUID,
    prior_errors: list[str] | None = None,
) -> tuple[str, str]:
    """Return the ``(system, user)`` prompt pair for the metadata call.

    The rotated description-template skeleton (deterministic per ``book_id``)
    is injected as a structural hint; Claude is instructed to adapt, not copy.
    """
    template = pick_template(book_id)
    # The skeleton text contains literal `{placeholders}` that must survive
    # str.format(). Python's format() does not recurse into substituted
    # values, so passing `template_block=` with raw braces is safe.
    user = _USER_TEMPLATE.format(
        niche=config.niche,
        audience=config.book.target_audience,
        design_count=config.book.page_count,
        subjects=", ".join(subject.name for subject in config.subjects),
        primary_keyword=config.metadata.keywords_seed[0],
        categories=" | ".join(config.metadata.categories),
        title_seed=config.metadata.title_seed,
        subtitle_seed=config.metadata.subtitle_seed,
        keywords_seed=", ".join(config.metadata.keywords_seed),
        template_block=_render_template_block(template),
    )
    if prior_errors:
        problems = "\n".join(f"- {error}" for error in prior_errors)
        user += (
            "\n\nThe previous attempt failed validation. Correct these issues "
            f"and return JSON only:\n{problems}"
        )
    return _SYSTEM_PROMPT, user


def validate_metadata(data: dict[str, Any], config: NicheConfig) -> list[str]:
    """Check a metadata object against KDP limits; return the issues found."""
    issues: list[str] = []
    missing = [field for field in _REQUIRED_FIELDS if field not in data]
    if missing:
        return [f"missing field: {field}" for field in missing]

    title, subtitle, description = data["title"], data["subtitle"], data["description"]
    keywords, categories = data["keywords"], data["categories"]

    if not all(isinstance(v, str) for v in (title, subtitle, description)):
        issues.append("title, subtitle and description must be strings")
    if not isinstance(keywords, list) or not isinstance(categories, list):
        issues.append("keywords and categories must be arrays")
    if issues:
        return issues

    combined = len(title) + len(subtitle)
    if combined > _TITLE_SUBTITLE_MAX:
        issues.append(f"title + subtitle is {combined} chars, max {_TITLE_SUBTITLE_MAX}")
    if not _DESCRIPTION_MIN <= len(description) <= _DESCRIPTION_MAX:
        issues.append(
            f"description is {len(description)} chars, must be "
            f"{_DESCRIPTION_MIN}-{_DESCRIPTION_MAX}"
        )
    if len(keywords) != _KEYWORD_COUNT:
        issues.append(f"{len(keywords)} keywords, must be exactly {_KEYWORD_COUNT}")
    for keyword in keywords:
        if not isinstance(keyword, str) or len(keyword) > _KEYWORD_MAX_CHARS:
            issues.append(f"keyword too long or not a string: {keyword!r}")
    if len({k.lower() for k in keywords if isinstance(k, str)}) != len(keywords):
        issues.append("keywords contain duplicates")
    if categories != list(config.metadata.categories):
        issues.append(f"categories must be exactly {list(config.metadata.categories)}")
    return issues


async def generate_metadata(
    provider: AnthropicProvider,
    config: NicheConfig,
    *,
    book_id: UUID,
    max_attempts: int = _MAX_ATTEMPTS,
) -> tuple[BookMetadata, AnthropicResult]:
    """Generate listing metadata, retrying with the validation errors fed back.

    The book's description-template id (rotated by `book_id`) and the n-gram
    overlap of the generated description against that skeleton are captured on
    the returned BookMetadata; overlap above `_SKELETON_OVERLAP_WARN` emits a
    `logger.warning` for manual review (v1: no auto-retry on this signal).
    """
    template = pick_template(book_id)
    errors: list[str] = []
    for attempt in range(1, max_attempts + 1):
        system, user = build_metadata_prompt(config, book_id=book_id, prior_errors=errors or None)
        try:
            data, result = await provider.generate_json(
                prompt=user,
                system=system,
                max_tokens=2048,
                operation="generate_metadata",
                book_id=book_id,
            )
        except AnthropicProviderError as exc:
            errors = [f"response was not valid JSON: {exc}"]
            logger.warning("metadata attempt {} returned invalid JSON", attempt)
            continue
        errors = validate_metadata(data, config)
        if not errors:
            logger.info("metadata generated and validated on attempt {}", attempt)
            overlap = skeleton_ngram_overlap(data["description"], template.skeleton)
            if overlap > _SKELETON_OVERLAP_WARN:
                logger.warning(
                    "description n-gram overlap with skeleton {!r} is "
                    "{:.0%} (> {:.0%}) — review manually for over-templating",
                    template.id,
                    overlap,
                    _SKELETON_OVERLAP_WARN,
                )
            return (
                BookMetadata(
                    title=data["title"],
                    subtitle=data["subtitle"],
                    description=data["description"],
                    keywords=list(data["keywords"]),
                    categories=list(data["categories"]),
                    description_template_id=template.id,
                    description_skeleton_overlap=round(overlap, 4),
                ),
                result,
            )
        logger.warning("metadata attempt {} failed validation: {}", attempt, errors)
    raise MetadataValidationError(
        f"metadata failed validation after {max_attempts} attempts: {errors}"
    )


def build_checklist(
    book: Book, config: NicheConfig, metadata: BookMetadata, *, interior_page_count: int
) -> str:
    """Render the ``kdp_checklist.md`` content for a book."""
    trim_w, trim_h = parse_trim_size(config.book.trim_size)
    cover = compute_cover_dimensions(interior_page_count, trim_w, trim_h)
    imprint = load_imprint(config.imprint)
    return _CHECKLIST_TEMPLATE.format(
        title=metadata.title,
        slug=book.slug,
        subtitle=metadata.subtitle,
        author=config.metadata.author,
        publisher=imprint.publisher_field,
        imprint_name=imprint.name,
        keywords_block="\n".join(f"  - {kw}" for kw in metadata.keywords),
        categories_block="\n".join(f"  - {cat}" for cat in metadata.categories),
        low_content_label="Yes" if config.metadata.low_content else "No",
        trim=config.book.trim_size.replace("x", " x "),
        page_count=interior_page_count,
        price=f"{config.book.price_usd:.2f}",
        cover_w=f"{cover.total_width_in:.3f}",
        cover_h=f"{cover.total_height_in:.3f}",
    )


def persist_metadata(
    book: Book, config: NicheConfig, metadata: BookMetadata, *, output_dir: Path
) -> None:
    """Write metadata to the database and to the book's output directory."""
    interior_page_count = config.book.page_count + 2
    update_metadata(
        book.id,
        title=metadata.title,
        subtitle=metadata.subtitle,
        description=metadata.description,
        keywords=metadata.keywords,
        categories=metadata.categories,
        page_count=interior_page_count,
        trim_size=config.book.trim_size,
        price_usd=Decimal(str(config.book.price_usd)),
    )
    book_dir = output_dir / book.slug
    book_dir.mkdir(parents=True, exist_ok=True)
    imprint = load_imprint(config.imprint)
    payload = {
        "title": metadata.title,
        "subtitle": metadata.subtitle,
        "author": config.metadata.author,
        "publisher": imprint.publisher_field,
        "imprint": imprint.name,
        "description": metadata.description,
        "keywords": metadata.keywords,
        "categories": metadata.categories,
        "low_content": config.metadata.low_content,
        "description_template_id": metadata.description_template_id,
        "description_skeleton_overlap": metadata.description_skeleton_overlap,
    }
    (book_dir / "metadata.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (book_dir / "description.txt").write_text(metadata.description, encoding="utf-8")
    (book_dir / "kdp_checklist.md").write_text(
        build_checklist(book, config, metadata, interior_page_count=interior_page_count),
        encoding="utf-8",
    )
    logger.info("metadata, description and checklist written to {}", book_dir)


def run_metadata(
    book: Book,
    config: NicheConfig,
    *,
    provider: AnthropicProvider,
    output_dir: Path,
) -> BookMetadata:
    """Generate and persist a book's listing metadata; advance status to ready."""
    if book.status == BookStatus.ASSEMBLING:
        transition_status(book.id, BookStatus.METADATA_PENDING)
    metadata, _result = asyncio.run(generate_metadata(provider, config, book_id=book.id))
    # generate_metadata already needs book_id for template selection.
    persist_metadata(book, config, metadata, output_dir=output_dir)
    transition_status(book.id, BookStatus.READY)
    return metadata
