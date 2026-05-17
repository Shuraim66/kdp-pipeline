"""Pydantic models for niche configuration — strict validation."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Reusable constrained string types.
HexColor = Annotated[str, Field(pattern=r"^#[0-9a-fA-F]{6}$")]
Slug = Annotated[str, Field(pattern=r"^[a-z0-9_]+$")]


class _Strict(BaseModel):
    """Base model: reject unknown keys so config typos fail loudly."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class BookSpec(_Strict):
    """The `book:` section — trim, length, price, audience."""

    trim_size: Literal["8.5x8.5", "8.5x11"]
    page_count: int = Field(ge=24, le=100)
    price_usd: float = Field(gt=0)
    target_audience: str


class StyleSpec(_Strict):
    """The `style:` section — art direction and negative prompts."""

    art_style: str
    line_weight: str
    negative_prompts: str


class MetadataSpec(_Strict):
    """The `metadata:` section — KDP listing seeds."""

    title_seed: str
    subtitle_seed: str
    keywords_seed: list[str] = Field(min_length=1, max_length=7)
    categories: list[str] = Field(min_length=2, max_length=2)


class CoverSpec(_Strict):
    """The `cover:` section — colors, hero art, and font."""

    background_color: HexColor
    accent_color: HexColor
    text_color: HexColor
    hero_subject: str
    hero_style: str
    font_family: str


class GenerationSpec(_Strict):
    """The `generation:` section — Fal.ai model and parameters."""

    model: str
    image_dimensions: tuple[int, int]
    num_inference_steps: int = Field(ge=1)
    guidance_scale: float = Field(ge=0)
    fixed_seed: int | None = None


class QASpec(_Strict):
    """The `qa:` section — image QA thresholds."""

    min_white_pct: float = Field(ge=0, le=100)
    max_gray_pct: float = Field(ge=0, le=100)
    required_white_margin_px: int = Field(ge=0)
    required_dimensions: tuple[int, int]
    max_retries_per_slot: int = Field(ge=0)


class NicheConfig(_Strict):
    """A fully validated niche configuration."""

    slug: Slug
    niche: str
    book: BookSpec
    style: StyleSpec
    subjects: list[str] = Field(min_length=1)
    variations_per_subject: int = Field(ge=1)
    composition_modifiers: list[str] = Field(min_length=1)
    metadata: MetadataSpec
    cover: CoverSpec
    generation: GenerationSpec
    qa: QASpec

    @model_validator(mode="after")
    def _check_subject_count(self) -> NicheConfig:
        """subjects x variations_per_subject must equal book.page_count."""
        produced = len(self.subjects) * self.variations_per_subject
        if produced != self.book.page_count:
            raise ValueError(
                f"{len(self.subjects)} subjects x {self.variations_per_subject} "
                f"variations = {produced} pages, but book.page_count is "
                f"{self.book.page_count}"
            )
        return self
