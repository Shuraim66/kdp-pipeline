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
    author: str
    keywords_seed: list[str] = Field(min_length=1, max_length=7)
    categories: list[str] = Field(min_length=2, max_length=2)


class CoverSpec(_Strict):
    """The `cover:` section — colors, hero art, font, and front/back text."""

    background_color: HexColor
    accent_color: HexColor
    text_color: HexColor
    hero_subject: str
    hero_style: str
    font_family: str
    # Front-cover text. Empty falls back to the metadata title/subtitle.
    headline: str = ""
    subtitle: str = ""
    badge_text: str = ""
    # Back-cover copy plus the interior pages (sequence numbers) to preview.
    tagline: str = ""
    bullets: list[str] = []
    thumbnails: list[int] = []


class LoraConfig(_Strict):
    """One LoRA adapter, injected at inference via the fal-ai/flux-lora endpoint."""

    path: str  # URL to the .safetensors weights (HuggingFace or Civitai)
    scale: float = Field(default=1.0, ge=0.0, le=2.0)
    name: str | None = None  # human label, for logs only
    trigger: str | None = None  # phrase prepended to the prompt to activate the style


class GenerationSpec(_Strict):
    """The `generation:` section — Fal.ai model and parameters."""

    model: str = "fal-ai/flux/dev"
    image_dimensions: tuple[int, int]
    num_inference_steps: int = Field(ge=1)
    guidance_scale: float = Field(ge=0)
    fixed_seed: int | None = None
    loras: list[LoraConfig] = Field(default_factory=list)


class QASpec(_Strict):
    """The `qa:` section — image QA thresholds."""

    min_white_pct: float = Field(ge=0, le=100)
    max_gray_pct: float = Field(ge=0, le=100)
    required_white_margin_px: int = Field(ge=0)
    required_dimensions: tuple[int, int]
    max_retries_per_slot: int = Field(ge=0)
    # Advisory ink-density band [low, high] in percent — the healthy ink-
    # coverage range for a well-tuned subject. Surfaced for human review by
    # the Q6 feedback tooling (`ink-density`, `validate-niche --ink-preview`);
    # it is never tied to `qa_status` and never gates QA. The default suits
    # Bold & Easy niches; an intricate niche can widen the upper bound.
    ink_density_band: tuple[float, float] = (3.0, 8.0)

    @model_validator(mode="after")
    def _check_ink_density_band(self) -> QASpec:
        """The band must be an ascending [low, high] pair within 0-100."""
        low, high = self.ink_density_band
        if not 0.0 <= low < high <= 100.0:
            raise ValueError(
                "qa.ink_density_band must be [low, high] with 0 <= low < high <= 100, "
                f"got {list(self.ink_density_band)}"
            )
        return self


class PostProcessSpec(_Strict):
    """The `post_process:` section — how a generated page is normalised.

    `minimal` upscales and whitens the background only, keeping the model's own
    line art; `dilate` / `auto` binarise and bold the strokes (for LoRAs whose
    output is pale and thin). See `src/utils/line_art.py`.
    """

    mode: Literal["minimal", "dilate", "auto"] = "minimal"


class Subject(_Strict):
    """One coloring-book subject: the phrase to draw and what kind of thing it is.

    `kind` steers the image prompt. An ``object`` is drawn isolated, with no
    invented characters or environment around it; a ``character`` as a single
    centred figure; a ``scene`` as a coherent multi-element composition.
    """

    name: str
    kind: Literal["object", "character", "scene"]


class NicheConfig(_Strict):
    """A fully validated niche configuration."""

    slug: Slug
    niche: str
    book: BookSpec
    style: StyleSpec
    subjects: list[Subject] = Field(min_length=1)
    variations_per_subject: int = Field(ge=1)
    composition_modifiers: list[str] = Field(min_length=1)
    metadata: MetadataSpec
    cover: CoverSpec
    generation: GenerationSpec
    post_process: PostProcessSpec = Field(default_factory=PostProcessSpec)
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
