"""Pydantic models for niche configuration — strict validation.

Niche configs are polymorphic on `body.kind`: a coloring book carries the
image-generation, style, subject, post-process, and QA blocks; a puzzle-maze
book carries a puzzle spec and a front-matter spec. The discriminator lives
on the nested `body` field, but YAML / dict input is accepted in either the
WRAPPED form (`body: {kind: ..., ...}`) or the LEGACY flat form (top-level
coloring keys or top-level `book_type: puzzle_maze` + `puzzle:` + `front_matter:`).
A ``mode="before"`` model validator rewrites legacy input into the wrapped form,
so existing YAML files and stored DB configs continue to validate unchanged.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

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
    # KDP's "Low-Content Book" listing toggle. Our coloring books are NOT
    # low-content (they have unique, designed interior pages); this defaults
    # to False. Set true for journals / planners / blank-pages products.
    low_content: bool = False


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
    back_headline: str = ""
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
    # Composition QA threshold: the subject's bounding box must cover at least
    # this fraction of the inner canvas (inset by `required_white_margin_px`).
    # 0.40 was calibrated against book 1's 50 filtered images — flags the
    # three named small-subject pages without punishing kawaii pages with
    # legitimate whitespace. The gate is a retry trigger, not a quality bar.
    # Per-niche tunable; dog-breed niches may want 0.50. See CALIBRATION_NOTES.
    min_subject_area_ratio: float = 0.40
    # Optional suffix appended to the regenerated image prompt when the
    # previous attempt was REJECTED_COMPOSITION. Niche-tunable hint, e.g.
    # "fill at least 70% of the canvas, centered, with bold thick outlines".
    composition_retry_prompt_suffix: str = ""

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

    @model_validator(mode="after")
    def _check_min_subject_area_ratio(self) -> QASpec:
        """The ratio must be in (0, 1] — a fraction of inner canvas area."""
        if not 0.0 < self.min_subject_area_ratio <= 1.0:
            raise ValueError(
                f"qa.min_subject_area_ratio must be in (0, 1], got {self.min_subject_area_ratio}"
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


class ColoringBody(_Strict):
    """The coloring-book-specific portion of a niche config."""

    kind: Literal["coloring"] = "coloring"
    style: StyleSpec
    subjects: list[Subject] = Field(min_length=1)
    variations_per_subject: int = Field(ge=1)
    composition_modifiers: list[str] = Field(min_length=1)
    generation: GenerationSpec
    post_process: PostProcessSpec = Field(default_factory=PostProcessSpec)
    qa: QASpec


class PuzzleRenderSpec(_Strict):
    """Per-niche rendering controls for puzzle pages.

    ``path_wall_ratios`` sets the path:wall pixel-width ratio per difficulty
    — higher values mean wider passages a young child can trace with a
    crayon, lower values pack tighter adult-style mazes. ``wall_thickness_px``
    is a FLOOR on the rendered wall pixel size; the renderer fills the page
    as much as it can while keeping walls ``>= wall_thickness_px``. For a
    standard 8.5x11 puzzle book at 300 DPI, the page-filling computation
    keeps walls comfortably above the floor — the floor only kicks in when
    the grid is too dense for the canvas (and prevents printing hair-thin
    walls that don't reproduce well on Amazon's offset press).
    """

    path_wall_ratios: dict[Literal["easy", "medium", "hard"], float] = Field(
        default_factory=lambda: dict[Literal["easy", "medium", "hard"], float](
            {"easy": 4.0, "medium": 2.5, "hard": 1.8}
        )
    )
    wall_thickness_px: int = Field(default=12, ge=4, le=40)

    @model_validator(mode="after")
    def _ratios_cover_all_difficulties(self) -> PuzzleRenderSpec:
        """All three difficulty keys must be set — partial overrides would silently default."""
        required = {"easy", "medium", "hard"}
        missing = required - set(self.path_wall_ratios)
        if missing:
            raise ValueError(
                f"puzzle.render.path_wall_ratios must set every difficulty; "
                f"missing: {sorted(missing)}"
            )
        for difficulty, ratio in self.path_wall_ratios.items():
            if ratio < 1.0:
                raise ValueError(
                    f"puzzle.render.path_wall_ratios[{difficulty!r}] must be >= 1.0 "
                    f"(paths cannot be narrower than walls); got {ratio}"
                )
        return self


class PuzzleSpec(_Strict):
    """The `puzzle:` section — maze generation parameters."""

    type: Literal["maze"] = "maze"
    algorithm: Literal["prim", "kruskal", "backtracker", "hunt_and_kill"] = "backtracker"
    count: int = Field(ge=1, le=200)
    # The curve CYCLES across mazes: difficulty for maze i is
    # `difficulty_curve[i % len(difficulty_curve)]`. No length-equals-count
    # constraint — a 5-element curve over 40 mazes is fine.
    difficulty_curve: list[Literal["easy", "medium", "hard"]] = Field(min_length=1)
    grid_sizes: dict[Literal["easy", "medium", "hard"], tuple[int, int]]
    include_solutions: bool = True
    solutions_section: Literal["end", "interleaved"] = "end"
    themed_borders: bool = False
    fixed_seed: int | None = None
    # Per-niche path/wall sizing — defaults suit a 6-10 audience; override
    # in the YAML to bias toddler-friendlier or adult-tighter.
    render: PuzzleRenderSpec = Field(default_factory=PuzzleRenderSpec)

    @model_validator(mode="after")
    def _grid_sizes_cover_curve(self) -> PuzzleSpec:
        """Every difficulty used by the curve must have a grid size entry."""
        for difficulty in set(self.difficulty_curve):
            if difficulty not in self.grid_sizes:
                raise ValueError(
                    f"puzzle.grid_sizes is missing an entry for difficulty {difficulty!r} "
                    f"used by puzzle.difficulty_curve"
                )
        return self


class PuzzleFrontMatterSpec(_Strict):
    """The puzzle-book `front_matter:` section — page toggles + custom text."""

    title_page: bool = True
    intro_page: bool = True
    intro_text: str = ""
    # Optional custom tips block for the copyright page (the maze-book default
    # is supplied by the puzzle book builder when this is empty).
    copyright_tips: tuple[str, ...] = ()


class PuzzleBody(_Strict):
    """The puzzle-book-specific portion of a niche config."""

    kind: Literal["puzzle_maze"]
    puzzle: PuzzleSpec
    front_matter: PuzzleFrontMatterSpec = Field(default_factory=PuzzleFrontMatterSpec)


# Keys that live under `body` in the wrapped form but appear at the YAML top
# level in the legacy/flat form. Used by the back-compat shim below.
_COLORING_BODY_KEYS: tuple[str, ...] = (
    "style",
    "subjects",
    "variations_per_subject",
    "composition_modifiers",
    "generation",
    "post_process",
    "qa",
)
_PUZZLE_BODY_KEYS: tuple[str, ...] = ("puzzle", "front_matter")


def _wrap_body_input(data: dict[str, Any]) -> dict[str, Any]:
    """Rewrite flat input into the wrapped `body: {kind, ...}` form.

    Existing coloring YAMLs have no `book_type` and no `body:` key — the
    coloring keys sit at the top level. New puzzle YAMLs carry top-level
    `book_type: puzzle_maze` plus `puzzle:` and `front_matter:` keys. Both
    shapes are normalised here so the discriminated union always sees the
    same wrapped form. Already-wrapped input passes through unchanged.
    """
    if "body" in data:
        return data
    data = dict(data)
    kind = data.pop("book_type", "coloring")
    if kind == "coloring":
        body_keys: tuple[str, ...] = _COLORING_BODY_KEYS
    elif kind == "puzzle_maze":
        body_keys = _PUZZLE_BODY_KEYS
    else:
        # Unknown kind — leave it for the discriminator to reject.
        data["body"] = {"kind": kind}
        return data
    body: dict[str, Any] = {"kind": kind}
    for key in body_keys:
        if key in data:
            body[key] = data.pop(key)
    data["body"] = body
    return data


class NicheConfig(_Strict):
    """A fully validated niche configuration — discriminated on `body.kind`."""

    slug: Slug
    niche: str
    # Imprint slug — references imprints/<imprint>.yaml. Default keeps existing
    # niches valid; the imprint resolves the publisher line and the cover's
    # visual identity (palette + font families) at use time.
    imprint: Slug = "quiet_hours_press"
    book: BookSpec
    metadata: MetadataSpec
    cover: CoverSpec
    body: Annotated[ColoringBody | PuzzleBody, Field(discriminator="kind")]

    @model_validator(mode="before")
    @classmethod
    def _wrap_legacy_input(cls, data: Any) -> Any:
        """Accept legacy flat input by wrapping it into the discriminated form."""
        if isinstance(data, dict):
            return _wrap_body_input(data)
        return data

    @property
    def coloring(self) -> ColoringBody | None:
        """The coloring body, or None if this is a different book type."""
        return self.body if isinstance(self.body, ColoringBody) else None

    @property
    def puzzle(self) -> PuzzleBody | None:
        """The puzzle body, or None if this is a different book type."""
        return self.body if isinstance(self.body, PuzzleBody) else None

    def require_coloring(self) -> ColoringBody:
        """Narrow `body` to ColoringBody; raise ValueError if not.

        Never use ``assert`` for this — ``assert`` is stripped under
        ``python -O`` and a mismatched config would silently access None.
        """
        if not isinstance(self.body, ColoringBody):
            raise ValueError(
                f"This code path requires book_type='coloring'; "
                f"got {self.body.kind!r} for slug {self.slug!r}."
            )
        return self.body

    def require_puzzle(self) -> PuzzleBody:
        """Narrow `body` to PuzzleBody; raise ValueError if not."""
        if not isinstance(self.body, PuzzleBody):
            raise ValueError(
                f"This code path requires book_type='puzzle_maze'; "
                f"got {self.body.kind!r} for slug {self.slug!r}."
            )
        return self.body

    @model_validator(mode="after")
    def _check_page_count(self) -> NicheConfig:
        """The interior content-page count must match `book.page_count`.

        Front-matter pages (title, copyright, intro, solutions divider) are
        added separately by `total_interior_pages()` in `src/utils/kdp_specs.py`
        — same convention as the existing coloring +2 (title + copyright).
        """
        if isinstance(self.body, ColoringBody):
            produced = len(self.body.subjects) * self.body.variations_per_subject
            if produced != self.book.page_count:
                raise ValueError(
                    f"{len(self.body.subjects)} subjects x {self.body.variations_per_subject} "
                    f"variations = {produced} pages, but book.page_count is "
                    f"{self.book.page_count}"
                )
        elif isinstance(self.body, PuzzleBody):
            spec = self.body.puzzle
            solutions_pages = (
                spec.count if spec.include_solutions and spec.solutions_section == "end" else 0
            )
            expected = spec.count + solutions_pages
            if expected != self.book.page_count:
                raise ValueError(
                    f"puzzle.count={spec.count} + {solutions_pages} solution pages = "
                    f"{expected}, but book.page_count is {self.book.page_count}. "
                    "book.page_count must equal puzzle.count + the solutions section "
                    "(when include_solutions=true and solutions_section='end')."
                )
        return self

    @model_validator(mode="after")
    def _puzzle_requires_cover_bullets(self) -> NicheConfig:
        """Puzzle books must set cover.bullets — the coloring fallback would lie."""
        if isinstance(self.body, PuzzleBody) and len(self.cover.bullets) < 3:
            raise ValueError(
                "puzzle books must specify cover.bullets explicitly (>=3 items). "
                "The coloring-specific fallback in src/generators/cover.py:"
                "_feature_bullets() does not apply to puzzle books — it references "
                "'coloring outlines' and would print nonsense on a maze book's "
                "back cover."
            )
        return self
