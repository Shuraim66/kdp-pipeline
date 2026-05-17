"""Image generation flow — plan slots, render concurrently, persist results.

A book's pages are a fixed grid of *slots*: every subject crossed with every
variation, numbered ``0 .. page_count-1``. `plan_generation` inspects the
images already in the database and decides which slots still need rendering;
`execute_plan` renders them through the Fal provider, writes the PNGs under
``output/<slug>/images/raw/``, and inserts an `images` row per result.
`run_generation` ties the two together and advances the book's status.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from uuid import UUID

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from src.config.loader import compute_config_hash, config_to_dict, load_niche_config
from src.config.schema import NicheConfig
from src.db.models import Book, BookNotFoundError, BookStatus, Image, ImageQAStatus
from src.db.repos.books import create_book, get_book_by_slug, transition_status
from src.db.repos.images import create_image
from src.providers.fal import FalProvider
from src.utils.hashing import sha256_bytes
from src.utils.line_art import normalize_line_art
from src.utils.logging import logger
from src.utils.prompts import build_image_prompt

_YAML_SUFFIXES = (".yaml", ".yml")


@dataclass(frozen=True, slots=True)
class Slot:
    """One page slot — a subject paired with one of its variations."""

    sequence_num: int
    subject: str
    variation_idx: int


@dataclass(frozen=True, slots=True)
class PlannedImage:
    """A slot queued for rendering, with its prompt and retry lineage."""

    slot: Slot
    retry_attempt: int
    retry_of_image_id: UUID | None
    prompt: str
    negative_prompt: str


@dataclass(frozen=True, slots=True)
class GenerationPlan:
    """The verdict of `plan_generation`: what to render, skip, or give up on."""

    total_slots: int
    to_generate: list[PlannedImage]
    skipped: int
    exhausted: list[Slot]


@dataclass(frozen=True, slots=True)
class SlotFailure:
    """A slot whose rendering raised after exhausting retries."""

    planned: PlannedImage
    error: str


@dataclass(frozen=True, slots=True)
class GenerationReport:
    """The outcome of an `execute_plan` run."""

    succeeded: int
    failures: list[SlotFailure]
    total_cost: Decimal


def expand_slots(config: NicheConfig) -> list[Slot]:
    """Expand a niche config into its ordered list of page slots."""
    slots: list[Slot] = []
    sequence_num = 0
    for subject in config.subjects:
        for variation_idx in range(config.variations_per_subject):
            slots.append(Slot(sequence_num, subject, variation_idx))
            sequence_num += 1
    return slots


def _planned(
    config: NicheConfig, slot: Slot, *, retry_attempt: int, retry_of: UUID | None
) -> PlannedImage:
    prompt, negative_prompt = build_image_prompt(config, slot.subject, slot.variation_idx)
    return PlannedImage(
        slot=slot,
        retry_attempt=retry_attempt,
        retry_of_image_id=retry_of,
        prompt=prompt,
        negative_prompt=negative_prompt,
    )


def plan_generation(
    config: NicheConfig,
    existing: Iterable[Image],
    *,
    limit: int | None = None,
) -> GenerationPlan:
    """Decide which slots still need an image.

    A slot is skipped if it already has a passed image, or its latest attempt
    is still awaiting QA. A slot whose latest attempt was rejected is requeued
    (as the next retry) while retries remain, and otherwise reported as
    `exhausted`. `limit`, when given, caps `to_generate` — used by test mode.
    """
    by_seq: defaultdict[int, list[Image]] = defaultdict(list)
    for image in existing:
        by_seq[image.sequence_num].append(image)

    max_retries = config.qa.max_retries_per_slot
    to_generate: list[PlannedImage] = []
    skipped = 0
    exhausted: list[Slot] = []

    for slot in expand_slots(config):
        images = by_seq.get(slot.sequence_num, [])
        if any(image.qa_status == ImageQAStatus.PASSED for image in images):
            skipped += 1
            continue
        if not images:
            to_generate.append(_planned(config, slot, retry_attempt=0, retry_of=None))
            continue
        latest = max(images, key=lambda image: image.retry_attempt)
        if latest.qa_status == ImageQAStatus.PENDING:
            # Rendered but not yet QA'd — not this command's job to redo.
            skipped += 1
        elif latest.retry_attempt < max_retries:
            to_generate.append(
                _planned(
                    config,
                    slot,
                    retry_attempt=latest.retry_attempt + 1,
                    retry_of=latest.id,
                )
            )
        else:
            exhausted.append(slot)

    total_slots = len(config.subjects) * config.variations_per_subject
    if limit is not None:
        to_generate = to_generate[:limit]
    return GenerationPlan(
        total_slots=total_slots,
        to_generate=to_generate,
        skipped=skipped,
        exhausted=exhausted,
    )


async def _generate_one(
    book: Book,
    config: NicheConfig,
    planned: PlannedImage,
    provider: FalProvider,
    raw_dir: Path,
) -> Decimal:
    """Render one planned slot, save the PNG, and insert its `images` row."""
    generation = config.generation
    width, height = generation.image_dimensions
    # A fixed seed gives the first attempt reproducibility; retries always get
    # a fresh (random) seed so a rejected image is not reproduced verbatim.
    seed = (
        generation.fixed_seed
        if generation.fixed_seed is not None and planned.retry_attempt == 0
        else None
    )
    # FLUX schnell takes no guidance; a niche signals that with guidance 0.
    guidance = generation.guidance_scale if generation.guidance_scale > 0 else None
    # `loras` and `negative_prompt` are only valid on the fal-ai/flux-lora
    # endpoint — send them only when the niche actually configures a LoRA.
    loras = [{"path": lora.path, "scale": lora.scale} for lora in generation.loras]

    result = await provider.generate_image(
        prompt=planned.prompt,
        model=generation.model,
        width=width,
        height=height,
        num_inference_steps=generation.num_inference_steps,
        seed=seed,
        guidance_scale=guidance,
        negative_prompt=planned.negative_prompt if loras else None,
        loras=loras or None,
        book_id=book.id,
    )

    # The model emits thin antialiased strokes at its own (capped) resolution;
    # normalise to crisp bold line art at the QA-required print resolution.
    processed = await asyncio.to_thread(
        normalize_line_art,
        result.image_bytes,
        target_size=config.qa.required_dimensions,
    )

    filename = f"{planned.slot.sequence_num:03d}_attempt{planned.retry_attempt}.png"
    path = raw_dir / filename
    await asyncio.to_thread(path.write_bytes, processed)

    final_width, final_height = config.qa.required_dimensions
    generation_params = {
        "model": generation.model,
        "width": final_width,
        "height": final_height,
        "model_output_width": result.width,
        "model_output_height": result.height,
        "num_inference_steps": generation.num_inference_steps,
        "guidance_scale": generation.guidance_scale,
        "subject": planned.slot.subject,
        "variation_idx": planned.slot.variation_idx,
        "loras": [lora.name or lora.path for lora in generation.loras],
    }
    await asyncio.to_thread(
        create_image,
        book.id,
        planned.slot.sequence_num,
        planned.prompt,
        result.seed,
        generation.model,
        generation_params,
        negative_prompt=planned.negative_prompt,
        local_path=str(path),
        file_sha256=sha256_bytes(processed),
        cost_usd=result.cost_usd,
        retry_of_image_id=planned.retry_of_image_id,
        retry_attempt=planned.retry_attempt,
    )
    logger.debug("slot {} saved -> {}", planned.slot.sequence_num, path)
    return result.cost_usd


async def execute_plan(
    book: Book,
    config: NicheConfig,
    planned: list[PlannedImage],
    provider: FalProvider,
    output_dir: Path,
    *,
    show_progress: bool = True,
) -> GenerationReport:
    """Render every planned slot concurrently; a per-slot failure is isolated."""
    raw_dir = output_dir / book.slug / "images" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        disable=not show_progress,
    ) as progress:
        task = progress.add_task(f"Generating {book.slug}", total=len(planned))

        async def _one(
            item: PlannedImage,
        ) -> tuple[PlannedImage, Decimal | None, str | None]:
            try:
                cost = await _generate_one(book, config, item, provider, raw_dir)
            except Exception as exc:
                logger.warning(
                    "slot {} failed: {}: {}",
                    item.slot.sequence_num,
                    type(exc).__name__,
                    exc,
                )
                return item, None, f"{type(exc).__name__}: {exc}"
            else:
                return item, cost, None
            finally:
                progress.advance(task)

        results: list[tuple[PlannedImage, Decimal | None, str | None]] = await asyncio.gather(
            *(_one(item) for item in planned)
        )

    succeeded = sum(1 for _, cost, _ in results if cost is not None)
    failures = [
        SlotFailure(planned=item, error=error) for item, _, error in results if error is not None
    ]
    total_cost = sum((cost for _, cost, _ in results if cost is not None), Decimal(0))
    return GenerationReport(succeeded=succeeded, failures=failures, total_cost=total_cost)


def _dump_failed_prompts(book: Book, report: GenerationReport, output_dir: Path) -> None:
    """Write the prompts of failed slots to ``failed_prompts.txt`` for review."""
    path = output_dir / book.slug / "failed_prompts.txt"
    blocks = [
        f"# seq {f.planned.slot.sequence_num:03d} | {f.planned.slot.subject} "
        f"| variation {f.planned.slot.variation_idx} | attempt {f.planned.retry_attempt}\n"
        f"# error: {f.error}\n"
        f"{f.planned.prompt}\n"
        for f in report.failures
    ]
    path.write_text("\n".join(blocks), encoding="utf-8")
    logger.info("dumped {} failed prompt(s) to {}", len(report.failures), path)


def run_generation(
    book: Book,
    config: NicheConfig,
    plan: GenerationPlan,
    *,
    test_mode: bool,
    provider: FalProvider,
    output_dir: Path,
) -> GenerationReport:
    """Run a planned generation: advance status, render, dump failed prompts.

    Test mode renders the planned slots without touching the book's status —
    a quality preview before committing to the full batch.
    """
    if not test_mode and book.status in (BookStatus.CREATED, BookStatus.QA_DONE):
        transition_status(book.id, BookStatus.GENERATING)

    report = asyncio.run(execute_plan(book, config, plan.to_generate, provider, output_dir))

    if report.failures:
        _dump_failed_prompts(book, report, output_dir)

    if not test_mode and not report.failures:
        # Every queued slot rendered — the book's images are complete.
        transition_status(book.id, BookStatus.GENERATION_DONE)
    return report


def resolve_book(target: str) -> tuple[Book, NicheConfig]:
    """Resolve a CLI target — a niche YAML path or a slug — to (book, config).

    A YAML path creates the book if it does not exist yet. A slug must already
    exist. For an existing book the stored config is authoritative; a YAML
    whose hash differs only triggers a warning.
    """
    path = Path(target)
    if path.suffix.lower() in _YAML_SUFFIXES:
        if not path.is_file():
            raise FileNotFoundError(f"niche config not found: {target}")
        config = load_niche_config(path)
        book = get_book_by_slug(config.slug)
        if book is None:
            book = create_book(config.slug, config.niche, config_to_dict(config))
            logger.info("created book {} from {}", config.slug, target)
            return book, config
        if book.config_hash != compute_config_hash(config):
            logger.warning(
                "book {} already exists with a different config; using the stored config, not {}",
                config.slug,
                target,
            )
        return book, NicheConfig.model_validate(book.config)

    book = get_book_by_slug(target)
    if book is None:
        raise BookNotFoundError(
            f"no book with slug {target!r} — pass the niche YAML path to create it"
        )
    return book, NicheConfig.model_validate(book.config)
