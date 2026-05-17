"""The QA run — evaluate a book's pages, regenerate rejects, finalise.

`run_qa` drives the loop the spec describes: QA every pending image, requeue
slots that were rejected (while retries remain), regenerate them through the
Fal provider, and repeat. When no slot can improve further the book is moved
to `qa_done` (and its passed pages copied to `images/filtered/`) or `failed`.
`build_review_html` and `apply_manual_verdict` support the manual review pass.
"""

from __future__ import annotations

import asyncio
import html
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from src.config.schema import NicheConfig, QASpec
from src.db.models import Book, BookStatus, Image, ImageQAStatus
from src.db.repos.books import fail_book, transition_status
from src.db.repos.images import list_images, update_qa
from src.generators.images import execute_plan, plan_generation
from src.providers.fal import FalProvider
from src.qa.image_qa import evaluate_image
from src.utils.logging import logger


@dataclass(frozen=True, slots=True)
class QAReport:
    """The outcome of a `run_qa` call."""

    rounds: int
    regenerated: int
    counts: dict[str, int]
    final_status: BookStatus


def _counts(images: list[Image]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for image in images:
        key = str(image.qa_status)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _qa_pending(book_id: UUID, qa: QASpec) -> int:
    """Evaluate every still-pending image of a book; return how many."""
    pending = [image for image in list_images(book_id) if image.qa_status == ImageQAStatus.PENDING]
    for image in pending:
        if image.local_path is None:
            update_qa(
                image.id,
                ImageQAStatus.REJECTED_RESOLUTION,
                {"reason": "image has no local file on disk"},
            )
            continue
        result = evaluate_image(Path(image.local_path), qa)
        update_qa(image.id, result.status, result.metrics())
        logger.debug(
            "QA slot {} attempt {}: {}",
            image.sequence_num,
            image.retry_attempt,
            result.status,
        )
    return len(pending)


def run_qa(
    book: Book,
    config: NicheConfig,
    *,
    provider_factory: Callable[[], FalProvider],
    output_dir: Path,
) -> QAReport:
    """Evaluate a book's pages, regenerating rejected slots until settled.

    `provider_factory` is called lazily — only when a round actually needs to
    regenerate — so QA on an all-passing book never touches the Fal provider.
    """
    if book.status == BookStatus.GENERATION_DONE:
        transition_status(book.id, BookStatus.QA_RUNNING)

    rounds = 0
    regenerated = 0
    for _ in range(config.qa.max_retries_per_slot + 2):
        rounds += 1
        _qa_pending(book.id, config.qa)
        plan = plan_generation(config, list_images(book.id))
        if not plan.to_generate:
            break
        logger.info(
            "QA round {}: regenerating {} rejected slot(s)",
            rounds,
            len(plan.to_generate),
        )
        asyncio.run(execute_plan(book, config, plan.to_generate, provider_factory(), output_dir))
        regenerated += len(plan.to_generate)

    final_images = list_images(book.id)
    final_plan = plan_generation(config, final_images)
    counts = _counts(final_images)

    if final_plan.exhausted:
        seqs = ", ".join(f"{slot.sequence_num:03d}" for slot in final_plan.exhausted)
        fail_book(book.id, phase="qa", reason=f"slots failed QA after retries: {seqs}")
        logger.warning("QA failed book {} — exhausted slots: {}", book.slug, seqs)
        return QAReport(rounds, regenerated, counts, BookStatus.FAILED)

    transition_status(book.id, BookStatus.QA_DONE)
    copied = copy_filtered_images(book, final_images, output_dir)
    logger.info("QA done for {} — {} pages copied to filtered/", book.slug, copied)
    return QAReport(rounds, regenerated, counts, BookStatus.QA_DONE)


def copy_filtered_images(book: Book, images: list[Image], output_dir: Path) -> int:
    """Copy each slot's passed image to ``images/filtered/<seq>.png``.

    When a slot passed on a retry, the latest passed attempt wins (the input
    is ordered by sequence then retry attempt). Returns the number copied.
    """
    filtered_dir = output_dir / book.slug / "images" / "filtered"
    filtered_dir.mkdir(parents=True, exist_ok=True)
    passed: dict[int, Image] = {}
    for image in images:
        if image.qa_status == ImageQAStatus.PASSED and image.local_path is not None:
            passed[image.sequence_num] = image
    for sequence_num, image in sorted(passed.items()):
        assert image.local_path is not None  # filtered above
        shutil.copyfile(image.local_path, filtered_dir / f"{sequence_num:03d}.png")
    return len(passed)


def apply_manual_verdict(image_id: UUID, *, approve: bool) -> None:
    """Override an image's QA verdict from a human review."""
    status = ImageQAStatus.PASSED if approve else ImageQAStatus.REJECTED_MANUAL
    update_qa(image_id, status, {"reason": "manual review", "approved": approve})


_REVIEW_CSS = """
body{font-family:system-ui,sans-serif;background:#1e1e1e;color:#eee;margin:1.5rem}
h1{font-size:1.1rem}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:1rem}
.cell{background:#2c2c2c;border-radius:6px;padding:.6rem}
.cell img{width:100%;height:auto;background:#fff;border-radius:4px}
.passed{border-left:5px solid #4caf50}
.rejected{border-left:5px solid #e53935}
.pending{border-left:5px solid #888}
.head{font-weight:bold;margin-top:.4rem}
.meta{font-size:.75rem;color:#aaa;margin-top:.3rem;white-space:pre-wrap}
"""


def _review_cell(image: Image, src: str) -> str:
    if image.qa_status == ImageQAStatus.PASSED:
        css = "passed"
    elif image.qa_status == ImageQAStatus.PENDING:
        css = "pending"
    else:
        css = "rejected"
    metrics = image.qa_metrics or {}
    meta = "\n".join(f"{key}: {value}" for key, value in metrics.items())
    return (
        f'<div class="cell {css}">'
        f'<img src="{html.escape(src)}" alt="page {image.sequence_num}">'
        f'<div class="head">#{image.sequence_num:03d} · attempt {image.retry_attempt}'
        f" · {html.escape(str(image.qa_status))}</div>"
        f'<div class="meta">id: {image.id}\n{html.escape(meta)}</div>'
        "</div>"
    )


def build_review_html(book: Book, images: list[Image], output_dir: Path) -> Path:
    """Write ``qa_review.html`` — a browsable grid of every image and verdict."""
    book_dir = output_dir / book.slug
    book_dir.mkdir(parents=True, exist_ok=True)
    cells = []
    for image in images:
        src = os.path.relpath(image.local_path, book_dir) if image.local_path is not None else ""
        cells.append(_review_cell(image, src))
    document = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>QA review — {html.escape(book.slug)}</title>"
        f"<style>{_REVIEW_CSS}</style></head><body>"
        f"<h1>QA review — {html.escape(book.slug)} · {len(images)} images</h1>"
        f'<div class="grid">{"".join(cells)}</div>'
        "</body></html>"
    )
    path = book_dir / "qa_review.html"
    path.write_text(document, encoding="utf-8")
    return path
