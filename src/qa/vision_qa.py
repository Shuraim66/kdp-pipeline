"""Vision-based QA for generated coloring-book pages.

Pixel QA (`src/qa/image_qa.py`) catches technical failures — resolution,
grey shading, clipped margins. It cannot see *semantic* failures: a wrong or
unrecognisable subject, an incoherent composition, broken anatomy. Those are
what earns a 1-star review.

`evaluate_image` grades one page with Claude Sonnet, scoring it on five
criteria against a strict rubric and returning a structured verdict. Low
scores map to a `rejected_vision_*` status; the QA runner's existing retry
loop regenerates them. This stage runs *after* pixel QA, only on pages that
already passed it — there is no point grading a broken pixel grid.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from uuid import UUID

from PIL import Image as PILImage

from src.config.schema import NicheConfig
from src.db.models import Image, ImageQAStatus
from src.providers.anthropic import AnthropicProvider
from src.utils.logging import logger

# The spec names `claude-sonnet-4-7`, which is not a real model id. Sonnet 4.6
# is the current generation and the cost-appropriate tier for rubric grading.
VISION_QA_MODEL = "claude-sonnet-4-6"
# Pass mark out of 100. Calibrated to 80: on the Q2 test set the scores split
# into a clean pass cluster (83-87) and reject cluster (65-74), and 80 sits in
# the middle of that 9-point gap — see CALIBRATION_NOTES.md. Re-tune per niche
# with `tune-vision-threshold`.
DEFAULT_PASS_THRESHOLD = 80
# Pages are downscaled to this square before upload — smaller is cheaper, and
# line-art structure survives the downscale fine.
EVAL_IMAGE_SIZE = 1024


@dataclass(frozen=True, slots=True)
class VisionQAResult:
    """The Vision QA verdict for one page."""

    score: int  # 0-100, the sum of the five subscores
    subscores: dict[str, int]
    passed: bool
    rejection_reason: ImageQAStatus | None  # a rejected_vision_* status, or None
    issues: list[str]
    prompt_hint: str | None
    cost_usd: Decimal
    raw_response: str  # the model's reply, kept for audit


SYSTEM_PROMPT = """You are a strict quality reviewer for a print-on-demand adult \
coloring book that will be sold on Amazon. Your job is to catch pages that would \
generate 1-star reviews before they get published.

You are NOT generous. You are NOT encouraging. You assess each image as a paying \
buyer would when flipping through a "look inside" preview on Amazon. If a buyer \
would skip, frown at, or complain about a page, that page fails.

You respond with valid JSON only. No surrounding text. No markdown fences."""


USER_PROMPT_TEMPLATE = """Evaluate this AI-generated page for a "Bold and Easy" \
adult coloring book.

BOOK CONTEXT:
- Niche: {niche}
- Audience: {target_audience}
- Style: thick black outlines on white, no shading, no gray, no color, designed \
for stress-relief coloring with markers
- This is page #{sequence_num} of 50

WHAT THIS PAGE WAS SUPPOSED TO DEPICT:
"{subject}"

JUDGING RULES (read before scoring):
- Do not penalize the image for subject choice. Subjects are pre-curated in the \
niche YAML. Judge only how well the requested subject is rendered, never whether \
the subject "fits" the niche.
- Simplicity is a feature for the Bold & Easy style. Do not deduct for "too simple \
for adults" or "lacks complex detail" — these are intentional style choices, not \
defects.

EVALUATE ON FIVE CRITERIA (be strict — each criterion has a hard standard):

1. SUBJECT RECOGNITION (0-40 points)
   - 40: subject is immediately, unambiguously recognizable as "{subject}". A buyer \
flipping past at 1 page/second knows what it is.
   - 30: recognizable but with minor ambiguity (e.g., generic medical object that \
could be one of several things)
   - 20: requires close inspection to identify; viewer might guess wrong
   - 10: looks like a different object entirely (e.g., asked for "stethoscope", got \
something that looks like headphones or fishing equipment)
   - 0: unidentifiable abstract shapes

2. COMPOSITION COHERENCE (0-25 points)
   - 25: single coherent illustration, all elements clearly belong together, \
intentional layout
   - 18: mostly coherent but with one weak element (e.g., a stray line, a small \
disconnected piece, an awkward gap)
   - 10: multiple disconnected fragments, floating pieces, or items that should be \
connected but aren't
   - 5: looks like exploded-view diagram or scattered debris
   - 0: incoherent mess

3. LINE QUALITY (0-15 points)
   - 15: bold, confident, continuous outlines throughout (~6-10px equivalent), no \
broken lines
   - 10: mostly bold but with some thinner secondary lines or minor breaks
   - 5: noticeably thin or sketchy lines, multiple visible breaks, wavy/uncertain \
strokes
   - 0: hairline strokes, sketch-like quality, lines look uncertain or pencil-drawn

4. ANATOMY / ACCURACY (0-10 points)
   - For images with humans, animals, or recognizable objects with known structure:
   - 10: proportions correct, structure makes sense
   - 7: minor anatomical oddity that's not obviously wrong
   - 4: clearly wrong proportions or structure (extra limbs, melted features, parts \
in wrong places)
   - 0: anatomically broken (e.g., disembodied parts, conjoined elements, AI-melt \
artifacts)
   - For abstract/decorative subjects with no anatomy: award full 10 by default; \
deduct only for clear logical errors.

5. WHITESPACE BALANCE (0-10 points)
   - 10: subject fills ~60-80% of the page; comfortable whitespace around edges
   - 7: slightly small or slightly large but acceptable
   - 4: subject is microscopic (<25% of page) or crammed (>90% of page, touching \
edges)
   - 0: tiny floating object in vast whitespace, OR content clipped at edges

CRITICAL RED FLAGS (auto-fail with passed=false regardless of total score):
- Disembodied human/animal parts (floating hands, separated heads, etc.)
- Text or letters appearing in the image (the model should never produce text)
- Color or gray fill present (this is a coloring book — pages must be line-only)
- Subject is fundamentally different from what was requested
- AI-melt artifacts (warped faces, fingers merging, objects bleeding into each other)
- Distorted, melted, malformed, or clearly mismatched faces or eyes on any character. \
Includes: asymmetric features that look broken rather than stylized, structural melts \
(mouths where they shouldn't be, fused eyes, smeared facial features), and faces that \
don't match the character type (animal with human-melt features, etc.). Note: \
deliberate kawaii style often includes minor asymmetries (slight cheek size variance, \
highlight positioning) — these are NOT defects. Only flag when distortion is clearly \
broken, not when it's stylized.

OUTPUT FORMAT — respond with this JSON structure only:

{{
  "subject_recognition": <int 0-40>,
  "composition_coherence": <int 0-25>,
  "line_quality": <int 0-15>,
  "anatomy_accuracy": <int 0-10>,
  "whitespace_balance": <int 0-10>,
  "total_score": <int 0-100, sum of above>,
  "red_flags": [<list of red flag strings, empty if none>],
  "primary_failure": <one of: "subject", "composition", "lines", "anatomy", \
"none">,
  "issues": [<2-5 specific concrete problems, each one short phrase>],
  "prompt_hint": <one-sentence suggestion for prompt improvement, or null if \
passed>,
  "passed": <true if total_score >= 80 AND no red_flags, else false>
}}

Be honest. False positives (passing a bad image) cost us in 1-star reviews. False \
negatives (failing a fine image) cost us in regeneration API calls. The first is \
worse than the second."""

_SUBSCORE_KEYS = (
    "subject_recognition",
    "composition_coherence",
    "line_quality",
    "anatomy_accuracy",
    "whitespace_balance",
)


def _downsize_for_eval(image_path: Path) -> bytes:
    """Return the page as a downscaled square PNG, ready to upload to Claude."""
    with PILImage.open(image_path) as handle:
        # The pages are already square; thumbnail only ever shrinks.
        copy = handle.convert("RGB")
        copy.thumbnail((EVAL_IMAGE_SIZE, EVAL_IMAGE_SIZE), PILImage.Resampling.LANCZOS)
        buffer = io.BytesIO()
        copy.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()


def _strip_fence(text: str) -> str:
    """Drop a leading ```/```json fence and its closing ``` if present."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _map_failure_to_status(primary_failure: str, red_flags: list[str]) -> ImageQAStatus:
    """Map Claude's failure reason to the matching `rejected_vision_*` status."""
    if red_flags:
        joined = " ".join(red_flags).lower()
        anatomy_terms = ("disembodied", "melt", "anatomy", "distort", "face", "eye")
        if any(term in joined for term in anatomy_terms):
            return ImageQAStatus.REJECTED_VISION_ANATOMY
        if "subject" in joined:
            # "Subject is fundamentally different from what was requested."
            return ImageQAStatus.REJECTED_VISION_SUBJECT
        # Stray text, colour fill, scattered pieces — all composition faults.
        return ImageQAStatus.REJECTED_VISION_COMPOSITION

    mapping = {
        "subject": ImageQAStatus.REJECTED_VISION_SUBJECT,
        "composition": ImageQAStatus.REJECTED_VISION_COMPOSITION,
        "lines": ImageQAStatus.REJECTED_VISION_LINES,
        "anatomy": ImageQAStatus.REJECTED_VISION_ANATOMY,
        "none": ImageQAStatus.REJECTED_VISION_LOWSCORE,
    }
    return mapping.get(primary_failure, ImageQAStatus.REJECTED_VISION_LOWSCORE)


def _conservative_fail(reason: str, cost: Decimal, raw: str) -> VisionQAResult:
    """A fail verdict for an unparseable reply — regenerate rather than ship it."""
    return VisionQAResult(
        score=0,
        subscores={},
        passed=False,
        rejection_reason=ImageQAStatus.REJECTED_VISION_LOWSCORE,
        issues=[reason],
        prompt_hint=None,
        cost_usd=cost,
        raw_response=raw,
    )


async def evaluate_image(
    image_path: Path,
    *,
    subject: str,
    sequence_num: int,
    config: NicheConfig,
    provider: AnthropicProvider,
    pass_threshold: int = DEFAULT_PASS_THRESHOLD,
    book_id: UUID | None = None,
) -> VisionQAResult:
    """Grade one generated page with Claude Sonnet vision.

    Raises only on unrecoverable API errors (after the provider's own retries);
    a malformed reply is turned into a conservative fail, never an exception.
    The caller decides what to do with a rejection.
    """
    image_bytes = _downsize_for_eval(image_path)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        niche=config.niche,
        target_audience=config.book.target_audience,
        sequence_num=sequence_num,
        subject=subject,
    )
    result = await provider.generate_vision(
        prompt=user_prompt,
        image_bytes=image_bytes,
        system=SYSTEM_PROMPT,
        max_tokens=600,
        operation="vision_qa",
        model=VISION_QA_MODEL,
        book_id=book_id,
    )

    raw_text = _strip_fence(result.text)
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        logger.warning("vision QA reply was not valid JSON: {}", exc)
        return _conservative_fail(
            f"vision_qa_json_parse_error: {exc}", result.cost_usd, result.text
        )
    if not isinstance(parsed, dict):
        return _conservative_fail(
            f"vision_qa_unexpected_json: got {type(parsed).__name__}, expected object",
            result.cost_usd,
            result.text,
        )

    subscores = {key: int(parsed.get(key, 0)) for key in _SUBSCORE_KEYS}
    score = sum(subscores.values())  # the rubric total — not the model's arithmetic
    red_flags = [str(flag) for flag in (parsed.get("red_flags") or [])]
    issues = [str(issue) for issue in (parsed.get("issues") or [])]
    hint = parsed.get("prompt_hint")
    prompt_hint = str(hint) if hint else None

    passed = score >= pass_threshold and not red_flags
    rejection_reason = (
        None
        if passed
        else _map_failure_to_status(str(parsed.get("primary_failure", "none")), red_flags)
    )

    return VisionQAResult(
        score=score,
        subscores=subscores,
        passed=passed,
        rejection_reason=rejection_reason,
        issues=issues,
        prompt_hint=prompt_hint,
        cost_usd=result.cost_usd,
        raw_response=result.text,
    )


# Fixed score buckets for the `show-vision-qa` distribution, high to low.
_SCORE_BUCKETS: tuple[tuple[int, int, str], ...] = (
    (90, 100, "90-100"),
    (80, 89, "80-89"),
    (75, 79, "75-79"),
    (70, 74, "70-74"),
    (60, 69, "60-69"),
    (0, 59, "<60"),
)


def _subject_of(image: Image) -> str:
    """Recover the slot's subject from the stored generation params."""
    subject = image.generation_params.get("subject")
    return str(subject) if subject else image.prompt[:40]


def build_vision_qa_summary(slug: str, images: list[Image], threshold: int) -> str:
    """Render the `show-vision-qa` report: distribution, top issues, cost."""
    evaluated = [img for img in images if img.vision_qa_score is not None]
    if not evaluated:
        return f"Vision QA Summary — {slug}\nNo images have been vision-evaluated yet."

    scores = [int(img.vision_qa_score or 0) for img in evaluated]
    passed = [img for img in evaluated if img.qa_status == ImageQAStatus.PASSED]
    rejected = [img for img in evaluated if img.qa_status != ImageQAStatus.PASSED]
    average = sum(scores) / len(scores)
    pass_rate = 100.0 * len(passed) / len(evaluated)

    lines = [
        f"Vision QA Summary — {slug}",
        "=" * 41,
        f"Total images evaluated: {len(evaluated)}",
        f"Average score: {average:.1f}",
        f"Pass rate: {pass_rate:.0f}% ({len(passed)}/{len(evaluated)} passed; "
        f"threshold = {threshold})",
        "",
        "Score distribution:",
    ]
    for low, high, label in _SCORE_BUCKETS:
        count = sum(1 for score in scores if low <= score <= high)
        note = "passing" if low >= threshold else "regenerated"
        lines.append(f"  {label:<8}{'█' * count} {count}   ← {note}")

    issue_counts: dict[str, int] = {}
    for img in rejected:
        for issue in img.vision_qa_issues or []:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
    lines.append("")
    lines.append("Most common issues in rejected images:")
    if issue_counts:
        ranked = sorted(issue_counts.items(), key=lambda kv: kv[1], reverse=True)
        for rank, (issue, count) in enumerate(ranked[:5], start=1):
            lines.append(f'  {rank}. "{issue}" ({count})')
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("Bottom 5 PASSING pages (review manually):")
    weakest = sorted(passed, key=lambda img: int(img.vision_qa_score or 0))[:5]
    for img in weakest:
        lines.append(
            f'  seq {img.sequence_num:03d} (subject: "{_subject_of(img)}"): '
            f"score {img.vision_qa_score}"
        )

    total_cost = sum((img.vision_qa_cost_usd or Decimal(0) for img in evaluated), Decimal(0))
    lines.append("")
    lines.append(f"Total Vision QA cost: ${total_cost:.4f}")
    return "\n".join(lines)
