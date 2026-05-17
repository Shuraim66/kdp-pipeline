# Vision QA Stage — Pipeline Integration Spec

## Context for Claude Code

You're adding a Vision QA stage to the existing KDP coloring book pipeline. This sits between the existing pixel-level QA (`src/qa/image_qa.py`) and the copy-to-filtered/ step.

**Purpose:** The existing pixel QA only catches technical failures (resolution, grayscale, gray-pixel %, margin clipping). It cannot see semantic failures: incorrect subjects, incoherent compositions, broken anatomy, disconnected line fragments, ambiguous shapes. These semantic failures are the actual reason our generated pages would get 1-star reviews on Amazon.

Vision QA uses Claude Sonnet 4 to grade each generated image like a human reviewer would. Low-scoring images are queued for regeneration (reusing the existing `retry_of_image_id` mechanism).

**You do NOT remove or modify the existing pixel QA.** Both stages run; pixel QA first (cheap, fast), Vision QA second (more expensive, but only on images that already passed pixel QA — no point evaluating broken pixel grids).

---

## Architecture

```
Flux dev generates image (Phase 5)
       ↓
Pixel QA (existing — src/qa/image_qa.py)
       ↓
[if pixel QA fails] → mark rejected_*, queue regen
[if pixel QA passes] → continue ↓
       ↓
Vision QA (new — src/qa/vision_qa.py)
       ↓
[if vision QA fails] → mark rejected_vision_<reason>, queue regen
[if vision QA passes] → copy to filtered/, mark passed
```

Regeneration uses the existing `retry_of_image_id` chain. Max retries per slot stays at `qa.max_retries_per_slot` (currently 3).

---

## Database schema changes

Add to the `image_qa_status` enum:
- `rejected_vision_subject` — wrong/unrecognizable subject
- `rejected_vision_composition` — incoherent or fragmented composition
- `rejected_vision_lines` — broken, sketchy, or thin lines despite passing pixel checks
- `rejected_vision_anatomy` — anatomy/proportion failure
- `rejected_vision_lowscore` — overall score below threshold but no single dominant failure

Add columns to the `images` table:

```sql
ALTER TABLE images
  ADD COLUMN vision_qa_score INT,                -- 0-100, null if not yet evaluated
  ADD COLUMN vision_qa_subcores JSONB,           -- {subject_recognition, composition, lines, anatomy, whitespace}
  ADD COLUMN vision_qa_issues TEXT[],            -- specific problems identified
  ADD COLUMN vision_qa_prompt_hint TEXT,         -- suggested prompt adjustment from Claude
  ADD COLUMN vision_qa_model TEXT,               -- e.g., 'claude-sonnet-4-7'
  ADD COLUMN vision_qa_cost_usd DECIMAL(8,5),
  ADD COLUMN vision_qa_evaluated_at TIMESTAMPTZ;

CREATE INDEX idx_images_vision_score ON images(book_id, vision_qa_score)
  WHERE vision_qa_score IS NOT NULL;
```

Write an Alembic migration for this. The migration is reversible.

---

## New module: `src/qa/vision_qa.py`

```python
"""Vision-based QA for generated coloring book images.

Uses Claude Sonnet 4 to grade images on semantic correctness:
subject recognition, composition coherence, line quality, anatomy,
whitespace balance. Returns a structured score and specific issues.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from PIL import Image

from src.providers.anthropic import AnthropicClient
from src.config.schema import NicheConfig


VISION_QA_MODEL = "claude-sonnet-4-7"
DEFAULT_PASS_THRESHOLD = 75  # tune after first book
EVAL_IMAGE_SIZE = 1024       # downsize before sending to Claude


@dataclass(frozen=True)
class VisionQAResult:
    score: int                          # 0-100 (sum of subcores)
    subscores: dict[str, int]
    passed: bool
    rejection_reason: str | None        # one of the image_qa_status enum values
    issues: list[str]
    prompt_hint: str | None
    cost_usd: Decimal
    raw_response: str                   # for audit


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
  "passed": <true if total_score >= 75 AND no red_flags, else false>
}}

Be honest. False positives (passing a bad image) cost us in 1-star reviews. False \
negatives (failing a fine image) cost us in regeneration API calls. The first is \
worse than the second."""


def _downsize_for_eval(image_path: Path) -> bytes:
    """Resize to EVAL_IMAGE_SIZE × EVAL_IMAGE_SIZE PNG bytes."""
    with Image.open(image_path) as img:
        # Preserve aspect ratio with thumbnail (it's already square in our case)
        img.thumbnail((EVAL_IMAGE_SIZE, EVAL_IMAGE_SIZE), Image.LANCZOS)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG", optimize=True)
        return buf.getvalue()


def _map_failure_to_status(primary_failure: str, red_flags: list[str]) -> str:
    """Map Claude's failure reason to our image_qa_status enum."""
    if red_flags:
        # Red flags usually indicate composition or anatomy issues
        if any("disembodied" in f.lower() or "melt" in f.lower() for f in red_flags):
            return "rejected_vision_anatomy"
        if any("color" in f.lower() or "gray" in f.lower() for f in red_flags):
            # This should have been caught by pixel QA but failsafe here
            return "rejected_vision_composition"
        if any("text" in f.lower() or "letter" in f.lower() for f in red_flags):
            return "rejected_vision_composition"
        return "rejected_vision_composition"

    mapping = {
        "subject": "rejected_vision_subject",
        "composition": "rejected_vision_composition",
        "lines": "rejected_vision_lines",
        "anatomy": "rejected_vision_anatomy",
        "none": "rejected_vision_lowscore",
    }
    return mapping.get(primary_failure, "rejected_vision_lowscore")


async def evaluate_image(
    image_path: Path,
    subject: str,
    sequence_num: int,
    config: NicheConfig,
    client: AnthropicClient,
    pass_threshold: int = DEFAULT_PASS_THRESHOLD,
) -> VisionQAResult:
    """Evaluate one image with Claude Sonnet 4 vision.

    Raises on API errors (let tenacity in the provider handle retries).
    Returns a structured result; caller decides what to do with rejections.
    """
    image_bytes = _downsize_for_eval(image_path)
    image_b64 = base64.standard_b64encode(image_bytes).decode("ascii")

    user_prompt = USER_PROMPT_TEMPLATE.format(
        niche=config.niche,
        target_audience=config.book.target_audience,
        sequence_num=sequence_num,
        subject=subject,
    )

    response = await client.create_message(
        model=VISION_QA_MODEL,
        system=SYSTEM_PROMPT,
        max_tokens=600,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": user_prompt},
                ],
            }
        ],
        operation="vision_qa",
    )

    raw_text = response.content[0].text.strip()
    # Defensive: strip accidental ```json fences
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        # Model returned malformed JSON. Fail the image conservatively — better
        # to regenerate than to accept a possibly-bad page.
        return VisionQAResult(
            score=0,
            subscores={},
            passed=False,
            rejection_reason="rejected_vision_lowscore",
            issues=[f"vision_qa_json_parse_error: {e}"],
            prompt_hint=None,
            cost_usd=response.cost_usd,
            raw_response=raw_text,
        )

    subscores = {
        "subject_recognition": int(parsed.get("subject_recognition", 0)),
        "composition_coherence": int(parsed.get("composition_coherence", 0)),
        "line_quality": int(parsed.get("line_quality", 0)),
        "anatomy_accuracy": int(parsed.get("anatomy_accuracy", 0)),
        "whitespace_balance": int(parsed.get("whitespace_balance", 0)),
    }
    total_score = int(parsed.get("total_score", sum(subscores.values())))
    red_flags = parsed.get("red_flags") or []
    passed = bool(parsed.get("passed", False)) and total_score >= pass_threshold and not red_flags

    rejection_reason = None
    if not passed:
        rejection_reason = _map_failure_to_status(
            parsed.get("primary_failure", "none"),
            red_flags,
        )

    return VisionQAResult(
        score=total_score,
        subscores=subscores,
        passed=passed,
        rejection_reason=rejection_reason,
        issues=list(parsed.get("issues", [])),
        prompt_hint=parsed.get("prompt_hint"),
        cost_usd=response.cost_usd,
        raw_response=raw_text,
    )
```

---

## Integration with the existing pipeline

In `src/generators/images.py` (or wherever the QA orchestration lives), after pixel QA passes:

```python
# Existing pixel QA already ran and marked qa_status='passed' tentatively.
# Now run vision QA on top.

vision_result = await evaluate_image(
    image_path=image.local_path,
    subject=subject,
    sequence_num=image.sequence_num,
    config=config,
    client=anthropic_client,
)

# Update the images row with vision QA fields (regardless of pass/fail)
await images_repo.update_vision_qa(
    image_id=image.id,
    score=vision_result.score,
    subscores=vision_result.subscores,
    issues=vision_result.issues,
    prompt_hint=vision_result.prompt_hint,
    model=VISION_QA_MODEL,
    cost_usd=vision_result.cost_usd,
)

if not vision_result.passed:
    # Override the qa_status — vision QA gets the final say
    await images_repo.update_qa_status(
        image_id=image.id,
        status=vision_result.rejection_reason,
    )
    # Existing retry logic kicks in on the next plan-stage pass
else:
    # Already marked passed by pixel QA; copy to filtered/
    await copy_to_filtered(image)
```

The retry loop in the existing plan-stage doesn't change. It already looks at images with rejection statuses and regenerates them. It just needs to recognize the new `rejected_vision_*` statuses as retryable.

---

## CLI additions

### `show-vision-qa <slug>`

Print score distribution and worst pages for manual review:

```
Vision QA Summary — nurses_bold_easy_v1
=========================================
Total images evaluated: 67 (50 final + 17 retries)
Average score: 81.4
Pass rate: 74% (50/67 passed; threshold = 75)

Score distribution:
  90-100: ████████████ 12
  80-89:  ███████████████████ 19
  75-79:  ██████████████ 14   ← passing
  70-74:  ████ 4              ← regenerated
  60-69:  ████████ 8          ← regenerated
  <60:    ██████████ 10       ← regenerated

Most common issues in rejected images:
  1. "subject not recognizable" (8 occurrences)
  2. "broken outlines" (6)
  3. "floating disconnected pieces" (5)
  4. "tiny subject lost in whitespace" (3)

Bottom 5 PASSING pages (review manually):
  seq 14 (subject: "nurse cap with stars"): score 76
  seq 22 (subject: "thermometer with face"): score 76
  seq 31 (subject: "pill bottle smiling"): score 77
  ...

Total Vision QA cost: $0.34
```

### `tune-vision-threshold <slug> <new_threshold>`

After looking at the data, the user may want to retroactively adjust the pass threshold. This doesn't regenerate; it just re-applies pass/fail to the recorded scores. Useful for iterating without burning more API calls.

---

## Cost expectations

- **Per image evaluation**: ~$0.005 (Sonnet 4, ~$0.003 input + ~$0.002 output, with image at 1024×1024)
- **Per book** (50 final images, average 1.5 attempts per slot before pass): ~75 evaluations × $0.005 = **$0.375**
- **Total book cost** with Flux dev + dilation + vision QA: ~$2.50

Trivial vs the time saved on manual review and the value of not getting 1-star-reviewed.

---

## Tuning notes

- `DEFAULT_PASS_THRESHOLD = 75` is the starting point. After the first book run, look at the `show-vision-qa` output and adjust:
  - If most rejected images look fine to you when reviewed manually → lower to 70
  - If passing images still look weak → raise to 80
- The `prompt_hint` field from Claude is your free prompt-engineering feedback. After a book, query `SELECT vision_qa_prompt_hint, COUNT(*) FROM images WHERE book_id = ? AND vision_qa_prompt_hint IS NOT NULL GROUP BY vision_qa_prompt_hint ORDER BY 2 DESC` and you'll see common suggestions. Roll those into your master prompt template.

---

## What this does not do

- Vision QA does not catch every problem. Some bad images will pass; some good ones will fail. It's a strong filter, not a perfect one. Always manually review the final 50 before PDF assembly.
- Vision QA does not regenerate or modify images. It only scores them. The existing regeneration loop handles actual regeneration.
- Vision QA does not replace pixel QA. Pixel QA is much faster ($0) and catches a different class of failures (resolution, grayscale, gray pct). Both stages are necessary.

---

## Acceptance

- New migration adds the columns and enum values cleanly
- `evaluate_image()` returns a valid VisionQAResult for a real generated image (test with one of the existing v4 test images)
- A book run that previously had 60% pass rate now produces 50 vision-validated images with all rejected ones tied back to specific issue strings in the DB
- `show-vision-qa <slug>` runs and prints the formatted summary above
- Total vision QA cost for one book is under $0.50

---

## Build order

1. Migration first (schema + enum values)
2. `src/qa/vision_qa.py` module (the eval logic)
3. Integration into the existing generator/QA flow
4. CLI commands
5. Test on the existing 5 v4 test images first to validate the scoring is sensible (the fish hook should score low on subject recognition; the kawaii heart and teddy should score high)
6. Once validated, run a full 50-image book with Vision QA enabled

Test against the v4 images first — that gives us calibration data on whether the threshold is set right before committing API spend to a full book.