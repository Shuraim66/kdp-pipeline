"""Deterministic image-prompt construction for Fal.ai generation."""

from __future__ import annotations

from src.config.schema import NicheConfig

# Universal line-quality negatives, appended to every niche's own negative
# prompt. They target the diffusion model's observed failure modes — thin,
# sketchy, broken strokes — independent of subject.
#
# Caveat: FLUX schnell is guidance-distilled and does not honour negative
# prompts, and the Fal provider does not forward them to the API anyway. The
# positive prompt below carries the real weight; the negative is kept for the
# record and for a possible future switch to a guidance model (e.g. flux/dev).
_LINE_QUALITY_NEGATIVES = (
    "thin lines, hairline strokes, sketchy, pencil sketch, broken lines, "
    "gaps in lines, disconnected pieces, incomplete shapes, floating artifacts"
)


def build_image_prompt(config: NicheConfig, subject: str, variation_idx: int) -> tuple[str, str]:
    """Build the ``(prompt, negative_prompt)`` pair for one image slot.

    Deterministic given its inputs. `variation_idx` selects a composition
    modifier, cycling through `composition_modifiers` so each variation of a
    subject is framed differently.

    The prompt front-loads the line-weight directive — diffusion models weight
    leading tokens most heavily — and states an explicit stroke width, since
    schnell ignores the negative prompt. The literal words "coloring book" are
    kept out: schnell renders them as garbled title text across the page.
    """
    modifiers = config.composition_modifiers
    modifier = modifiers[variation_idx % len(modifiers)]
    prompt = " ".join(
        [
            "thick black outlines",
            config.style.art_style.strip(),
            "in a cute simple children's cartoon style",
            subject,
            modifier,
            "complete object, fully connected outlines, no broken lines, "
            "no disconnected pieces, single coherent illustration",
            "isolated on pure white background",
            "thick continuous black lines 4-6 pixels wide",
            f"{config.style.line_weight} black lines",
            "simple black and white line drawing",
            "vector style, professional illustration",
            "clean composition with margin around edges",
        ]
    )
    negative_prompt = f"{config.style.negative_prompts.strip()}, {_LINE_QUALITY_NEGATIVES}"
    return prompt, negative_prompt
